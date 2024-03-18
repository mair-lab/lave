import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from accelerate import Accelerator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from tqdm import trange
import numpy as np
from scipy.stats import spearmanr, kendalltau

from lave import LaveSFTBase


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    datasets = load_dataset("mair-lab/lave")
    train_ds = datasets["dev"].shuffle(seed=42)
    val_ds = datasets["test"]
    assert len(set(train_ds["qid"]) & set(val_ds["qid"])) == 0

    print(f"num train: {len(train_ds)} || num val: {len(val_ds)}")

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        attn_implementation="flash_attention_2"
    )

    model.train()
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=Path.home() / "scratch/tmp/sft/lave_llama2",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        max_steps=-1,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        bf16=True,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    response_template_ids = tokenizer.encode(LaveSFTBase.get_response_template(), add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=LaveSFTBase.formatting_prompts_func,
        peft_config=lora_config,
        max_seq_length=256
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    del model
    torch.cuda.empty_cache()
    if trainer.accelerator.is_main_process:
        # model = trainer.model
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().local_process_index},
            attn_implementation="flash_attention_2"
        )

        model.eval()
        tokenizer.padding_side = "left"

        predicted = []
        target = []
        errors = []
        batch_size = trainer.args.per_device_eval_batch_size * 4
        for i in trange(0, len(val_ds), batch_size):
            batch = val_ds[i:i + batch_size]
            prompts = [LaveSFTBase.get_prompt(batch["question"][j], batch["references"][j], batch["prediction"][j]) for j in range(len(list(batch.values())[0]))]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs.input_ids[:, :-1],
                    attention_mask=inputs.attention_mask[:, :-1],
                    max_new_tokens=2,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )
            scores = [l.strip() for l in tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]-1:], skip_special_tokens=True)]
            for j, score_str in enumerate(scores):
                try:
                    predicted.append(LaveSFTBase.int2float(LaveSFTBase.label2int(score_str)))
                    target.append(batch["human_score"][j])
                except:
                    errors.append(score_str)
        predicted = np.array(predicted)
        target = np.array(target)
        print(f"errors: {len(errors)}/{len(val_ds)}")

        print(f"accuracy: {(predicted == target).mean()}")
        print(f"spearman: {spearmanr(predicted, target)[0]}")
        print(f"kendall: {kendalltau(predicted, target)[0]}")
