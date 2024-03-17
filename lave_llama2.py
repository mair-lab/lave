from typing import List, Any, Union
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from lave import LaveSFTBase


class LaveLlama2(LaveSFTBase):

    def __init__(self, model_path: Union[str, Path], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
    
    def compute(
        self,
        prediction: str,
        references: List[str],
        question: str,
        caption: str = None
    ) -> float:
        prompt = self.get_prompt(question, references, prediction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids[:, :-1],
                attention_mask=inputs.attention_mask[:, :-1],
                max_new_tokens=3,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        score_str = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        score = self.int2float(self.label2int(score_str))
        return score
    
    def batch_compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        questions: List[str],
        captions: List[str] = None
    ) -> List[float]:
        prompts = [self.get_prompt(q, refs, pred) for pred, refs, q in zip(predictions, references, questions)]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, return_length=True, return_attention_mask=True).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids[:, :-1],
                attention_mask=inputs.attention_mask[:, :-1],
                max_new_tokens=3,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        score_strs = [s.strip() for s in self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)]
        scores = [self.int2float(self.label2int(s)) for s in score_strs]
        return scores
