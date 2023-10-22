from typing import Any, List, Union

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import numpy as np

from lave import LaveBase


class LaveFT5(LaveBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = 'google/flan-t5-xxl'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
        self.model.eval().to(self.device)
    
    def generate(self, prompt: Union[str, List[str]], **generate_kwargs) -> torch.Tensor:
        encoding = self.tokenizer(prompt, padding='longest', return_tensors='pt').to(self.device)
        if self.debug:
            print(f"Prompt ({encoding.input_ids.size(1)}): {prompt}")

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=100,
                return_dict_in_generate=True,
                **generate_kwargs
            )
        output_ids = output.sequences
        if self.debug:
            print(f"Generated text ({output_ids.size(1)}): {self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)}")
        
        return output_ids

    def postprocess(self, output_ids: torch.Tensor) -> List[float]:
        generated_token_id = output_ids[
            torch.arange(output_ids.size(0), dtype=torch.long, device=self.device),
            (output_ids == self.tokenizer.eos_token_id).float().argmax(dim=1) - 1
        ]
        if self.debug:
            print(f"Token id: {generated_token_id.tolist()}")
        
        score_str = self.tokenizer.batch_decode(generated_token_id, clean_up_tokenization_spaces=True)
        for i in range(len(score_str)):
            if score_str[i] not in ['1', '2', '3']:
                if self.debug:
                    print(f"Unexpected output: {score_str[i]}")
                score_str[i] = '2'
        
        scores = np.array(score_str, dtype=float)
        scores = ((scores - 1.) / 2.).tolist()

        return scores
    
    def compute(
        self,
        prediction: str,
        references: List[str],
        question: str,
        caption: str = None
    ) -> float:
        prompt = self.build_prompt(prediction, references, question, caption)
        output_ids = self.generate(prompt)
        score = self.postprocess(output_ids)[0]
        return score
    
    def batch_compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        questions: List[str],
        captions: List[str] = None
    ) -> List[float]:
        if not self.use_caption:
            captions = [None] * len(predictions)
        prompts = [self.build_prompt(pred, refs, q, cap) for pred, refs, q, cap in zip(predictions, references, questions, captions)]
        output_ids = self.generate(prompts)
        scores = self.postprocess(output_ids)
        return scores
