import json
from pathlib import Path
from typing import List, Any, Union
from collections import Counter


class LaveBase:

    def compute(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError
    
    def batch_compute(self, *args: Any, **kwargs: Any) -> List[float]:
        raise NotImplementedError


class LaveICLBase(LaveBase):

    def __init__(
        self,
        num_shots: int = 8,
        rationalize: bool = True,
        filter_refs: bool = True,
        use_caption: bool = False,
        demos_file: Union[str, Path] = 'data/lave_demos.json',
        binary_demos_file: Union[str, Path] = 'data/lave_demos_binary.json',
        debug: bool = False
    ) -> None:
        super().__init__()
        self.num_shots = num_shots
        self.rationalize = rationalize
        self.filter_refs = filter_refs
        self.use_caption = use_caption
        self.debug = debug

        if demos_file is not None:
            with open(demos_file) as f:
                self.demos = json.load(f)
        if binary_demos_file is not None:
            with open(binary_demos_file) as f:
                self.binary_demos = json.load(f)

    @property
    def task_definition(self) -> str:
        text = ""
        if self.use_caption:
            text += ("You are given an image description, a question about the image, a set of gold-standard reference answers written by experts, and a candidate answer. "
                     "Please rate the accuracy of the candidate answer for the question considering the reference answers and the image description. "
                     "In case of discrepancy, prioritize the reference answers.")
        else:
            text += ("You are given a question, a set of gold-standard reference answers written by experts, and a candidate answer. "
                     "Please rate the accuracy of the candidate answer for the question considering the reference answers.")
        text += " Use a scale of 1-3, with 1 indicating an incorrect or irrelevant answer, 2 indicating an ambiguous or incomplete answer, and 3 indicating a correct answer."
        if self.rationalize:
            text += " Give the rationale before rating."
        text += "\n\nTHIS IS VERY IMPORTANT: A binary question should only be answered with 'yes' or 'no', otherwise the candidate answer is incorrect (rating=1)."
        return text

    @property
    def example_template(self) -> str:
        text = ""
        if self.use_caption:
            text += "Image description: '{caption}'\n"
        text += "Question: '{question}'\nReference answers: {references}\nCandidate answer: '{prediction}'\nOutput: {output}"
        return text

    @staticmethod
    def filter_references(references: List[str], p: float = 0.25) -> List[str]:
        c = Counter(references)
        max_v = max(c.values())
        outliers = {k for k, v in c.items() if (v / max_v) <= p}
        return [r for r in references if r not in outliers]
    
    def format_references(self, references: List[str], filter: bool = True) -> str:
        if filter:
            references = self.filter_references(references)
        return ', '.join([f"'{ref}'" for ref in sorted(references)])
    
    @staticmethod
    def is_binary_question(question: str, references: List[str]) -> bool:
        return Counter(references).most_common(1)[0][0].lower().rstrip('.') in ['yes', 'no']

    def select_demos(self, question: str, references: List[str]) -> List[dict]:
        if self.num_shots == 0:
            return []
        binary_question = self.is_binary_question(question, references)
        demos = self.binary_demos if binary_question else self.demos
        demos = demos[:self.num_shots]
        return demos

    def build_prompt(self, prediction: str, references: List[str], question: str, caption: str = None) -> str:
        prompt = self.task_definition + "\n\n"

        demos = self.select_demos(question, references)
        for demo in demos:
            kwargs = {
                'question': demo['question'],
                'references': self.format_references(demo['references'], filter=False),
                'prediction': demo['prediction'],
                'output': f"{demo['explanation']} So rating={demo['output']}" if self.rationalize else demo['output']
            }
            if self.use_caption:
                kwargs['caption'] = demo['caption']
            prompt += self.example_template.format(**kwargs) + "\n\n"

        kwargs = {
            'question': question,
            'references': self.format_references(references, filter=self.filter_refs),
            'prediction': prediction,
            'output': ''
        }
        if self.use_caption:
            kwargs['caption'] = caption.strip()
        prompt += self.example_template.format(**kwargs)

        prompt = prompt.strip()

        return prompt


class LaveSFTBase(LaveBase):

    @staticmethod
    def get_prompt(question: str, references: List[str], prediction: str) -> str:
        return f"### Question: {question}\n### References: {', '.join(references)}\n### Prediction: {prediction}\n### Judgment:"

    @staticmethod
    def get_response_template() -> str:
        return "\n### Judgment:"

    @staticmethod
    def formatting_prompts_func(example):
        texts = []
        for i in range(len(list(example.values())[0])):
            prompt = LaveSFTBase.get_prompt(example["question"][i], example["references"][i], example["prediction"][i])
            score = LaveSFTBase.float2int(example["human_score"][i])
            label = LaveSFTBase.int2label(score)
            text = f"{prompt} {label}"
            texts.append(text)
        return texts
    
    @staticmethod
    def float2int(score: float) -> int:
        return int(score * 2 + 1)
    
    @staticmethod
    def int2float(score: int) -> float:
        return (score - 1) / 2
    
    @staticmethod
    def int2label(score: int) -> str:
        return {1: "incorrect", 2: "ambiguous", 3: "correct"}[score]
    
    @staticmethod
    def label2int(label: str) -> int:
        return {"incorrect": 1, "ambiguous": 2, "correct": 3}[label]
