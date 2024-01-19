# LAVE (LLM-Assisted VQA Evaluation)

This repository contains the official implementation of **LAVE**, a new metric for automatic VQA evaluation using LLMs, as described in the paper:

[Improving Automatic VQA Evaluation Using Large Language Models](https://arxiv.org/abs/2310.02567). [Oscar Mañas](https://oscmansan.github.io), [Benno Krojer](https://bennokrojer.github.io), [Aishwarya Agrawal](https://www.iro.umontreal.ca/~agrawal/). Accepted at AAAI 2024

## Usage

```python
from lave_ft5 import LaveFT5

metric = LaveFT5()

lave_score = metric.compute(
    prediction="white and red",
    references=["red white blue", "white, red, blue", "white, red, and blue", "white red blue black"],
    question="What color is the plane?"
)
print(lave_score)
```

## Citation

If you find our project useful in your research, please cite the following paper:

```bibtex
@article{manas2023improving,
  title={Improving automatic vqa evaluation using large language models},
  author={Ma{\~n}as, Oscar and Krojer, Benno and Agrawal, Aishwarya},
  journal={arXiv preprint arXiv:2310.02567},
  year={2023}
}
```