# LAVE (LLM-Assisted VQA Evaluation)

### Usage
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