# Cumulative for the TransformChain examples



<details><summary>Learning Objectives</summary>

# Learning Objectives for the TransformChain examples topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use LCEL to modify inputs during their passage as a transformation
- Filter and transform with LCEL
</details>
<details><summary>Description</summary>

# Description of the TransformChain examples topic.

### TransformChain examples

There is a desire to modify inputs during their passage from one component to the next, i.e. a transform. The legacy way is using the TransformChain, but the up to date way utilizes LCEL.
</details>
<details><summary>Implementation</summary> 

# Implementation for the TransformChain examples topic

### TransformChain examples
We take as input in a long text, then filter the text to only the first 3 paragraphs, and finally pass that into a chain for summarization.
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Summarize this text:

{output_text}

Summary:"""
)

with open("../../test_text.txt") as f:
    test_text = f.read()
```
We combine functions in any RunnableSequence:
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

runnable = (
    {"output_text": lambda text: "\n\n".join(text.split("\n\n")[:3])}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)
runnable.invoke(test_text)
```

</details>
