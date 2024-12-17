# Cumulative for MathChain
<details><summary>Learning Objectives</summary>

# Learning Objectives for the MathChain topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use MathChain to solve complex math word problems
- Integrate math solving of word problems with LLMs and Python
 
</details>
<details><summary>Description</summary>

# Description of the MathChain topic.

### MathChain
Chain that interprets a prompt and executes Python code for mathematical operations.

</details>
<details><summary>Implementation</summary> 

# Implementation for the MathChain topic

### MathChain
We can use LLMs and Python to do complex word math problems.


```python
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
llm_math = LLMMathChain.from_llm(OpenAI())

llm_math.run("What is 16 raised to the .5 power?")
```
</details>
