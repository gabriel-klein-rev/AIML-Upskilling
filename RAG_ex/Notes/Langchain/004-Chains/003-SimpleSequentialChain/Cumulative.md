# Cumulative for the SimpleSequentialChain
<details><summary>Learning Objectives</summary>

# Learning Objectives for the SimpleSequentialChain topic.

### Learning Objectives

After completing this module, associates should be able to:
- connect multiple chains
- compose them into pipelines
 
</details>
<details><summary>Description</summary>

# Description of the SimpleSequentialChain topic.

### SimpleSequentialChain
Sequential chains provide the capability to link various chains together, crafting pipelines that execute specific scenarios. There exist two categories of sequential chains:

1. SimpleSequentialChainChain: The most straightforward version of sequential chains, featuring each step with a singular input/output relationship. The output of one step becomes the input for the next.

2. SequentialChain: A more versatile form of sequential chains, accommodating multiple inputs/outputs.

</details>
<details><summary>Implementation</summary> 

# Implementation for the SimpleSequentialChain topic

### SimpleSequentialChain
```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# LLMChain to generate a recipe based on a dish name
llm = OpenAI(temperature=0.7)
recipe_chain = LLMChain(llm=recipe_generator, prompt=recipe_prompt)

# LLMChain to compose a cooking tips article using the generated recipe
llm = OpenAI(temperature=0.7)
cooking_tips_chain = LLMChain(llm=cooking_tips_generator, prompt=cooking_tips_prompt)

# Overall chain running the two chains in sequence
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains=[recipe_chain, cooking_tips_chain], verbose=True)

cokking = overall_chain.run("shrimp gumbo")
```

</details>
