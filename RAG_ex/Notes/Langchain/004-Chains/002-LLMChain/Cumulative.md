# Cumulative for the LLMChain
<details><summary>Learning Objectives</summary>

# Learning Objectives for the LLMChain topic.

### Learning Objectives

After completing this module, associates should be able to:
- chain LLMs with LLMchain
- Use LLMChain for chain composition
 
</details>
<details><summary>Description</summary>

# Description of the LLMChain topic.

### LLMChain
Chains serve as the legacy interface for interconnected applications. Broadly speaking, we define a Chain as a sequence of calls to components, encompassing the potential inclusion of other chains.



</details>
<details><summary>Implementation</summary> 

# Implementation for the LLMChain topic

### LLMChain
The simple base interface:
```python
class Chain(BaseModel, ABC):
    """Base interface implemented by chains."""

    memory: BaseMemory
    callbacks: Callbacks

    def __call__(
        self,
        inputs: Any,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
    ) -> Dict[str, Any]:
        ...

We can replicate the LCEL runnable here using the builtin LLMChain:

from langchain.chains import LLMChain

chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())
chain.run(question="What is the most important scientific discovery of the 20th century?")
```
</details>
