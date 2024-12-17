# Cumulative for the Using LLMs with LangChain example



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Using LLMs with LangChain example topic.

### Learning Objectives

After completing this module, associates should be able to:
- Combine a prompt template with an LLM
- Implement Langchain Expression Language (LCEL)
</details>
<details><summary>Description</summary>

# Description of the Using LLMs with LangChain example topic.

### Using LLMs with LangChain example
The prevalent form of linking in any LLM application involves merging a prompt template with an LLM, optionally incorporating an output parser.

The suggested approach for accomplishing this task is by utilizing the LangChain Expression Language. An alternative method is employing the traditional LLMChain, a singular class designed to streamline the composition of these three elements.
</details>
<details><summary>Implementation</summary> 

# Implementation for the Using LLMs with LangChain example topic

### Using LLMs with LangChain example
Use LCEL, where BasePromptTemplate, BaseLanguageModel, and BaseOutputParser all adhere to the Runnable interface. These components are purposefully crafted to seamlessly integrate with one another, simplifying the process of composing LCEL.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

prompt = PromptTemplate.from_template(
    "What would be an ideal name for a business specializing in {product}?"
)
runnable = prompt | ChatOpenAI() | StrOutputParser()
runnable.invoke({"product": "sporty shoes"})
```
</details>
