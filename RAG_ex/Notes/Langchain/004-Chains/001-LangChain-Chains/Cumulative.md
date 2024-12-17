# Cumulative for the LLMChain
<details><summary>Learning Objectives</summary>

# Learning Objectives for the LLMChain topic.

### Learning Objectives

After completing this module, associates should be able to:
- chain LLMs with Langchain
- Use Langchain Expression Language for chain composition
 
</details>
<details><summary>Description</summary>

# Description of the LLMChain topic.

Employing an LLM in solitary situations suits straightforward applications, yet for more intricate scenarios, it becomes essential to interconnect LLMs, either with each other or with additional components.

LangChain offers two overarching frameworks for "chaining" components. The legacy method involves utilizing the Chain interface, while the updated approach incorporates the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/). When constructing new applications, we generally employ LCEL for chain composition. However, there is support for several practical, pre-existing Chains. It's worth noting that Chains can also seamlessly integrate within LCEL, rendering the two approaches not mutually exclusive.

</details>
<details><summary>Implementation</summary> 

# Implementation for the LLMChain topic

### LLMChain
The primary facet of LCEL that stands out is its provision of an intuitive and easily comprehensible syntax for composition. Beyond this visibility, it offers robust support for essential functionalities such as:

- streaming,
- asynchronous calls,
- batching,
- parallelization,
- retries,
- fallbacks,
- tracing,

To illustrate this, let's examine a straightforward but common example of combining a prompt, model, and output parser:
and more.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a well trained scientist who provides correct and expressive answers to scientific questions.",
        ),
        ("human", "{question}"),
    ]
)
runnable = prompt | model | StrOutputParser()


for chunk in runnable.stream({"question": "What was the most important scientific discovery of the 20th century?"}):
    print(chunk, end="", flush=True)
```
</details>
