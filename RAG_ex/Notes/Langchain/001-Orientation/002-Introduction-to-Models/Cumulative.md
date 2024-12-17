# Cumulative for the Introduction to Models



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Introduction to Models topic.

### Learning Objectives

After completing this module, associates should be able to:
- Understand the basics of LLMs
- Understand the two basic types of models: LLMs and Chat Models
</details>
<details><summary>Description</summary>

# Description of the Introduction to Models topic.

### Introduction to Models
LangChain serves as a framework tailored for interacting with language models, emphasizing that it is not a language model itself. In this installment of my series on LangChain, we delve into the functionalities of LangChain concerning various models.

An artificial intelligence model is a mathematical representation of data employed by an algorithm to mimic human(-like) behavior. It comes to life through the machine learning method of training on extensive data, acquiring the ability to discern patterns, make predictions, or execute specific tasks.

(Large) Language models (LLMs), exemplified by OpenAI's GPT (Generative Pre-trained Transformer) models, predominantly rely on deep learning architectures known as transformers. These transformers utilize self-attention mechanisms to grasp relationships between words within a sentence or text sequence, enabling a nuanced understanding of context and word meanings.

Notably, LangChain distinguishes itself as a model-agnostic framework. This implies that a seamless transition from one model, such as OpenAI's, to another, like AlephAlpha's, is possible without necessitating alterations to your code, given that the interface remains consistent. Naturally, the outcomes will vary depending on the chosen model.


</details>
<details><summary>Implementation</summary> 

# Implementation for the Introduction to Models topic

### Introduction to Models
LangChain offers the flexibility to engage with two distinct model types: LLMs and Chat Models. While LLMs essentially operate on a "text in — matching text out" principle, Chat Models are structured around chat messages. An example of a chat model is ChatGPT, which you may already be familiar with. The choice between the two depends on your specific use case.

Expanding its horizons, LangChain consistently incorporates new LLM providers. The following snapshot highlights a subset of presently supported providers:

- Aleph Alpha
- Azure
- OpenAI
- Cohere
- Hugging Face Hub, Local Pipelines, TextGen Inference

For an exhaustive and up-to-date list, please refer to the [docs](https://python.langchain.com/en/latest/reference/modules/llms.html).

When it comes to utilizing an LLM, the fundamental methods are through ```__call__``` and ```generate```. Let's explore how these operations unfold:

```python
import os
os.environ['OPENAI_API_KEY'] = "..." 
from langchain.llms import OpenAI
llm = OpenAI() # giving no llm will default to text-davinci-003
print(llm("Tell me a funny joke"))

llm_result = llm.generate(["Tell me a funny joke", "Tell me a place name with 'G'"]*5)
len(llm_result.generations)

print(llm_result.generations)

print(llm_result.llm_output)

```
Chat Models:
Within a chat model, the chat messages come with designated roles:

- AIMessage
- HumanMessage
- SystemMessage
- Chatmessage (capable of adopting arbitrary role parameters)
Once more, the functionalities of both ```__call__``` and ```generate``` can be employed. Let's explore how chat models manage inquiries:
```python
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
chat([HumanMessage(content="Tell me a funny joke.")])
```
</details>
