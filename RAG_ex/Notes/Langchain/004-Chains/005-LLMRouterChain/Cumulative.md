# Cumulative for the Router
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Router topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use Routers to use output of a previous step to define the next step
- provide structure and consistency around interactions with LLMs
 
</details>
<details><summary>Description</summary>

# Description of the Router topic.

### Router
Routing empowers the creation of non-deterministic chains, where the outcome of a preceding step dictates the subsequent step. This mechanism contributes to establishing a systematic and consistent framework for engaging with LLMs.

To illustrate, consider a scenario where we possess two templates tailored for distinct question types, aiming to select the appropriate template depending on the user input.

```python
from langchain.prompts import PromptTemplate

science_template = """You are a very smart science professor. \
You are great at answering questions about science in a factually correct and fluent manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""
science_prompt = PromptTemplate.from_template(science_template)

history_template = """You are a very good historian. You are great at answering history questions. 

Here is a question:
{input}"""
history_prompt = PromptTemplate.from_template(history_template)
```
</details>
<details><summary>Implementation</summary> 

# Implementation for the Router topic

### Router

Achieving this is straightforward through the utilization of a RunnableBranch. Initializing a RunnableBranch involves providing a list of (condition, runnable) pairs alongside a default runnable. This branch determines the appropriate route by evaluating each condition with the input it receives. It chooses the first condition that yields a True result and executes the corresponding runnable associated with that condition using the input.

In the event that none of the provided conditions align, the branch resorts to executing the default runnable.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

general_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
prompt_branch = RunnableBranch(
    (lambda x: x["topic"] == "history", history_prompt),
    (lambda x: x["topic"] == "science", science_prompt),
    general_prompt,
)
```
```python
from typing import Literal

from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


class TopicClassifier(BaseModel):
    "Classify the topic of the user question"

    topic: Literal["history", "science", "general"]
    "The topic of the user question. One of 'history', 'science' or 'general'."


classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm = ChatOpenAI().bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)
parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="topic"
)
classifier_chain = llm | parser
```
```python
from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
    | prompt_branch
    | ChatOpenAI()
    | StrOutputParser()
)

final_chain.invoke(
    {
        "input": "Who was the first Roman Emporer?"
    }
)
```
</details>
