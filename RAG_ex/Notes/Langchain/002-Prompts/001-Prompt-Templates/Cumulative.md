# Cumulative for the  Prompt Templates
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Prompt Templates topic.

### Learning Objectives

After completing this module, associates should be able to:
- Define Prompts
- Describe the methods of the Prompt Templates
- Use PromptTemplate and ChatPromptTemplate to implement the basic building block of LCEL, Runnable Interface
 
</details>
<details><summary>Description</summary>

# Description of the Prompt Templates topic.

### Prompt Templates
A prompt framework denotes a systematic approach for generating prompts, providing predefined formulas for crafting inputs for language models.

These templates may encompass guidelines, few-shot instances, and specific contextual details and queries suitable for a particular task.

LangChain offers utilities for the creation and manipulation of prompt frameworks.

LangChain is committed to developing templates that transcend model-specific constraints, facilitating the seamless reuse of templates across various language models.

Typically, language models anticipate prompts in the form of either a string or a list of chat messages. A prompt framework can include:

- directives for instructing the language model,
- a series of few-shot examples to enhance the language model's response,
- a query posed to the language model.
Illustratively:

Employ PromptFramework to establish a template for a string prompt.

By default, PromptFramework utilizes Python's str.format syntax for templating.
```python

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {content}."
)
prompt_template.format(adjective="scary", content="ghosts")
```

The template supports any number of variables, including no variables:
```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a story")
prompt_template.format()
```


</details>
<details><summary>Implementation</summary> 

# Implementation for the Prompt Templates topic

### Prompt Templates
The input to chat models is structured as a series of chat messages.

Every chat message is linked to content, and an extra parameter known as role. For instance, when using the OpenAI Chat Completions API, a chat message can be attributed to an AI assistant, a human, or a system role.

Develop a chat prompt blueprint in the following manner:
```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Jake", user_input="What is your name?")
```
ChatPromptTemplate.from_messages accepts a variety of message representations.

For example, in addition to using the 2-tuple representation of (type, content) used above, you could pass in an instance of MessagePromptTemplate or BaseMessage.
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to sound more uplifting"
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

llm = ChatOpenAI()
llm(chat_template.format_messages(text="i do not like frogs."))
```




