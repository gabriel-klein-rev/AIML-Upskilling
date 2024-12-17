# Cumulative for the ConversationBufferMemory examples

<details><summary>Learning Objectives</summary>

# Learning Objectives for the ConversationBufferMemory examples topic.

### Learning Objectives

After completing this module, associates should be able to:
- Store messages and extract as a variable
- Use LCEL with ConversationBufferMemory
</details>
<details><summary>Description</summary>

# Description of the ConversationBufferMemory examples topic.

### ConversationBufferMemory examples
With Memory we can store messages and then extract the messages in a variable.
</details>
<details><summary>Implementation</summary> 

# Implementation for the ConversationBufferMemory examples topic

### ConversationBufferMemory We can first extract it as a string.
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})

memory.load_memory_variables({})
```

We also get the history as a list of messages.
```python
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hello"}, {"output": "how are you"})

memory.load_memory_variables({})
```
Using in a chain
Finally, let's take a look at using this in a chain.
```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI()
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="Hello!")


conversation.predict(input="I am well, and I am having a great conversation with AI.")
```
</details>
