# Cumulative for the ConversationBufferWindowMemory



<details><summary>Learning Objectives</summary>

# Learning Objectives for the ConversationBufferWindowMemory topic.

### Learning Objectives

After completing this module, associates should be able to:
- Keeps a list of the interactions of the conversation over time.
- Limit this list to the last K interactions.
</details>
<details><summary>Description</summary>

# Description of the ConversationBufferWindowMemory topic.

### ConversationBufferWindowMemory
The ConversationBufferWindowMemory maintains a record of the conversation's interactions, focusing on the most recent K interactions. This approach proves valuable for establishing a sliding window that captures the latest interactions, preventing the buffer from becoming excessively large.

Here's a simple example:
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory( k=1)
memory.save_context({"input": "hello"}, {"output": "how are you"})
memory.save_context({"input": "good and you"}, {"output": "good"})

memory.load_memory_variables({})
```

We can retrieve the history in the form of a message list, a practical approach when integrating this with a chat model.
```python
memory = ConversationBufferWindowMemory( k=1, return_messages=True)
memory.save_context({"input": "hello"}, {"output": "how are you"})
memory.save_context({"input": "good and you"}, {"output": "good"})

memory.load_memory_variables({})
```

</details>
<details><summary>Implementation</summary> 

# Implementation for the ConversationBufferWindowMemory topic

### ConversationBufferWindowMemory

Let's go through an example:
```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    # We set a low k=3, keeping the last 3 interactions in memory
    memory=ConversationBufferWindowMemory(k=3),
    verbose=True
)
conversation_with_summary.predict(input="Hello, how are you?")

conversation_with_summary.predict(input="How are they doing?")

conversation_with_summary.predict(input="Are you well?")
```
</details>
