# Cumulative for the ConversationSummaryMemory



<details><summary>Learning Objectives</summary>

# Learning Objectives for the ConversationSummaryMemory topic.

### Learning Objectives

After completing this module, associates should be able to:
- Create a summary of conversation over time.
- Inject the summary of the conversation into a chain.
</details>
<details><summary>Description</summary>

# Description of the ConversationSummaryMemory topic.

### ConversationSummaryMemory
Now, let's delve into the utilization of a slightly more intricate memory type - ConversationSummaryMemory. This memory variant generates a summary of the ongoing conversation, offering a condensed version of the information gathered over time. It proves beneficial for distilling key details from the conversation's progression. Conversation summary memory continuously updates and stores the current summary. Subsequently, this memory can be employed to inject the ongoing conversation's summary into a prompt or chain. This type of memory becomes particularly advantageous for extended conversations, where incorporating the entire past message history verbatim in the prompt would consume an excessive number of tokens.

To begin, let's explore the fundamental features of this memory type.
```python
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI

memory = ConversationSummaryMemory(llm=OpenAI())
memory.save_context({"input": "hello"}, {"output": "how are you"})

memory.load_memory_variables({})
```
Additionally, we have the option to retrieve the conversation history as a list of messages, great when integrating this memory type with a chat model.
```python
memory = ConversationSummaryMemory(llm=OpenAI(), return_messages=True)
memory.save_context({"input": "hello"}, {"output": "how are you"})

memory.load_memory_variables({})
```

Now utilize the predict_new_summary method.
```python
messages = memory.chat_memory.messages
previous_summary = ""
memory.predict_new_summary(messages, previous_summary)
```
</details>
<details><summary>Implementation</summary> 

# Implementation for the ConversationSummaryMemory topic

### ConversationSummaryMemory
Let's use this in a chain, again setting verbose=True so we can see the prompt.
```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
llm = OpenAI()
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=OpenAI()),
    verbose=True
)
conversation_with_summary.predict(input="Hello, how are you?")

conversation_with_summary.predict(input="I want to hear more!")

conversation_with_summary.predict(input="How interesting -- what else are you doing?")
```
</details>
