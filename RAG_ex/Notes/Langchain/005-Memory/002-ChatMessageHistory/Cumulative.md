# Cumulative for the ChatMessageHistory


<details><summary>Learning Objectives</summary>

# Learning Objectives for the ChatMessageHistory topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use ChatMessageHistory class
- Utilize convenience methods for saving HumanMessages, AIMessages, and then fetching them all
</details>
<details><summary>Description</summary>

# Description of the ChatMessageHistory topic.

### ChatMessageHistory

An integral utility class that forms the foundation of many memory modules is the ChatMessageHistory class. This class serves as an exceptionally lightweight wrapper, offering convenient methods for storing HumanMessages, AIMessages, and retrieving all of them.

Direct utilization of this class might be suitable if you are overseeing memory management independently of a chain.

</details>
<details><summary>Implementation</summary> 

# Implementation for the ChatMessageHistory topic

### ChatMessageHistory


```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hello!")

history.add_ai_message("how are you?")

history.messages
```
</details>
