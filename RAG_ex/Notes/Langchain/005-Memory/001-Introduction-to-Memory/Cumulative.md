# Cumulative for the Introduction to Memory
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Introduction to Memory topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use LLMs to implement conversational interface
- Maintain information about entities and their relationships with Memory
 
</details>
<details><summary>Description</summary>

# Description of the Introduction to Memory topic.

### Introduction to Memory
Many LLM applications incorporate a conversational interface, a crucial aspect of which involves referencing information from earlier parts of the conversation. At the very least, a conversational system should have the capability to directly access a certain window of past messages. For a more sophisticated system, the inclusion of a constantly updating world model becomes essential. This world model facilitates tasks such as maintaining information about entities and their relationships.

The term we attribute to this ability to retain information from past interactions is "memory." LangChain offers a plethora of utilities designed to enhance a system with memory. These utilities can be employed independently or seamlessly integrated into a chain.

A memory system essentially needs to support two fundamental actions: reading and writing. Every chain defines a core execution logic that anticipates specific inputs. While some inputs directly come from the user, others can be retrieved from memory. In a single run, a chain interacts with its memory system twice:

 - After receiving the initial user inputs but before executing the core logic, a chain reads from its memory system to augment the user inputs.
 - After executing the core logic but before returning the answer, a chain writes the inputs and outputs of the current run to memory, enabling future reference in subsequent runs.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Introduction to Memory topic

### Introduction to Memory
The fundamental considerations in designing any memory system revolve around:

1. How state is stored
2. How state is queried

At the core of any memory system lies a history of all chat interactions. Even if not all interactions are directly utilized, they must be stored in some manner. The LangChain memory module encompasses a range of integrations dedicated to storing these chat messages, spanning from in-memory lists to persistent databases.

- [Chat message storage](https://python.langchain.com/docs/modules/memory/chat_messages/): for insights into working with Chat Messages and the various integrations available.

Maintaining a list of chat messages is relatively straightforward. What poses a greater challenge is the development of data structures and algorithms built atop chat messages, providing a view of those messages that is most valuable.

A basic memory system might simply retrieve the most recent messages in each run. A slightly more intricate system could offer a concise summary of the last K messages. An even more sophisticated approach might involve extracting entities from stored messages and exclusively providing information about entities referenced in the current run.

Different applications may have varied requirements for how memory is queried. The memory module is designed to facilitate the initiation of simple memory systems while allowing users to develop custom systems if necessary.

- [Memory types](https://python.langchain.com/docs/modules/memory/types/): data structures and algorithms constituting the memory types supported by LangChain.

</details>
