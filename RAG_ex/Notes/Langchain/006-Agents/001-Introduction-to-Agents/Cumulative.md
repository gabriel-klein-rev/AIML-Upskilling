# Cumulative for the Introduction to Agents



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Introduction to Agents topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use LLMs to choose a sequence of actions to take
- Understand an LLM as a reasoning engine to determine which actions to take
</details>
<details><summary>Description</summary>

# Description of the Introduction to Agents topic.

### Introduction to Agents

The fundamental concept behind agents involves employing a language model to decide on a series of actions to execute. Unlike chains, where actions are pre-defined in the code, agents utilize a language model as a reasoning engine to deduce the appropriate actions and their sequence.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Introduction to Agents topic

### Introduction to Agents
The agent serves as the link responsible for determining the subsequent course of action. This process is fueled by a combination of a language model and a given prompt. The components feeding into this process include:

- Tools: Descriptions outlining the available tools.
- User input: The overarching objective provided at a high level.
- Intermediate steps: Any pair of action and tool previously executed to fulfill the user input.
The outcome consists of the next action(s) to be taken or the conclusive response to be conveyed to the user. An action specifies both the tool and its corresponding input.

Various agents exhibit distinct styles of prompting for reasoning, diverse methods of encoding inputs, and different approaches to parsing the output. Refer to the agent types for a comprehensive list of pre-built agents. Additionally, creating custom agents is a straightforward process.
</details>
