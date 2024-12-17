# Cumulative for the Agent Tools



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Agent Tools topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use tools as interface for agents
- Use tools as functions for agents
</details>
<details><summary>Description</summary>

# Description of the Agent Tools topic.

### Agent Tools

Tools are the interfaces through which an agent can interact.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Agent Tools topic

### Agent Tools
Tools are functions that agents leverage to interact. These utilities may take the form of generic tools, such as search functions, other chains, or even interactions with other agents.

Presently, loading tools can be achieved through the utilization of the following code snippet:
```python
from langchain.agents import load_tools
tool_names = [tool1, tool2]
tools = load_tools(tool_names)
```
Some tools (e.g. chains, agents) may require a base LLM to use to initialize them. In that case, you can pass in an LLM as well:
```python
from langchain.agents import load_tools
from langchain.llms import OpenAI

tool_names = [tool1,tool2]
llm = OpenAI()
tools = load_tools(tool_names, llm=llm)
```
</details>
