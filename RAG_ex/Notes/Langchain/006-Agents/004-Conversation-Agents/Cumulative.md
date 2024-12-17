# Cumulative for the Conversation Agent



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Conversation Agents topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use agent optimized for conversation
- Create a conversation agent using LCEL
</details>
<details><summary>Description</summary>

# Description of the Conversation Agents topic.

### Conversation Agents
We illustrate the utilization of an agent fine-tuned for conversational interactions. Many agents are typically tailored for utilizing tools to deduce the optimal response, a scenario less suitable for conversational settings where interaction with the user is paramount.

In contrast to the conventional ReAct agent, the primary distinction lies in the prompt. Here, our aim is to cultivate a more conversational tone.
```python
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for a general search tool",
    ),
]


llm = OpenAI()
```

</details>
<details><summary>Implementation</summary> 

# Implementation for the Conversation Agents topic

### Conversation Agents
We show how to create this agent using LCEL
```python
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description

prompt = hub.pull("somegithub")

prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

llm_with_stop = llm.bind(stop=["\nObservation"])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)
```
```python
from langchain.agents import AgentExecutor

memory = ConversationBufferMemory(memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "hello, i am jon"})["output"]

agent_executor.invoke({"input": "what is my name?"})["output"]

```
</details>
