# Cumulative for the Custom Tools



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Custom Tools topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use the Tool dataclass to accept a single query string and return a string output
- Subclass the BaseTool class accept a single query string and return a string output
</details>
<details><summary>Description</summary>

# Description of the Custom Tools topic.

### Custom Tools
When developing your custom agent, you must furnish it with a roster of Tools at its disposal. Beyond the underlying function that gets invoked, the Tool comprises various elements:

- name (str): obligatory and must maintain uniqueness within the set of tools assigned to an agent
- description (str): discretionary but advisable, as the agent relies on it to discern tool functionality
- return_direct (bool): defaults to False
- args_schema (Pydantic BaseModel): voluntary but recommended, serving to offer additional insights (e.g., few-shot examples) or validation for anticipated parameters.


```python
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import SerpAPIWrapper

llm = ChatOpenAI(temperature=0)
```
</details>
<details><summary>Implementation</summary> 

# Implementation for the Custom Tools topic

### Custom Tools
The simplest tools accept a single query string and return a string output. 

There are two ways to do this: either by using the Tool dataclass, or by subclassing the BaseTool class.

The 'Tool' dataclass is designed to handle a single string input and yield a string as output.
```python
# Load the tool configs that are needed.
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful search for answering generic questions",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
]
```
Define a custom args_schema for more information about inputs.

```python
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    question: str = Field()


tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="solve math problems",
        args_schema=CalculatorInput,
    )
)

# using the default agent here.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "Who is Robert Deniro? What is his current age mutiplied by 2?"
)
```
An alternative option is to create a direct subclass of BaseTool. This proves advantageous when seeking enhanced control over instance variables or when intending to extend callbacks to nested chains or other tools.
```python
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class CustomSearchTool(BaseTool):
    name = "another_search"
    description = "useful for generic seraching"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """useful tool."""
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("another_search does not support async")


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """useful tool."""
        return llm_math_chain.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")

tools = [CustomSearchTool(), CustomCalculatorTool()]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What age is Robert Deniro? What is his age multiplied by 2?"
)
```
An @tool decorator can be used to create a Tool from a simple function. The decorator employs the function name as the tool name; however, this can be altered by providing a string as the initial argument. The decorator adopts the function's docstring as the description for the tool.
```python
from langchain.tools import tool


@tool
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return f"Results for query {query}"


search_api
```
Use arguments like the tool name and whether to return directly.
```python
@tool("search", return_direct=True)
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"

search_api
```


Use args_schema to for information about the argument.
```python
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")


@tool("search", return_direct=True, args_schema=SearchInput)
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"
```
</details>
