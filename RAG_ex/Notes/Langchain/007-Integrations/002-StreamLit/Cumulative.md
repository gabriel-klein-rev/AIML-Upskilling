# Cumulative for the StreamLit



<details><summary>Learning Objectives</summary>

# Learning Objectives for the StreamLit topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use StreamLit to build build and share data apps
- Create shareable LLM webapps with StreamLit
</details>
<details><summary>Description</summary>

# Description of the StreamLit topic.

### StreamLit
With Streamlit, you can build and share data apps. With Streamlit, you can turn data scripts into shareable web apps. It's All in pure Python, and no frontend experience is required. See more examples at [streamlit.io](https://streamlit.io/generative-ai).

We can use StreamlitCallbackHandler to display the actions of an agent in an interactive Streamlit app.
</details>
<details><summary>Implementation</summary> 

# Implementation for the StreamLit topic

### StreamLit
Provide a parent container to render the output.
```command
pip install streamlit
```
```python
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())
```
To visualize the actions of an Agent with Tools (or Agent Executor), you can create an agent in your Streamlit app and pass the StreamlitCallbackHandler to agent.run()

```python
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

llm = OpenAI(temperature=0, streaming=True)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
```
</details>
