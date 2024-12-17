# Cumulative for the LangSmith



<details><summary>Learning Objectives</summary>

# Learning Objectives for the LangSmith topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use Langsmith to build production-grade LLM applications
- debug, test, evaluate, and monitor chains with Langsmith
</details>
<details><summary>Description</summary>

# Description of the LangSmith topic.

### LangSmith
LangSmith is a platform for building production-grade LLM applications.

You can debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and integrate with LangChain. LangSmith was developed by LangChain.

</details>
<details><summary>Implementation</summary> 

# Implementation for the LangSmith topic

### LangSmith
You can connect to LangSmith in a few steps:

1. Create a LangSmith account using one of the supported login methods.
2. Create an API Key.
3. Install the latest version LangChain.
```command
pip install -U langchain
```

4. Configure runtime environment:

Replace "<your-api-key>" with the API key generated in step 1
Replace "<your-openai-api-key>" with an OpenAI API Key from here
```command
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"

# The below examples use the OpenAI API, so you will need
export OPENAI_API_KEY=<your-openai-api-key>
```
5. Run the example code:
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
```
Your first run is now visible in LangSmith! Navigate to the projects page to view your "Hello, world!" trace.

Next Steps:
Read the [LangSmith Overview](https://docs.smith.langchain.com/overview) to learn more about what LangSmith.
</details>
