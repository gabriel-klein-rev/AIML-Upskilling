# Cumulative for the Multi-Query Retrieval
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Multi-Query Retrieval topic.

### Learning Objectives

After completing this module, associates should be able to:
- Use MultiQueryRetriever to automate the process of prompt tuning
- Optimize Retrieval to generate multiple perspectives on the same question
 
</details>
<details><summary>Description</summary>

# Description of the Multi-Query Retrieval topic.

### Multi-Query Retrieval

The retrieval process in distance-based vector database retrieval involves embedding queries in a high-dimensional space and identifying similar embedded documents based on "distance." However, subtle changes in query wording or limitations in capturing semantic nuances by embeddings may yield different results. To address these challenges, prompt engineering or tuning is sometimes performed manually, which can be a laborious task.

The MultiQueryRetriever streamlines prompt tuning by automating the generation of multiple queries from various perspectives for a given user input query, utilizing an LLM in the process. For each query, it retrieves a set of pertinent documents and consolidates the unique union across all queries to assemble a more extensive collection of potentially relevant documents. By presenting multiple viewpoints on the same question, the MultiQueryRetriever aims to mitigate some of the drawbacks of distance-based retrieval, providing a more diverse set of results.


</details>
<details><summary>Implementation</summary> 

# Implementation for the Multi-Query Retrieval topic

### Multi-Query Retrieval
```python
# Build a sample vectorDB
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load blog post
loader = WebBaseLoader("https://amaarora.github.io/posts/2023-07-25-llmchain.html")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
```

Specify the LLM to use for query generation, and the retriever will do the rest.
```python
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

question = "What are the approaches Langchain uses for text generation?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

# logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)
```

You also have the option to provide a prompt alongside an output parser to segment the outcomes into a list of queries.

```python
from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


# segment the outcomes into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""As an AI language model assistant, your objective is to produce three
    distinct variations of the provided user question for retrieving pertinent documents from a vector
    database. Your aim is to assist the user in overcoming certain constraints associated with distance-based similarity search.
    Present these alternative questions with newline separation.
    Initial question: {question}""",
)
llm = ChatOpenAI(temperature=0)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Other inputs
question = "What are the approaches to Task Decomposition?"

# Run
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output

# Results
relevantDocs = retriever.get_relevant_documents(
    query="What does the author say about the code complexity?"
)
len(relevantDocs)
```
</details>
