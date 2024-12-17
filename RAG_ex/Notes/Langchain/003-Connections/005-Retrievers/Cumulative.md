# Cumulative for the Retrievers
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Retrievers topic.

### Learning Objectives

After completing this module, associates should be able to:
- Return documents with an unstructured query
- Use Retrievers
 
</details>
<details><summary>Description</summary>

# Description of the Retrievers topic.

### Retrievers
A retriever functions as an interface designed to provide documents based on an unstructured query. Its scope is broader than that of a vector store. Unlike a vector store, a retriever isn't necessarily required to store documents; its primary function is to retrieve and return them. While vector stores can serve as the foundation for a retriever, there exist alternative types of retrievers.

Retrievers adhere to the Runnable interface, a fundamental component of the LangChain Expression Language (LCEL). This implies that they support invoke, ainvoke, stream, astream, batch, abatch, and astream_log calls.

Retrievers accept a string query as input and return a list of Document's as output.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Retrievers topic

### Retrievers
We use a Chroma vector store-backed retriever. First:
```command
pip install chromadb
```
Then
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

full_text = open("test_text.txt", "r").read()
text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=75)
texts = text_splitter.split_text(full_text)

embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

retrieved_docs = retriever.invoke(
    "What questions are asked in the text?"
)
print(retrieved_docs[0].page_content)
```
Retrievers are Runnable's, so we can easily compose them with other Runnable objects:
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What was the first question asked in the text?")
```
</details>
