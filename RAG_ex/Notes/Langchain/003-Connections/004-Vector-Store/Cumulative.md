# Cumulative for the Vector Store
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Vector Store topic.

### Learning Objectives

After completing this module, associates should be able to:
- Store and search over unstructured data
- Store embedded data and perform vector search with vector stores
 
</details>
<details><summary>Description</summary>

# Description of the Vector Store topic.

### Vector Store

A prevalent technique for managing and searching unstructured data involves embedding it and storing the resultant embedding vectors. During a query, the unstructured query is embedded, and the embedding vectors most closely resembling the embedded query are retrieved. Vector stores, specifically designed for storing embedded data and executing vector searches, streamline this process.

An integral aspect of engaging with vector stores is generating the vector to be stored, a task typically accomplished through embeddings. Consequently, it is advisable to acquaint yourself with the interfaces of text embedding models before delving into this realm.

Numerous commendable options exist for vector stores, and for our purposes, we will utilize Chroma, an option that is both free, open-source, and operates entirely on your local machine.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Vector Store topic

### Vector Store
We can use chroma vector database, running on your local machine as a library.
```command
pip install chromadb
```
To use OpenAIEmbeddings we get the OpenAI API Key.
```python
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('../../../test_text.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

#Similarity search
query = "What questions were asked in the text?"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

Conducting a similarity search based on vectors is another viable option. With similarity_search_by_vector, you have the capability to search for documents akin to a provided embedding vector, as it accepts an embedding vector as a parameter instead of a string.
```python
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
```
Since the query remains unchanged, we will see the same output.
</details>
