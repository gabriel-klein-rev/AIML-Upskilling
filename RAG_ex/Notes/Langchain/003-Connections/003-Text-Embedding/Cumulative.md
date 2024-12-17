# Cumulative for the Text embedding
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Text embedding topic.

### Learning Objectives

After completing this module, associates should be able to:
- Create a vector representation of a piece of text
- Distinguish methods for embedding documents and for embedding a query
 
</details>
<details><summary>Description</summary>

# Description of the Text embedding topic.

### Text embedding

The Embeddings class serves as an interface for interacting with various text embedding models. Given the multitude of embedding model providers available (such as OpenAI, Cohere, Hugging Face, etc.), this class is crafted to establish a standardized interface applicable to all.

Embeddings function by creating a vector representation for a given piece of text. This approach proves advantageous as it allows us to conceptualize text within the vector space. Consequently, tasks like semantic search become feasible, where we seek text segments that exhibit high similarity in the vector space.

Within the core Embeddings class of LangChain, two essential methods are provided: one for embedding documents and another for embedding a query. The former accommodates multiple texts as input, while the latter is tailored for a single text. The rationale for maintaining these as distinct methods lies in the fact that certain embedding providers employ different techniques for documents (intended for searching) compared to queries (representing the search query itself).

</details>
<details><summary>Implementation</summary> 

# Implementation for the Text embedding topic

### Text embedding
First install the OpenAI Python package:
```command
pip install openai
```
Next obtain an API key and set it as an environment variable by running:
```command
export OPENAI_API_KEY="..."
```

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

embed_documents
Embed list of texts
embeddings = embeddings_model.embed_documents(
    [
        "What a great morning!",
        "Oh, hi!",
        "What is your name?",
        "My family calls me Jon",
        "Hello Jon!"
    ]
)
len(embeddings), len(embeddings[0])
```
Now, generate an embedding for a singular text segment with the intention of comparing it to other embedded text segments.

```python
embedded_query = embeddings_model.embed_query("What was the name discussed in the dialogue?")
embedded_query[:5]
```
</details>
