### Local Vector DB with ChromaDB
<details><summary>Prerequisites and Learning Objectives</summary>


#### Prerequisites and Learning Objectives:

**Prerequisites:**
- Basic knowledge of Python
- ChromaDB installation

**Learning Objectives:**
- Understand the steps involved in setting up a Vector DB with ChromaDB.

</details>
<details><summary>Description</summary>

#### Description:

**1. **Get the Chroma Client**
   ```python
   import chromadb
   chroma_client = chromadb.Client()
   
   ```

**2. **Create a Collection:**
   ```python
   collection = chroma_client.create_collection(name="my_collection")
   ```

**3. **Add some tect documents to the collection:**
   ```python
collection.add(
      documents=["This is a document", "This is another document"],
      metadatas=[{"source": "my_source"}, {"source": "my_source"}],
      ids=["id1", "id2"]
)

# If you already have generated embeddings, you can load them directly

collection.add(
      embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
      documents=["This is a document", "This is another document"],
      metadatas=[{"source": "my_source"}, {"source": "my_source"}],
      ids=["id1", "id2"]
)


   ```

**4. **Query the collection:**
```
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)

```

</details>
<details><summary>Real World Application</summary>

#### Real World Application:

**Content Recommendation System:**
   - **Scenario:** Building a recommendation system for news articles.
   - **Implementation:** Store article embeddings in ChromaDB.
   - **Benefit:** Enable fast and accurate content recommendations based on article similarity.

</details>
<details><summary>Summary</summary>

#### Summary:

Setting up ChromaDB involves installing it, importing to your project, and creating a collection.

</details>
<details><summary>Practice Questions</summary>

#### Practice Questions:

1. What are the key steps involved in setting up ChromaDB?
2. How do you create a new collection in ChromaDB?

</details>