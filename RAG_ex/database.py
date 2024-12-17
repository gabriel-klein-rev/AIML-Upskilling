import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.Client()

# Create a new collection
collection = client.create_collection("faq_collection")

# Initialize the embedding model (e.g., all-MiniLM-L6-v2 for lightweight embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to add documents with embeddings to ChromaDB
def add_documents(documents):
    for doc in documents:
        # Generate the embedding for each document
        embedding = embedding_model.encode(doc["content"])
        
        # Add document along with the embedding to ChromaDB
        collection.add(ids=[doc["id"]], documents=[doc["content"]], embeddings=[embedding])

# Function to retrieve documents based on query
def retrieve_docs(question):
    # Retrieve similar documents from ChromaDB
    results = collection.query(query_texts=[question], n_results=1)
    return results["documents"][0] if results["documents"] else ""
