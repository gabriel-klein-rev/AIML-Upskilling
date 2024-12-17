# Cumulative for the Introduction to Retrieval-Augmented Generation (RAG)



<details><summary>Learning Objectives</summary>

# Learning Objectives for the Introduction to Retrieval-Augmented Generation (RAG) topic.

### Learning Objectives

After completing this module, associates should be able to:
- Understand the architecture of a RAG
- Use the main components of a RAG
</details>
<details><summary>Description</summary>

# Description of the Introduction to Retrieval-Augmented Generation (RAG) topic.

### Introduction to Retrieval-Augmented Generation (RAG)
RAG serves as a method to enhance LLM understanding by incorporating supplementary data, frequently of a private or real-time nature.

While LLMs possess the ability to analyze diverse subjects, their comprehension is confined to publicly available information up to a certain training point. To develop AI applications capable of reasoning about confidential or post-cutoff-date data, it becomes imperative to enrich the model's knowledge with the requisite information. This act of retrieving pertinent data and seamlessly integrating it into the model prompt is termed as Retrieval Augmented Generation (RAG).
</details>
<details><summary>Implementation</summary> 

# Implementation for the Introduction to Retrieval-Augmented Generation (RAG) topic

### Introduction to Retrieval-Augmented Generation (RAG)
A standard RAG application comprises two primary elements:

Indexing: a pipeline for ingesting data from a source and indexing it, typically performed offline.

Data Retrieval and Generation: the functional RAG process, where, during runtime, the user query is taken, and the pertinent data is extracted from the index, then forwarded to the model.

The most prevalent complete sequence from original data to response appears as follows:

Indexing:
1. Initially, data loading is required. DocumentLoaders are employed for this purpose.
2. Split: Text splitters divide extensive documents into smaller segments. This proves beneficial for both data indexing and model input, as searching and processing large segments can be challenging within a model's finite context window.
3. Store: A designated location is essential for storing and indexing the segmented data to facilitate future searches. This is commonly achieved using a VectorStore and an Embeddings model.

Retrieval and generation:

4. Retrieve: When presented with a user input, the relevant segments are fetched from storage utilizing a Retriever.
5. Generate: A ChatModel / LLM generates a response using a prompt that encompasses the question and the retrieved data.
</details>
