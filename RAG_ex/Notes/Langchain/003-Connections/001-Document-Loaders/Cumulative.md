# Cumulative for the Document Loaders
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Document Loaders topic.

### Learning Objectives

After completing this module, associates should be able to:
- Define Documents
- Load data from a source
 
</details>
<details><summary>Description</summary>

# Description of the Document Loaders topic.

### Document Loaders
Employ document loaders to retrieve data from a specified source in the form of a Document. A Document, in this context, encompasses a segment of text along with its associated metadata. For instance, document loaders can handle tasks such as loading a straightforward .txt file, fetching the textual content of any webpage, or acquiring a transcript from a YouTube video.

These document loaders offer a "load" method designed to fetch data as documents from a preconfigured source. Additionally, they may optionally incorporate a "lazy load" mechanism, allowing data to be loaded into memory only when required.
</details>
<details><summary>Implementation</summary> 

# Implementation for the Document Loaders topic

### Document Loaders
The most straightforward loader interprets a file as text, consolidating it into a single document.

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("./dir.md")
loader.load()
```
</details>
