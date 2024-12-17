# Cumulative for the Document Transformers
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Document Transformers topic.

### Learning Objectives

After completing this module, associates should be able to:
- Transform documents
- Split, combine, filter, and manipulate documents
 
</details>
<details><summary>Description</summary>

# Description of the Document Transformers topic.

### Document Transformers
When confronted with extensive text, it becomes imperative to partition it into manageable segments. Despite its apparent simplicity, this process entails considerable potential complexity. The goal is to ideally preserve the semantic coherence of text segments, with the definition of "semantically related" contingent upon the nature of the text. This notebook presents various methodologies for achieving this.

On a broader scale, text segmentation operates according to the following principles:

1. Split the text into concise, semantically meaningful units, often at the sentence level.
2. Begin merging these small units into more extensive chunks until a predetermined size is reached, measured by a designated function.
3. Upon reaching this size threshold, designate the chunk as an independent piece of text, then initiate the creation of a new chunk with some overlap to maintain contextual continuity.
This implies that there are two distinct axes for customizing your text segmentation:

- How the text is divided.
- How to determine the chunk size.

</details>
<details><summary>Implementation</summary> 

# Implementation for the Document Transformers topic

### Document Transformers

The default and recommended text segmentation tool is the RecursiveCharacterTextSplitter. This splitter operates on a specified list of characters, attempting to generate chunks by splitting at the first character. If any resulting chunks exceed a defined size, the splitter moves on to the next character and repeats the process. By default, the characters targeted for splitting are ["\n\n", "\n", " ", ""].

Beyond influencing the characters used for splitting, there are additional parameters under your control:

- length_function: Determines how the length of chunks is computed. The default is a character count, but it's common to utilize a token counter.
- chunk_size: Sets the maximum size for your chunks, measured by the specified length function.
- chunk_overlap: Establishes the maximum overlap between consecutive chunks. Introducing some overlap can be beneficial to maintain continuity, resembling a sliding window effect.
- add_start_index: Dictates whether to incorporate the starting position of each chunk within the original document in the metadata.
```python
# split a long document
with open('../../testtext.txt') as f:
    test_text = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # with a small chunk siz
    chunk_size = 75,
    chunk_overlap  = 15,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([test_text])
print(texts[0])
print(texts[1])
```
</details>
