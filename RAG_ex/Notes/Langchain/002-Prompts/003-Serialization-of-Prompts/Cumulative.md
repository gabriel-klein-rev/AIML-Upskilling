# Cumulative for the  Serialization of Prompts
<details><summary>Learning Objectives</summary>

# Learning Objectives for the Serialization of Prompts topic.

### Learning Objectives

After completing this module, associates should be able to:
- Save Prompts as Files
- Share, Store and Version Prompts
 
</details>
<details><summary>Description</summary>

# Description of the Serialization of Prompts topic.

### Serialization of Prompts
It is often more advantageous to store prompts not within Python code but as files. This approach streamlines the sharing, storage, and versioning of prompts. This guide explores the methodology for achieving this in LangChain, offering a thorough examination of various prompt types and the array of serialization options at your disposal.

From a broader perspective, the following guiding principles dictate the serialization approach:

We ensure support for both JSON and YAML. Our goal is to advocate for serialization methods that maintain human readability on disk, and JSON and YAML emerge as two widely embraced approaches for achieving this. It's essential to clarify that this guideline is specifically tailored to prompts, and different serialization methods may be considered for other assets, such as examples.

Flexibility is retained by allowing the specification of everything in a single file or by storing distinct components (templates, examples, etc.) in separate files with references. Although consolidating all assets in a single file might be most practical in certain scenarios, there are instances where dividing assets (lengthy templates, substantial examples, reusable components) proves more advantageous. LangChain accommodates both approaches.

We establish a unified entry point for loading prompts from disk, streamlining the process of effortlessly loading any prompt type.

example:
```python
# Load all prompts with the `load_prompt` function.
from langchain.prompts import load_prompt
```

</details>
<details><summary>Implementation</summary> 

# Implementation for the Serialization of Prompts topic

### Serialization of Prompts

Here we load a PromptTemplate from YAML.
```command
cat first_prompt.yaml


    _type: prompt
    input_variables:
        ["adjective", "content"]
    template: 
        Tell me a {adjective} joke about {content}.
```
```python
prompt = load_prompt("first_prompt.yaml")
print(prompt.format(adjective="scary", content="ghosts"))
```

Here we load a PromptTemplate from JSON.
```command
cat first_prompt.json

    {
        "_type": "prompt",
        "input_variables": ["adjective", "content"],
        "template": "Tell me a {adjective} story about {content}."
    }
```
```python
prompt = load_prompt("first_prompt.json")
print(prompt.format(adjective="scary", content="ghosts"))
```


</details>
