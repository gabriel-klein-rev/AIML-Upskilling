# Cumulative for the SequentialChain
<details><summary>Learning Objectives</summary>

# Learning Objectives for the SequentialChain topic.

### Learning Objectives

After completing this module, associates should be able to:
- make a series of calls to a language model
- take the output from one call and use it as the input to another
 
</details>
<details><summary>Description</summary>

# Description of the SequentialChain topic.

### SequentialChain
After initiating a call to a language model, the subsequent action involves executing a sequence of calls to a language model. This proves especially beneficial when you intend to utilize the output from one call as the input for another.

The suggested method for accomplishing this is by employing the LangChain Expression Language. The legacy approach involves utilizing the SequentialChain.
</details>
<details><summary>Implementation</summary> 

# Implementation for the SequentialChain topic

### SequentialChain
```python
from langchain.prompts import PromptTemplate

recipe_prompt = PromptTemplate.from_template(
    """You are a chef. Given the title of a dish, it is your job to write a recipe for that dish.

Name: {name}
Chef: This is a recipe for the above dish:"""
)

review_prompt = PromptTemplate.from_template(
    """You are a food critic for a magazine. Given the recipe of a dish, it is your job to write a review for that dish.

Dish Synopsis:
{synopsis}
Review from a food critic of the above recipe:"""
)
```

Using LCEL:

Creating a sequence of calls (to LLMs or any other component/arbitrary function) is precisely what LangChain Expression Language was designed for.
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

llm = ChatOpenAI()
chain = (
    {"recipe": recipe_prompt | llm | StrOutputParser()}
    | review_prompt
    | llm
    | StrOutputParser()
)
chain.invoke({"name": "shrimp Gumbo"})
```

For the synopsis as output:

```python
from langchain.schema.runnable import RunnablePassthrough

synopsis_chain = synopsis_prompt | llm | StrOutputParser()
review_chain = review_prompt | llm | StrOutputParser()
chain = {"synopsis": synopsis_chain} | RunnablePassthrough.assign(review=review_chain)
chain.invoke({"name": "Shrimp Gumbo"})
```
</details>
