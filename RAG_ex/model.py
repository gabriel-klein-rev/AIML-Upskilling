from langchain_community.llms import GPT4All

# Load the local GPT4All model
# TODO: enter path to model.
gpt_model = GPT4All(model="../Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Function to generate a response using the model
def generate_response(context, question, prompt):
    input_prompt = prompt.format(context=context, question=question)
    return gpt_model(input_prompt)
