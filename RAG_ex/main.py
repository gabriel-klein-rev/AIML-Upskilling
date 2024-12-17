import streamlit as st
from langchain.prompts import PromptTemplate
from database import retrieve_docs, add_documents
from model import generate_response

# Define the prompt template
prompt_template = """You are a helpful assistant. Use the context below to answer the user's question.

Context: {context}
User: {question}

Assistant:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Example documents (assuming they are already added to ChromaDB)
documents = [
    {"id": "1", "content": "Langchain is a framework for developing applications powered by language models."},
    {"id": "2", "content": "A vector database stores data as high-dimensional vectors, often used in machine learning and AI for efficient similarity searches."},
    {"id": "3", "content": "Retrieval-Augmented Generation combines information retrieval with a language model, allowing the model to fetch relevant information to improve its responses."},
    {"id": "4", "content": "Csontu is a creature that lives in the Taiga region of Alaska. It is notable for it's long legs and dark black hair."}
]

add_documents(documents)

# Function to get chatbot response
def chatbot_response(user_input):
    # Retrieve relevant documents from ChromaDB
    context = retrieve_docs(user_input)
    
    # Generate a response using the LLM with context
    answer = generate_response(context, user_input, prompt)
    
    return answer

# Streamlit UI setup
def create_chat_interface():
    # Title and description
    st.title("Chat with the RAG Chatbot")
    st.write("Ask a question below, and the chatbot will retrieve relevant information from its knowledge base.")

    # Maintain conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.text_input("Ask your question:")

    # If the user has input a question, process it
    if user_input:
        response = chatbot_response(user_input)
        
        # Add user query and chatbot response to session history
        st.session_state.history.append((user_input, response))

    # Display the conversation
    if st.session_state.history:
        for user_query, bot_response in st.session_state.history:
            st.write(f"**User:** {user_query}")
            st.write(f"**Chatbot:** {bot_response}")
            st.write("---")  # Separator between exchanges

# Run the app
if __name__ == "__main__":
    create_chat_interface()
