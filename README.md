#  ICT Kerala Chatbot using LangChain, FAISS, and HuggingFace

This project implements a virtual assistant for the ICT Academy of Kerala using the LangChain framework, FAISS for vector-based retrieval, and a HuggingFace Language Model. The assistant, named Anika, is designed to answer queries related to ICT Academy of Kerala’s offerings, including courses, programs, admissions, and events.

# Features

Virtual Assistant (Anika): Anika is tailored to provide accurate information based on the context of ICT Academy of Kerala.

Vector Store (FAISS): Efficiently retrieves relevant information from stored knowledge using sentence embeddings.

HuggingFace Model: Uses the Mistral-7B-Instruct model from HuggingFace for generating accurate, context-aware responses.

Streamlit UI: A web-based UI to interact with Anika.

Conversation Memory: Keeps track of the conversation context using ConversationBufferMemory.

Custom Prompt Template: Ensures that responses are concise and restricted to relevant ICT Academy information.
