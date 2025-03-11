import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory

# Page Configuration
st.set_page_config(page_title="ICT Kerala Chatbot", page_icon="🤖", layout="wide")
st.title("Ask Anika 🤖")
st.caption("Your Virtual Assistant for ICT Kerala")

# Load environment variables
load_dotenv(find_dotenv())

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result")

# Sidebar Info
st.sidebar.title("🔧 Configuration")
st.sidebar.write("This chatbot helps users with queries about ICT Kerala's offerings.")

# Progress bar for initialization
progress = st.sidebar.progress(0)

# Check environment variables
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    st.sidebar.error("❌ HF_TOKEN not found.")
else:
    st.sidebar.success("✓ HF_TOKEN loaded.")
progress.progress(20)

# Check vector store path
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    st.sidebar.error(f"❌ Vector store path not found: {DB_FAISS_PATH}")
else:
    st.sidebar.success(f"✓ Vector store path found.")
progress.progress(40)

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt():
    prompt_template = """
    You are Anika, the virtual assistant for ICT Academy of Kerala. You are here to assist users with any questions they may have regarding ICT Academy of Kerala, including information about courses, programs, admissions, events, and other relevant details.
    Context: {context}
    Question: {question}
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def load_llm():
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.5,
            model_kwargs={
                "token": hf_token,
                "max_length": 512
            }
        )
        return llm
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None

def main():
    # Main Section Layout
    st.header("Chat with Anika!")
    st.write("Feel free to ask any questions regarding ICT Kerala's courses, events, and more.")
    
    # Progress Bar during initialization
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("❌ Failed to initialize vector store")
        return
    llm = load_llm()
    if llm is None:
        st.error("❌ Failed to initialize LLM")
        return
    
    progress.progress(80)
    
    # Chatting UI
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Enter your query here...")

    if prompt:
        with st.chat_message('user'):
            st.markdown(f"**You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()},
                memory=memory
            )
            with st.spinner("Anika is thinking..."):
                response = qa_chain.invoke({'context': '', 'query': prompt})
                result = response["result"]
                with st.chat_message('assistant'):
                    st.markdown(f"**Anika:** {result}")
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

    progress.progress(100)

if __name__ == "__main__":
    main()
