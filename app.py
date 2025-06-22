import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv
from transformers import pipeline
import requests

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# Sidebar file upload
with st.sidebar:
    st.header("📁 Upload PDF or TXT")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    if uploaded_file is not None:
        os.makedirs("docs", exist_ok=True)
        file_path = f"docs/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(ingest_file(file_path))
        if "chat" in st.session_state:
            del st.session_state.chat

# Init session
if "chat" not in st.session_state:
    st.session_state.chat = []

# Load vectorstore and QA chain
rag_available = os.path.exists("modules/vectorstore/index.faiss")
if rag_available and "qa" not in st.session_state:
    st.session_state.qa = get_chain()

# Setup fallback LLM (Hugging Face Inference API)
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query_hf(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "error" in result:
        return f"Error: {result['error']}"
    else:
        return str(result)

if "llm" not in st.session_state:
    st.session_state.llm = query_hf

# Show chat history
for msg in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# Chat input
prompt = st.chat_input("Ask your question (about uploaded file or general topic)...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            if rag_available:
                response = st.session_state.qa({"question": prompt})
                answer = response["answer"]
            else:
                answer = st.session_state.llm(prompt)
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat.append({"user": prompt, "bot": answer})
