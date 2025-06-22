import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload PDF or TXT")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    if uploaded_file is not None:
        os.makedirs("docs", exist_ok=True)
        path = f"docs/{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(ingest_file(path))
        # Clear previous chat history
        if "chat" in st.session_state:
            del st.session_state.chat

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Check if vectorstore exists and load RAG chain
rag_available = os.path.exists("modules/vectorstore/index.faiss")
if rag_available and "qa" not in st.session_state:
    st.session_state.qa = get_chain()

# Setup fallback LLM using Falcon (no SentencePiece dependency)
if "llm" not in st.session_state:
    st.session_state.llm = pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

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
                result = st.session_state.llm(prompt)[0]["generated_text"]
                answer = result[len(prompt):].strip()
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat.append({"user": prompt, "bot": answer})
