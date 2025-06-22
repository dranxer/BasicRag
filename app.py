import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# Sidebar uploader
with st.sidebar:
    st.header("📁 Upload PDF or TXT")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    if uploaded_file is not None:
        os.makedirs("docs", exist_ok=True)
        path = f"docs/{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(ingest_file(path))
        if "chat" in st.session_state:
            del st.session_state.chat

# Init session state
if "chat" not in st.session_state:
    st.session_state.chat = []

rag_available = os.path.exists("modules/vectorstore/index.faiss")
if rag_available and "qa" not in st.session_state:
    st.session_state.qa = get_chain()

# Fallback Falcon model (no SentencePiece needed)
if "llm" not in st.session_state:
    hf_pipeline = pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        max_new_tokens=256,
        do_sample=False,
        temperature=0.3
    )

    def falcon_response(prompt):
        output = hf_pipeline(prompt)[0]["generated_text"]
        return output[len(prompt):].strip()

    st.session_state.llm = falcon_response

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

prompt = st.chat_input("Ask a question (with or without PDF)...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        if rag_available:
            response = st.session_state.qa({"question": prompt})
            answer = response["answer"]
        else:
            answer = st.session_state.llm(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat.append({"user": prompt, "bot": answer})
