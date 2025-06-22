import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🧠 Gemini RAG Chatbot")

with st.sidebar:
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    if uploaded_file is not None:
        os.makedirs("docs", exist_ok=True)
        save_path = f"docs/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        msg = ingest_file(save_path)
        st.success(msg)

if os.path.exists("app/vectorstore/index.faiss"):
    st.subheader("💬 Ask Questions About the Uploaded File")

    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.qa = get_chain()

    user_input = st.chat_input("Ask a question")
    if user_input:
        with st.spinner("Thinking..."):
            response = st.session_state.qa({"question": user_input})
            st.session_state.chat.append((user_input, response["answer"]))

    for q, a in st.session_state.chat:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("ai"):
            st.write(a)
else:
    st.info("Upload a file to begin.")
