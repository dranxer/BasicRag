import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🤖 Gemini RAG Chatbot")

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
            del st.session_state.chat  # Reset chat if new file uploaded

if os.path.exists("modules/vectorstore/index.faiss"):
    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.qa = get_chain()

    for msg in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])

    prompt = st.chat_input("Ask something about your document...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = st.session_state.qa({"question": prompt})
            answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat.append({"user": prompt, "bot": answer})
else:
    st.info("📂 Please upload a file to begin chatting.")