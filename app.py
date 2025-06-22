import streamlit as st
import os
from modules.ingest import ingest_file
from modules.chat_chain import get_chain
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot")

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

# Setup FLAN-T5 fallback model
if "llm" not in st.session_state:
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    def flan_t5(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=128)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    st.session_state.llm = flan_t5

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
