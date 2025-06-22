import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
import os

st.set_page_config(page_title="Llama 2 RAG App", page_icon="🦙", layout="centered")
st.title("🦙 Llama 2 RAG Chat (LlamaIndex)")

# Sidebar for HuggingFace token
token = st.sidebar.text_input("HuggingFace API Token", type="password", value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

# File uploader
doc_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"]) 

if doc_file:
    with open("uploaded_doc", "wb") as f:
        f.write(doc_file.read())
    docs = SimpleDirectoryReader(input_files=["uploaded_doc"]).load_data()
    st.success("Document loaded!")
    # Set HuggingFace token as env variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    # Build index
    llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2-7B-Instruct",
        tokenizer_name="Qwen/Qwen2-7B-Instruct",
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        tokenizer_kwargs={},
        device_map="auto",
    )
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    chat_engine = index.as_chat_engine()
    st.session_state["chat_engine"] = chat_engine
    st.session_state["chat_history"] = []

if "chat_engine" in st.session_state:
    prompt = st.chat_input("Ask something about your document...")
    if prompt:
        st.session_state["chat_history"].append(("user", prompt))
        response = st.session_state["chat_engine"].chat(prompt)
        st.session_state["chat_history"].append(("llama2", response.response))
    for role, msg in st.session_state["chat_history"]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg) 