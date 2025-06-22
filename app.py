import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
import os

st.set_page_config(page_title="RAG Chat (Phi-3 Mini)", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chat with Phi-3 Mini (LlamaIndex)")
st.markdown("""
**Instructions:**
- Enter your HuggingFace API token in the sidebar (get one at https://huggingface.co/settings/tokens)
- Upload a small .txt file
- Ask questions about your document below
""")

# Sidebar for HuggingFace token
token = st.sidebar.text_input("HuggingFace API Token", type="password", value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

# File uploader (txt only)
doc_file = st.file_uploader("Upload a text file", type=["txt"]) 

# Error if no token
token_error = False
if not token:
    st.warning("Please enter your HuggingFace API token in the sidebar.")
    token_error = True

# Build index if file and token are present
if doc_file and not token_error:
    with st.spinner("Loading document and building index..."):
        with open("uploaded_doc.txt", "wb") as f:
            f.write(doc_file.read())
        try:
            docs = SimpleDirectoryReader(input_files=["uploaded_doc.txt"]).load_data()
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
            llm = HuggingFaceLLM(
                model_name="microsoft/phi-3-mini-4k-instruct",
                tokenizer_name="microsoft/phi-3-mini-4k-instruct",
                context_window=2048,
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
            st.success("Document loaded and indexed!")
        except Exception as e:
            st.error(f"Error loading document or building index: {e}")
            st.stop()

# Prompt input and send button (always visible)
prompt_disabled = "chat_engine" not in st.session_state
with st.form(key="chat_form", clear_on_submit=True):
    prompt = st.text_input("Your message:", disabled=prompt_disabled)
    send = st.form_submit_button("Send", disabled=prompt_disabled)

if send and prompt and not prompt_disabled:
    st.session_state["chat_history"].append(("user", prompt))
    with st.spinner("Getting answer from Phi-3 Mini..."):
        try:
            response = st.session_state["chat_engine"].chat(prompt)
            st.session_state["chat_history"].append(("phi3", response.response))
        except Exception as e:
            st.session_state["chat_history"].append(("phi3", f"Error: {e}"))

if "chat_history" in st.session_state:
    for role, msg in st.session_state["chat_history"]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg) 