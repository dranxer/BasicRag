import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import base64
import requests

# Load environment variables
load_dotenv()

# Set the Hugging Face token from Streamlit secrets (or .env)
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Configure the Llama index settings (embeddings only)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def get_context(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    # Retrieve relevant context from the vectorstore
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    context_str = "\n".join([n.get_content() for n in nodes])
    return context_str

def query_hf(prompt, context=None):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    else:
        full_prompt = prompt
    payload = {"inputs": full_prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    # Check for HTTP errors or empty response
    if response.status_code != 200:
        return f"API Error: {response.status_code} - {response.text}"
    if not response.text.strip():
        return "API Error: Empty response from Hugging Face Inference API."
    try:
        result = response.json()
    except Exception as e:
        return f"API Error: Could not decode JSON. Raw response: {response.text}"
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "error" in result:
        return f"Error: {result['error']}"
    else:
        return str(result)

def handle_query(query):
    context = get_context(query)
    # Custom system prompt for Suriya
    if "suriya" in query.lower():
        return ("I was created by Suriya, an enthusiast in Artificial Intelligence. "
                "He is dedicated to solving complex problems and delivering innovative solutions. "
                "With a strong focus on machine learning, deep learning, Python, generative AI, NLP, and computer vision, "
                "Suriya is passionate about pushing the boundaries of AI to explore new possibilities.")
    answer = query_hf(query, context)
    return answer

# Streamlit app initialization
st.title("(PDF) Information and Inference🗞️")
st.markdown("Retrieval-Augmented Generation") 
st.markdown("start chat ...🚀")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF and ask me anything about its content.'}]

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            filepath = "data/saved_pdf.pdf"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # displayPDF(filepath)  # Display the uploaded PDF
            data_ingestion()  # Process PDF every time new file is uploaded
            st.success("Done")

user_prompt = st.chat_input("Ask me anything about the content of the PDF:")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({'role': 'assistant', "content": response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])