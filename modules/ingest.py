
import os
import shutil
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def ingest_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("❌ Only PDF or TXT files are supported.")

    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("❌ No usable content found in document after splitting.")

    if os.path.exists("modules/vectorstore"):
        shutil.rmtree("modules/vectorstore")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("modules/vectorstore")
    return "✅ File indexed and ready to chat!"
