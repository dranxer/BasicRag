import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def ingest_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    # 📄 Load PDF or TXT
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("❌ Only PDF or TXT files are supported.")

    docs = loader.load()

    # 🔪 Chunk text into smaller parts
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 🧼 Remove empty or invalid chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("❌ No usable content found in document after splitting.")

    # 🔍 Create Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise RuntimeError("⚠️ Gemini embedding failed. Try using a simpler or smaller file.") from e

    # 💾 Save FAISS index
    vectorstore.save_local("modules/vectorstore")

    return "✅ File indexed and ready to chat!"
