from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)

    hf_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa