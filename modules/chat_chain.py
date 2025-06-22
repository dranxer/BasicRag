from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def get_chain():
    # Use a lightweight embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)

    # Replace FLAN-T5 with a model that does not require SentencePiece
    hf_pipeline = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=256, do_sample=True, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RetrievalQA setup
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=False,
        chain_type="stuff"
    )
    return qa
