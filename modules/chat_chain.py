from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, set_seed

def get_chain():
    # Load embeddings (fast + accurate, CPU-friendly)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load vector store from local path
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)

    # Use a small, CPU-friendly text generation model
    set_seed(42)
    hf_pipeline = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",  # Small model for demo/cloud
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Wrap the pipeline with LangChain LLM wrapper
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        return_source_documents=False,
        chain_type="stuff"
    )

    return qa
