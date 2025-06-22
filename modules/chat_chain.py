from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline

def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)

    hf_pipeline = pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        max_new_tokens=256,
        do_sample=False,
        temperature=0.3
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa
