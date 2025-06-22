from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

def get_chain():
    # Load embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)

    # Use flan-t5-base for better QA performance
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    def flan_t5_prompt(x):
        input_ids = tokenizer(x, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=128)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Wrap with HuggingFacePipeline
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa
