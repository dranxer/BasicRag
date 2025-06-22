from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("modules/vectorstore", embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa
