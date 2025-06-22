import gradio as gr
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("sample.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        return [self.documents[i]["content"] for i in I[0]]

app = MyApp()

def respond(message: str, history: List[Tuple[str, str]], system_message: str, max_tokens: int, temperature: float, top_p: float):
    system_message = "You are a knowledgeable DBT coach. You always talk about one option at a time..."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🧘‍♀️ **Dialectical Behaviour Therapy**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a DBT exercise book that is publicly available. "
        "We are not medical practitioners, and the use of this chatbot is at your own responsibility.‼️"
    )

    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["I feel overwhelmed with work."],
            ["Can you guide me through a quick meditation?"],
            ["How do I stop worrying about things I can't control?"],
            ["What are some DBT skills for managing anxiety?"],
            ["Can you explain mindfulness in DBT?"]
        ],
        title='Dialectical Behaviour Therapy Assistant 👩‍⚕️'
    )

if __name__ == "__main__":
    demo.launch()
