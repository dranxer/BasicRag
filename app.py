"""
DBT Coach — a Retrieval-Augmented Generation chatbot.

A Gradio chat app that answers questions about Dialectical Behaviour Therapy (DBT).
It indexes a source PDF with sentence-transformer embeddings + FAISS, retrieves the
most relevant passages for each question, and streams a grounded reply from a hosted
open-source LLM (Zephyr-7B) via the Hugging Face Inference API.

Disclaimer: this is an educational demo built on a publicly available DBT workbook.
It is not medical advice.
"""
import fitz  # PyMuPDF
import faiss
import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

PDF_PATH = "saved_pdf.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

client = InferenceClient(LLM_MODEL)


class RagStore:
    """Loads a PDF once, embeds it once, and serves fast similarity search."""

    def __init__(self, pdf_path: str):
        # Load the embedding model a single time (previously reloaded on every query).
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.documents = self._load_pdf(pdf_path)
        self.index = self._build_index()

    def _load_pdf(self, file_path: str):
        doc = fitz.open(file_path)
        docs = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            if text:
                docs.append({"page": page_num + 1, "content": text})
        print(f"PDF processed: {len(docs)} non-empty pages")
        return docs

    def _build_index(self):
        embeddings = self.encoder.encode(
            [d["content"] for d in self.documents], convert_to_numpy=True
        )
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("Vector database built")
        return index

    def search(self, query: str, k: int = 3):
        q = self.encoder.encode([query], convert_to_numpy=True)
        k = min(k, len(self.documents))
        _, idx = self.index.search(q, k)
        return [self.documents[i]["content"] for i in idx[0]]


store = RagStore(PDF_PATH)

SYSTEM_PROMPT = (
    "You are a knowledgeable and supportive DBT (Dialectical Behaviour Therapy) coach. "
    "Discuss one skill or option at a time, keep answers concise and practical, and ground "
    "your guidance in the retrieved context. You are not a medical professional."
)


def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    context = "\n".join(store.search(message))
    messages.append({"role": "system", "content": "Relevant context:\n" + context})
    messages.append({"role": "user", "content": message})

    response = ""
    for chunk in client.chat_completion(
        messages, max_tokens=max_tokens, stream=True,
        temperature=temperature, top_p=top_p,
    ):
        token = chunk.choices[0].delta.content or ""
        response += token
        yield response


demo = gr.Blocks()
with demo:
    gr.Markdown("# 🧘 DBT Coach — RAG Chatbot")
    gr.Markdown(
        "‼️ **Disclaimer:** based on a publicly available DBT workbook. "
        "This is not medical advice; use at your own discretion. ‼️"
    )
    gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Textbox(value=SYSTEM_PROMPT, label="System prompt"),
            gr.Slider(1, 2048, value=512, step=1, label="Max new tokens"),
            gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p"),
        ],
        examples=[
            ["I feel overwhelmed with work."],
            ["Can you guide me through a quick grounding exercise?"],
            ["How do I stop worrying about things I can't control?"],
            ["What are some DBT skills for managing anxiety?"],
            ["Can you explain mindfulness in DBT?"],
        ],
        title="DBT Coach",
    )

if __name__ == "__main__":
    demo.launch()
