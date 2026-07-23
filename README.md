# DBT Coach — RAG Chatbot 🧘

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-0467DF)
![LLM](https://img.shields.io/badge/LLM-Zephyr--7B-7C3AED)
![License](https://img.shields.io/badge/License-MIT-green)

A **Retrieval-Augmented Generation** chatbot that answers questions about
**Dialectical Behaviour Therapy (DBT)**. It grounds every reply in a source workbook
instead of free-styling: the PDF is embedded into a vector index, the most relevant
passages are retrieved for each question, and an open-source LLM streams a concise,
context-aware answer.

> ‼️ **Disclaimer:** built on a publicly available DBT workbook for educational purposes.
> This is **not** medical advice.

## How it works

```
saved_pdf.pdf ─► PyMuPDF extract ─► MiniLM embeddings ─► FAISS index
                                                              │
question ─────────────────────────► retrieve top-k ──────────┤
                                                              ▼
                       Zephyr-7B (HF Inference API) ─► streamed, grounded answer
```

| Layer | Choice |
|-------|--------|
| PDF parsing | `PyMuPDF` (fitz) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | `faiss` (L2 flat index) |
| Generator | `HuggingFaceH4/zephyr-7b-beta` via `huggingface_hub` |
| UI | `gradio` chat interface with tunable temperature / top-p / max-tokens |

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL. Adjust the system prompt and sampling parameters live
from the UI, or try one of the built-in example prompts.

## Notes & design

- The embedding model is loaded **once** at startup and reused for every query (the index
  and encoder are held on a single `RagStore` instance) — retrieval stays fast per message.
- Answers stream token-by-token for responsiveness.
- Swap `PDF_PATH`, `EMBED_MODEL`, or `LLM_MODEL` at the top of `app.py` to point it at a
  different document or model.

## License

MIT — see [LICENSE](LICENSE).
