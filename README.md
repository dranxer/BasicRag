# BasicRag Streamlit Cloud

This is a Retrieval-Augmented Generation (RAG) chatbot demo for Streamlit Cloud.

## How to deploy on Streamlit Cloud

1. **Fork or upload this repo to your GitHub.**
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app.**
3. **Point it to your repo and set the main file to `app.py`.**
4. **(Optional) Add any secrets or environment variables in the Streamlit Cloud UI.**

### Notes
- This demo uses only CPU-friendly models (`sshleifer/tiny-gpt2` for text generation, `all-MiniLM-L6-v2` for embeddings).
- File uploads are ephemeral on Streamlit Cloud.
- If you want to use larger models, you must run locally or on a GPU-enabled server. 