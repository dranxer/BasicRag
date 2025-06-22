# Llama 2 RAG Chat App (Streamlit + LlamaIndex)

A minimal Retrieval-Augmented Generation (RAG) app using Llama 2 (via HuggingFace) and LlamaIndex, with a Streamlit UI.

## Features
- Upload a document (txt or PDF)
- Build a vector index with LlamaIndex
- Chat with your document using Llama 2 (HuggingFace Inference)

## How to Deploy on Streamlit Cloud

1. **Push this repo to GitHub.**
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and create a new app.
3. **Set the main file path to `app.py`.**
4. **Add your HuggingFace API token** in the Streamlit Cloud "Secrets" tab:
   ```
   HUGGINGFACEHUB_API_TOKEN = "your-hf-token"
   ```
   You can get a token at https://huggingface.co/settings/tokens
5. **Click Deploy!**

## Local Development
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- You need a HuggingFace account and an access token to use Llama 2 models.
- For larger models or more speed, you can swap the model name in `app.py`. 