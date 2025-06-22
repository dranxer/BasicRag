# Do your LLM even RAG bro?

RAG web application using Python, Streamlit and LangChain, so you can chat with Documents, Websites and other custom data.

To run it locally:

```bash
$ git clone <this-repo-url>

$ cd <this-repo-folder>

$ python -m venv venv

$ venv\Scripts\activate  # or source venv/bin/activate in Linux/Mac

$ pip install -r requirements.txt

$ streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "New app".
3. Connect your GitHub and select this repo.
4. Set the main file path to `app.py`.
5. In the Streamlit Cloud app settings, go to the "Secrets" tab and add your API keys:

```
AZ_OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
```

6. Click "Deploy".

Video: https://youtu.be/abMwFViFFhI  
Blog: https://medium.com/@enricdomingo  