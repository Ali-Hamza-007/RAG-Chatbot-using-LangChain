# RAG Chatbot using LangChain

A completely scratch-built RAG Chatbot using LangChain, Chroma, and Groq.
Includes a FastAPI backend and a Streamlit UI.

## Features
- Upload PDF, TXT.
- Process and chunk documents with metadata preservation
- Store embeddings in a local Chroma vector database
- Retrieve answers accurately with traceable sources (file name, page, chunk id)
- Fallback logic for unanswerable questions
- Modern Streamlit UI

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set your `GROQ_API_KEY`.
3. Run the FastAPI backend:
   ```bash
   uvicorn src.main:app --reload
   ```
4. In a separate terminal, run the Streamlit frontend:
   ```bash
   streamlit run src/app.py
   ```
5. Python version 3.13+ 
### Note: Add the documents one by one and click on " Process Document " Button for Giving it to RAG, once you get a successful response , it means your file successfully Stored in ChromaDB (vectorDB)
