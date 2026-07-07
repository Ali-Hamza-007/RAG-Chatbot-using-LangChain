import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Using Llama 3 for fast and accurate RAG responses
    LLM_MODEL = "llama-3.3-70b-versatile" 
    # Using HuggingFace's sentence transformers for embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Paths
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded_files")

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
