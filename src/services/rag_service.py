import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

class RAGService:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # Initialize or load Chroma DB
        self.vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_and_store_documents(self, documents):
        if not documents:
            return 0
            
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
            chunk.metadata["chunk_id"] = chunk_id
            
            # Ensure basic metadata is present for the response formatting
            if "source" in chunk.metadata:
                chunk.metadata["file_name"] = os.path.basename(chunk.metadata["source"])
            else:
                chunk.metadata["file_name"] = "unknown"
                
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 1 # Default for txt/csv
                
        # Add to vector store
        self.vector_store.add_documents(documents=chunks)
        
        return len(chunks)

    def retrieve_context(self, query: str, top_k: int = 5):
        # Retrieve top_k chunks
        docs = self.vector_store.similarity_search(query, k=top_k)
        return docs
