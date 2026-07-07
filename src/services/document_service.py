import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from fastapi import UploadFile
from src.config import Config

class DocumentService:
    @staticmethod
    async def save_uploaded_file(file: UploadFile) -> str:
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return file_path

    @staticmethod
    def load_document(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()
