from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.services.document_service import DocumentService
from src.services.rag_service import RAGService
from src.services.llm_service import LLMService

app = FastAPI(title="RAG Chatbot API")

rag_service = RAGService()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = await DocumentService.save_uploaded_file(file)
        
        # Load document
        documents = DocumentService.load_document(file_path)
        
        # Process and store
        num_chunks = rag_service.process_and_store_documents(documents)
        
        return {"message": "File processed successfully", "chunks": num_chunks, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        # Retrieve context
        retrieved_docs = rag_service.retrieve_context(request.query)
        
        # Generate answer
        response = llm_service.generate_answer(request.query, retrieved_docs)
        
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
