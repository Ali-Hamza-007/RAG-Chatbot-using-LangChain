from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config

# Define the structured output schema
class SourceMetadata(BaseModel):
    file: str = Field(description="The name of the source file")
    page: int = Field(description="The page number where the information was found")
    chunk: str = Field(description="The chunk ID of the source text")

class RAGResponse(BaseModel):
    answer: str = Field(description="The answered generated based ONLY on the context. If the answer is not in the context, state that clearly.")
    sources: List[SourceMetadata] = Field(description="List of sources used to generate the answer. Empty if the answer is not found in the context.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of the answer based on the provided context.")

class LLMService:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name=Config.LLM_MODEL,
            groq_api_key=Config.GROQ_API_KEY
        )
        
        # Use Pydantic structured output
        self.structured_llm = self.llm.with_structured_output(RAGResponse)
        
        # Define the prompt
        system_prompt = """You are a precise, document-based AI assistant. 
Your task is to answer the user's question based strictly on the provided Context.

Context:
{context}

Instructions:
1. ONLY use the information provided in the Context above. Do not use outside knowledge.
2. If the Context does not contain the answer, your 'answer' field should say exactly: "The answer is not available in the provided documents.", return an empty list for 'sources', and set 'confidence' to "low".
3. When you find the answer, include the metadata of the exact chunks you used in the 'sources' field.
4. Provide a 'confidence' score based on how explicitly the Context answers the question.

Do not hallucinate.
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.structured_llm

    def format_context(self, docs) -> str:
        formatted_chunks = []
        for doc in docs:
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            
            chunk_text = f"Source: {file_name} | Page: {page} | Chunk ID: {chunk_id}\nContent:\n{doc.page_content}\n"
            formatted_chunks.append(chunk_text)
            
        return "\n---\n".join(formatted_chunks)

    def generate_answer(self, question: str, retrieved_docs) -> RAGResponse:
        if not retrieved_docs:
            return RAGResponse(
                answer="The answer is not available in the provided documents.",
                sources=[],
                confidence="low"
            )
            
        context_text = self.format_context(retrieved_docs)
        
        # Invoke the chain
        response = self.chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return response
