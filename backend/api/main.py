from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.vectorstore.retrieve import retrieve_top_k
from backend.core.generate_answer import generate_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Create FastAPI app
app = FastAPI(
    title="Ask My Course RAG API",
    description="A RAG-based question answering API for course materials",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for request/response
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QuestionResponse(BaseModel):
    answer: str
    context: List[str]
    question: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Ask My Course RAG API is running!"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Ask My Course RAG API",
        "version": "1.0.0"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer using the RAG pipeline.
    
    Args:
        request: QuestionRequest containing the question and optional top_k
        
    Returns:
        QuestionResponse with the answer and supporting context
    """
    try:
        logging.info(f"Received question: {request.question}")
        
        # Step 1: Retrieve relevant chunks
        logging.info(f"Retrieving top {request.top_k} chunks...")
        top_chunks = retrieve_top_k(request.question, k=request.top_k)
        logging.info(f"Retrieved {len(top_chunks)} chunks")
        
        # Step 2: Generate answer using the chunks
        logging.info("Generating answer...")
        answer = generate_answer(top_chunks, request.question)
        logging.info("Answer generated successfully")
        
        # Step 3: Extract context text for response
        context_texts = [chunk['text'] for chunk in top_chunks]
        
        return QuestionResponse(
            answer=answer,
            context=context_texts,
            question=request.question
        )
        
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
async def get_models():
    """Get information about the models being used"""
    return {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "mistralai/Mistral-7B-v0.1",
        "vector_store": "FAISS"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 