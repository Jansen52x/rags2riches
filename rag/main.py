from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from config import settings
from embedding_service import EmbeddingService
from llm_service import LLMService
from rag_service import RAGService

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Query Service",
    description="Microservice for querying RAG system using NVIDIA NIM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    k: Optional[int] = Field(None, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    k: Optional[int] = Field(None, description="Number of results to return")


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    chromadb: bool
    llm: bool
    embeddings: bool


# Global service instances
embedding_service = None
llm_service = None
rag_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedding_service, llm_service, rag_service

    logger.info("Starting RAG Query Service...")

    try:
        # Initialize services
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        rag_service = RAGService(embedding_service, llm_service)

        # Health checks
        if not embedding_service.health_check():
            logger.warning("Embedding service health check failed")

        if not llm_service.health_check():
            logger.warning("LLM service health check failed")

        if not rag_service.health_check():
            logger.warning("ChromaDB health check failed")

        logger.info("RAG Query Service is ready!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation)

    Retrieves relevant documents from the vector database and uses an LLM to generate an answer.
    """
    try:
        result = rag_service.query(
            query=request.query,
            k=request.k,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform similarity search in the vector database

    Returns relevant document chunks without LLM generation.
    """
    try:
        results = rag_service.search(
            query=request.query,
            k=request.k
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check health status of all services"""
    chromadb_healthy = rag_service.health_check() if rag_service else False
    llm_healthy = llm_service.health_check() if llm_service else False
    embeddings_healthy = embedding_service.health_check() if embedding_service else False

    status = "healthy" if all([chromadb_healthy, llm_healthy, embeddings_healthy]) else "degraded"

    return {
        "status": status,
        "chromadb": chromadb_healthy,
        "llm": llm_healthy,
        "embeddings": embeddings_healthy
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Query Service",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
