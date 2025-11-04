from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from config import settings
from embedding_service import EmbeddingService
from llm_service import LLMService
from rag_service import RAGService
from query_builder import QueryBuilder

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


class BuilderQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    k: Optional[int] = Field(None, description="Number of results to return")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score (0.0-1.0)")
    include_sources: bool = Field(True, description="Include source documents in response")


class TemplateQueryRequest(BaseModel):
    template_name: str = Field(..., description="Name of the template to use")
    variables: Dict[str, str] = Field(..., description="Variables to fill in the template")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_sources: bool = Field(True, description="Include source documents in response")


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class BuilderQueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    filters_applied: Dict[str, Any]
    results_count: int


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
query_builder = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedding_service, llm_service, rag_service, query_builder

    logger.info("Starting RAG Query Service...")

    try:
        # Initialize services
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        rag_service = RAGService(embedding_service, llm_service)
        query_builder = QueryBuilder(rag_service)

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


@app.post("/query/builder", response_model=BuilderQueryResponse)
async def query_with_builder(request: BuilderQueryRequest):
    """
    Execute a structured query with filters and advanced options

    Allows filtering by metadata, setting score thresholds, and more control over results.
    """
    try:
        result = query_builder.build_query(
            query=request.query,
            filters=request.filters,
            k=request.k,
            score_threshold=request.score_threshold,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error processing builder query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/template", response_model=BuilderQueryResponse)
async def query_with_template(request: TemplateQueryRequest):
    """
    Use a pre-built query template

    Templates provide structured queries for common tasks like summaries, comparisons, etc.
    """
    try:
        result = query_builder.use_template(
            template_name=request.template_name,
            variables=request.variables,
            filters=request.filters,
            include_sources=request.include_sources
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing template query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/templates")
async def list_templates():
    """
    List available query templates

    Returns all available templates with their descriptions.
    """
    try:
        templates = query_builder.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/document/{document_id}", response_model=BuilderQueryResponse)
async def query_specific_document(document_id: str, request: QueryRequest):
    """
    Query within a specific document by document ID

    Restricts search to chunks from a single document.
    """
    try:
        result = query_builder.search_by_document(
            query=request.query,
            document_id=document_id,
            k=request.k,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error querying document {document_id}: {e}")
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
