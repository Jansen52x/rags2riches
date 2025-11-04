from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import shutil
from config import settings
from embedding_service import EmbeddingService
from document_service import DocumentService

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG CRUD Service",
    description="Document management service for RAG system",
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
class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_count: int
    page_count: int = None
    has_images: bool = False
    image_count: int = 0


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    indexed_at: str


class DeleteResponse(BaseModel):
    status: str
    document_id: str
    chunks_deleted: int


class RebuildResponse(BaseModel):
    status: str
    message: str


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    collection_name: str


class HealthResponse(BaseModel):
    status: str
    chromadb: bool
    embeddings: bool


# Global service instances
embedding_service = None
document_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedding_service, document_service

    logger.info("Starting RAG CRUD Service...")

    try:
        # Initialize services
        embedding_service = EmbeddingService()
        document_service = DocumentService(embedding_service)

        # Health checks
        if not embedding_service.health_check():
            logger.warning("Embedding service health check failed")

        if not document_service.health_check():
            logger.warning("ChromaDB health check failed")

        logger.info("RAG CRUD Service is ready!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.post("/documents", response_model=DocumentResponse)
async def upload_and_index_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document

    The file is processed, chunked, embedded, and stored in ChromaDB.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate file size (rough estimate)
    file_size_mb = 0
    try:
        # Save file temporarily
        temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check file size
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size_mb:.2f}MB) exceeds maximum ({settings.MAX_UPLOAD_SIZE_MB}MB)"
            )

        # Process the PDF
        logger.info(f"Processing uploaded file: {file.filename}")
        result = document_service.process_pdf(temp_path, file.filename)

        # Clean up temp file
        os.remove(temp_path)

        if result['status'] == 'failed':
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))

        return result

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentListItem])
async def list_documents():
    """
    List all indexed documents
    """
    try:
        documents = document_service.list_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks
    """
    try:
        result = document_service.delete_document(document_id)

        if result['status'] == 'not_found':
            raise HTTPException(status_code=404, detail="Document not found")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index():
    """
    Delete all documents and rebuild the index from scratch

    WARNING: This will delete all indexed documents!
    """
    try:
        result = document_service.rebuild_index()
        return result
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about indexed documents
    """
    try:
        stats = document_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check health status of all services"""
    chromadb_healthy = document_service.health_check() if document_service else False
    embeddings_healthy = embedding_service.health_check() if embedding_service else False

    status = "healthy" if all([chromadb_healthy, embeddings_healthy]) else "degraded"

    return {
        "status": status,
        "chromadb": chromadb_healthy,
        "embeddings": embeddings_healthy
    }


@app.post("/documents/json")
async def upload_json_file(file: UploadFile = File(...)):
    """
    Upload and process a JSON file

    Supports both generic JSON format and synthetic data format
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        # Save file temporarily
        temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the JSON
        logger.info(f"Processing JSON file: {file.filename}")
        result = document_service.process_json(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error processing JSON upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """
    Upload and process a CSV file

    Each row will be indexed as a separate document
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Save file temporarily
        temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the CSV
        logger.info(f"Processing CSV file: {file.filename}")
        result = document_service.process_csv(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error processing CSV upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch-pdfs")
async def batch_upload_pdfs(directory_path: str):
    """
    Process multiple PDF files from a local directory

    Args:
        directory_path: Path to directory containing PDF files
    """
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        result = document_service.process_batch_pdfs(directory_path)
        return result
    except Exception as e:
        logger.error(f"Error processing batch PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch-images")
async def batch_upload_images(directory_path: str):
    """
    Process multiple image files from a local directory using OCR

    Supports PNG, JPG, JPEG, WEBP formats

    Args:
        directory_path: Path to directory containing image files
    """
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        result = document_service.process_batch_images(directory_path)
        return result
    except Exception as e:
        logger.error(f"Error processing batch images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RAG CRUD Service",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
