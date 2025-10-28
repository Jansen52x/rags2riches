import chromadb
from pypdf import PdfReader
from typing import List, Dict, Any
import logging
import os
import uuid
from datetime import datetime
from config import settings
from embedding_service import EmbeddingService
from multimodal_processor import MultimodalProcessor

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents and indexing into ChromaDB"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

        # Initialize multimodal processor
        self.multimodal_processor = MultimodalProcessor(settings)

        # Connect to ChromaDB
        self.chroma_client = chromadb.HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "RAG document collection"}
            )
            logger.info(f"Connected to collection: {settings.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    def process_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process a PDF file: extract text, images, chunk, embed, and store in ChromaDB

        Args:
            file_path: Path to the PDF file
            filename: Original filename

        Returns:
            Dictionary with processing results
        """
        try:
            # Generate document ID early for image extraction
            document_id = f"doc_{uuid.uuid4().hex[:12]}"

            # Extract text from PDF
            reader = PdfReader(file_path)
            all_text = ""
            page_count = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                all_text += f"\n\n[Page {page_num + 1}]\n{text}"

            logger.info(f"Extracted text from {page_count} pages of {filename}")

            # Process images and convert to text
            multimodal_result = self.multimodal_processor.process_pdf_with_images(
                file_path, document_id, all_text
            )
            all_text = multimodal_result['content']
            has_images = multimodal_result['has_images']
            image_count = multimodal_result['image_count']

            logger.info(f"Multimodal processing: {image_count} images found")

            # Chunk the text
            chunks = self.chunk_text(all_text)

            if not chunks:
                logger.warning(f"No chunks created from {filename}")
                return {
                    "document_id": None,
                    "filename": filename,
                    "status": "failed",
                    "error": "No text content found in PDF"
                }

            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_service.embed_batch(chunks)

            # Prepare data for ChromaDB
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": datetime.utcnow().isoformat(),
                    "has_images": has_images,
                    "image_count": image_count
                }
                for i in range(len(chunks))
            ]

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )

            logger.info(f"Successfully indexed {filename} with {len(chunks)} chunks")

            return {
                "document_id": document_id,
                "filename": filename,
                "status": "indexed",
                "chunks_count": len(chunks),
                "page_count": page_count,
                "has_images": has_images,
                "image_count": image_count
            }

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents

        Returns:
            List of document metadata
        """
        try:
            # Get all items from collection
            results = self.collection.get()

            # Group by document_id
            documents = {}
            for i, metadata in enumerate(results['metadatas']):
                doc_id = metadata.get('document_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get('filename'),
                        "chunks_count": metadata.get('total_chunks', 0),
                        "indexed_at": metadata.get('indexed_at')
                    }

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its chunks from ChromaDB

        Args:
            document_id: Document ID to delete

        Returns:
            Dictionary with deletion results
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )

            if not results['ids']:
                logger.warning(f"Document {document_id} not found")
                return {
                    "status": "not_found",
                    "document_id": document_id,
                    "chunks_deleted": 0
                }

            # Delete all chunks
            self.collection.delete(ids=results['ids'])

            logger.info(f"Deleted document {document_id} with {len(results['ids'])} chunks")

            return {
                "status": "deleted",
                "document_id": document_id,
                "chunks_deleted": len(results['ids'])
            }

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    def rebuild_index(self) -> Dict[str, Any]:
        """
        Delete all documents and rebuild index from scratch

        Returns:
            Dictionary with rebuild results
        """
        try:
            # Delete collection
            self.chroma_client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)

            # Recreate collection
            self.collection = self.chroma_client.create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "RAG document collection"}
            )

            logger.info("Successfully rebuilt index")

            return {
                "status": "rebuilt",
                "message": "Index cleared. Ready for new documents."
            }

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documents

        Returns:
            Dictionary with stats
        """
        try:
            results = self.collection.get()

            # Count unique documents
            unique_docs = set()
            for metadata in results['metadatas']:
                doc_id = metadata.get('document_id')
                if doc_id:
                    unique_docs.add(doc_id)

            return {
                "total_documents": len(unique_docs),
                "total_chunks": len(results['ids']),
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }

    def health_check(self) -> bool:
        """
        Check if ChromaDB connection is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.chroma_client.heartbeat()
            logger.info("ChromaDB connection is healthy")
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False
