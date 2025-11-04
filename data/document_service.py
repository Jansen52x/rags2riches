import chromadb
from pypdf import PdfReader
from typing import List, Dict, Any, Optional
import logging
import os
import uuid
import json
import pandas as pd
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

    def process_text_data(
        self,
        text_content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process plain text or structured text data and add to ChromaDB

        Args:
            text_content: The text content to process
            document_id: Optional custom document ID
            metadata: Optional metadata to attach to chunks

        Returns:
            Dictionary with processing results
        """
        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = f"doc_{uuid.uuid4().hex[:12]}"

            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}

            # Add default metadata
            metadata.update({
                "document_id": document_id,
                "indexed_at": datetime.utcnow().isoformat()
            })

            # Chunk the text
            chunks = self.chunk_text(text_content)

            if not chunks:
                logger.warning(f"No chunks created from text content")
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "No text content to process"
                }

            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_service.embed_batch(chunks)

            # Prepare data for ChromaDB
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
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

            logger.info(f"Successfully indexed document {document_id} with {len(chunks)} chunks")

            return {
                "document_id": document_id,
                "status": "indexed",
                "chunks_count": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error processing text data: {e}")
            raise

    def process_json(self, file_path: str) -> Dict[str, Any]:
        """
        Process a JSON file and add documents to ChromaDB

        Expected JSON format:
        [
            {
                "content": "text content",
                "metadata": {"key": "value", ...}
            },
            ...
        ]

        Or for synthetic data format:
        [
            {
                "client_data": {...},
                "industry_overview": "...",
                "document_text": [tokens] or "text"
            },
            ...
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing JSON file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            total_docs = 0
            total_chunks = 0
            errors = []

            for idx, item in enumerate(data):
                try:
                    # Handle synthetic data format
                    if 'client_data' in item and 'industry_overview' in item:
                        client_data = item['client_data']

                        # Build text content
                        text_parts = [
                            f"Company: {client_data.get('company_name', '')}",
                            f"Industry: {client_data.get('industry', '')}",
                            f"Contact: {client_data.get('contact_person', '')}",
                            f"Email: {client_data.get('contact_email', '')}",
                            f"Description: {client_data.get('company_description', '')}",
                            f"Industry Overview: {item.get('industry_overview', '')}"
                        ]

                        content = "\n".join(text_parts)

                        # Prepare metadata
                        doc_metadata = {
                            "source": "synthetic_data",
                            "company_name": client_data.get('company_name', ''),
                            "industry": client_data.get('industry', ''),
                            "contact_email": client_data.get('contact_email', ''),
                            "source_file": os.path.basename(file_path)
                        }

                        # Add PDF path if present
                        if 'pdf_brochure_path' in item:
                            doc_metadata['pdf_brochure_path'] = item['pdf_brochure_path']

                        # Add image paths if present
                        if 'brochure_path' in item:
                            doc_metadata['brochure_path'] = item['brochure_path']
                        if 'flyer_path' in item:
                            doc_metadata['flyer_path'] = item['flyer_path']

                    # Handle generic format
                    elif 'content' in item:
                        content = item['content']
                        doc_metadata = item.get('metadata', {})
                        doc_metadata['source_file'] = os.path.basename(file_path)

                    else:
                        # Fallback: use entire item as text
                        content = json.dumps(item, indent=2)
                        doc_metadata = {
                            "source_file": os.path.basename(file_path),
                            "record_index": idx
                        }

                    # Process the document
                    result = self.process_text_data(
                        text_content=content,
                        metadata=doc_metadata
                    )

                    if result['status'] == 'indexed':
                        total_docs += 1
                        total_chunks += result['chunks_count']

                except Exception as e:
                    error_msg = f"Error processing record {idx}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

            return {
                "status": "completed",
                "source_file": os.path.basename(file_path),
                "documents_processed": total_docs,
                "total_chunks": total_chunks,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            raise

    def process_csv(self, file_path: str, text_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a CSV file and add documents to ChromaDB

        Args:
            file_path: Path to CSV file
            text_columns: List of column names to combine as content (if None, uses all)

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing CSV file: {file_path}")

            df = pd.read_csv(file_path)

            total_docs = 0
            total_chunks = 0
            errors = []

            for idx, row in df.iterrows():
                try:
                    # Determine which columns to use
                    if text_columns:
                        content_parts = [str(row[col]) for col in text_columns if col in df.columns]
                    else:
                        content_parts = [f"{col}: {str(row[col])}" for col in df.columns]

                    content = "\n".join(content_parts)

                    # Create metadata from all columns
                    doc_metadata = {
                        "source": "csv",
                        "source_file": os.path.basename(file_path),
                        "row_index": int(idx)
                    }

                    # Add all row data as metadata (convert to strings)
                    for col in df.columns:
                        try:
                            doc_metadata[col] = str(row[col])
                        except:
                            pass

                    # Process the document
                    result = self.process_text_data(
                        text_content=content,
                        metadata=doc_metadata
                    )

                    if result['status'] == 'indexed':
                        total_docs += 1
                        total_chunks += result['chunks_count']

                except Exception as e:
                    error_msg = f"Error processing row {idx}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

            return {
                "status": "completed",
                "source_file": os.path.basename(file_path),
                "documents_processed": total_docs,
                "total_chunks": total_chunks,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise

    def process_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single image file (PNG, JPG, etc.) using OCR

        Args:
            image_path: Path to image file
            metadata: Optional metadata to attach

        Returns:
            Dictionary with processing results
        """
        try:
            if not os.path.exists(image_path):
                return {
                    "status": "failed",
                    "error": f"Image file not found: {image_path}"
                }

            logger.info(f"Processing image: {image_path}")

            # Extract text from image using OCR
            image_text = self.multimodal_processor.process_image_to_text(image_path)

            if not image_text or not image_text.strip():
                logger.warning(f"No text extracted from image: {image_path}")
                return {
                    "status": "failed",
                    "error": "No text extracted from image"
                }

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                "source": "image",
                "image_path": image_path,
                "filename": os.path.basename(image_path)
            })

            # Process the extracted text
            result = self.process_text_data(
                text_content=image_text,
                metadata=metadata
            )

            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def process_batch_images(self, image_directory: str) -> Dict[str, Any]:
        """
        Process multiple image files from a directory

        Args:
            image_directory: Path to directory containing image files

        Returns:
            Dictionary with batch processing results
        """
        try:
            logger.info(f"Processing images from directory: {image_directory}")

            # Get all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
            image_files = [
                f for f in os.listdir(image_directory)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]

            if not image_files:
                return {
                    "status": "completed",
                    "message": "No image files found in directory",
                    "documents_processed": 0,
                    "total_chunks": 0
                }

            total_docs = 0
            total_chunks = 0
            errors = []

            for image_file in image_files:
                try:
                    image_path = os.path.join(image_directory, image_file)
                    logger.info(f"Processing image: {image_file}")

                    # Extract metadata from filename if it follows pattern
                    # e.g., "brochure_001.png" or "flyer_025.png"
                    file_metadata = {
                        "source_directory": os.path.basename(image_directory)
                    }

                    # Try to extract type and index from filename
                    base_name = os.path.splitext(image_file)[0]
                    if '_' in base_name:
                        parts = base_name.rsplit('_', 1)
                        file_metadata['material_type'] = parts[0]
                        try:
                            file_metadata['material_index'] = int(parts[1])
                        except ValueError:
                            pass

                    result = self.process_image(image_path, file_metadata)

                    if result['status'] == 'indexed':
                        total_docs += 1
                        total_chunks += result['chunks_count']
                    else:
                        errors.append(f"{image_file}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    error_msg = f"Error processing {image_file}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

            return {
                "status": "completed",
                "source_directory": image_directory,
                "documents_processed": total_docs,
                "total_chunks": total_chunks,
                "total_files": len(image_files),
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Error processing batch images: {e}")
            raise

    def process_batch_pdfs(self, pdf_directory: str) -> Dict[str, Any]:
        """
        Process multiple PDF files from a directory

        Args:
            pdf_directory: Path to directory containing PDF files

        Returns:
            Dictionary with batch processing results
        """
        try:
            logger.info(f"Processing PDFs from directory: {pdf_directory}")

            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

            if not pdf_files:
                return {
                    "status": "completed",
                    "message": "No PDF files found in directory",
                    "documents_processed": 0,
                    "total_chunks": 0
                }

            total_docs = 0
            total_chunks = 0
            errors = []

            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(pdf_directory, pdf_file)
                    logger.info(f"Processing PDF: {pdf_file}")

                    result = self.process_pdf(pdf_path, pdf_file)

                    if result['status'] == 'indexed':
                        total_docs += 1
                        total_chunks += result['chunks_count']
                    else:
                        errors.append(f"{pdf_file}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    error_msg = f"Error processing {pdf_file}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

            return {
                "status": "completed",
                "source_directory": pdf_directory,
                "documents_processed": total_docs,
                "total_chunks": total_chunks,
                "total_files": len(pdf_files),
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Error processing batch PDFs: {e}")
            raise

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
