import chromadb
from typing import List, Dict, Any
import logging
from config import settings
from embedding_service import EmbeddingService
from llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service that orchestrates retrieval and generation"""

    def __init__(self, embedding_service: EmbeddingService, llm_service: LLMService):
        self.embedding_service = embedding_service
        self.llm_service = llm_service

        # Connect to ChromaDB
        self.chroma_client = chromadb.HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=settings.CHROMA_COLLECTION_NAME
            )
            logger.info(f"Connected to existing collection: {settings.CHROMA_COLLECTION_NAME}")
        except Exception:
            logger.warning(f"Collection {settings.CHROMA_COLLECTION_NAME} not found, will be created by CRUD service")
            self.collection = None

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector database

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of search results with content, metadata, and scores
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []

        k = k or settings.DEFAULT_K

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i]
                    })

            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def query(self, query: str, k: int = None, include_sources: bool = True) -> Dict[str, Any]:
        """
        Perform RAG query: retrieve relevant documents and generate answer

        Args:
            query: User's question
            k: Number of documents to retrieve
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary containing answer and optionally source documents
        """
        if not self.collection:
            return {
                "answer": "No documents indexed yet. Please upload documents first.",
                "sources": []
            }

        try:
            # Retrieve relevant documents
            search_results = self.search(query, k)

            if not search_results:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": []
                }

            # Build context from search results
            context = "\n\n".join([
                f"Source {i+1}:\n{result['content']}"
                for i, result in enumerate(search_results)
            ])

            # Generate answer using LLM
            answer = self.llm_service.generate(query, context)

            # Prepare response
            response = {
                "answer": answer,
            }

            if include_sources:
                response["sources"] = [
                    {
                        "content": result['content'],
                        "metadata": result['metadata'],
                        "score": result['score']
                    }
                    for result in search_results
                ]

            logger.info(f"Successfully answered query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
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
