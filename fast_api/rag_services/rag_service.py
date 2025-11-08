import chromadb
from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder
from fast_api.config import settings
from embedding_service import EmbeddingService
from llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service that orchestrates retrieval, re-ranking, and generation"""

    def __init__(self, embedding_service: EmbeddingService, llm_service: LLMService):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        
        self.reranker = None
        if settings.USE_RERANKER:
            try:
                # Load the cross-encoder model
                self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
                logger.info(f"Loaded re-ranker model: {settings.RERANKER_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to load re-ranker model '{settings.RERANKER_MODEL_NAME}': {e}")
                self.reranker = None  # Ensure it's None if loading fails

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
        Perform similarity search in the vector database, followed by re-ranking.

        Args:
            query: Search query string
            k: Number of *initial* results to retrieve (before re-ranking)

        Returns:
            List of re-ranked and filtered search results
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []

        # k (or DEFAULT_INITIAL_K) is the number of docs to fetch *before* re-ranking
        initial_k = k or settings.DEFAULT_INITIAL_K

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        # Convert L2 distance to cosine similarity-like score (0-1)
                        'score': 1 - results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i]
                    })
            
            if not formatted_results:
                logger.info("No results found in vector store.")
                return []

            if self.reranker:
                logger.info(f"Re-ranking {len(formatted_results)} initial documents...")
                
                # Prepare pairs for the cross-encoder: [query, doc_content]
                pairs = [[query, doc['content']] for doc in formatted_results]
                
                # Get relevance scores
                scores = self.reranker.predict(pairs)
                
                # Add scores to documents
                for doc, score in zip(formatted_results, scores):
                    doc['rerank_score'] = float(score)
                
                # Sort by new re-rank score (descending)
                formatted_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # Filter to the final top-k (e.g., RERANK_TOP_K = 3)
                final_k = settings.RERANK_TOP_K
                reranked_results = formatted_results[:final_k]
                
                logger.info(f"Found {len(reranked_results)} re-ranked results for query: {query[:50]}...")
                return reranked_results

            logger.info(f"Found {len(formatted_results)} results (no re-ranking) for query: {query[:50]}...")
            # If no reranker, return results from vector search, slicing to final_k
            return formatted_results[:settings.RERANK_TOP_K]

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def query(self, query: str, k: int = None, include_sources: bool = True) -> Dict[str, Any]:
        """
        Perform RAG query: retrieve, re-rank, and generate.

        Args:
            query: User's question
            k: Number of *initial* documents to retrieve
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary containing answer and optionally sources
        """
        if not self.collection:
            return {
                "answer": "No documents indexed yet. Please upload documents first.",
                "sources": []
            }

        try:
            # Retrieve and re-rank relevant documents
            initial_k = k or settings.DEFAULT_INITIAL_K
            search_results = self.search(query, k=initial_k)

            if not search_results:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": []
                }

            # Build context from (re-ranked) search results
            context = "\n\n".join([
                f"Source {i+1} (ID: {result['id']}):\n{result['content']}"
                for i, result in enumerate(search_results)
            ])

            # Generate answer using LLM
            answer = self.llm_service.generate(query, context)

            response = {
                "answer": answer,
            }

            if include_sources:
                response["sources"] = [
                    {
                        "content": result['content'],
                        "metadata": result['metadata'],
                        "id": result['id'],
                        "score": result.get('rerank_score', result.get('score', 0.0)) # Use rerank_score if available
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