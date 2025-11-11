"""
RAG Services Module

This module provides Retrieval-Augmented Generation (RAG) functionality including:
- Embedding generation via NVIDIA NIM
- LLM-based answer generation
- Vector search with ChromaDB
- Query building and filtering
- Cross-encoder reranking
- Evaluation metrics (LLM-as-judge and non-LLM)
"""

from .embedding_service import EmbeddingService
from .llm_service import LLMService
from .rag_service import RAGService
from .query_builder import QueryBuilder
from .rag_evaluators import RAGEvaluator

__all__ = [
    "EmbeddingService",
    "LLMService",
    "RAGService",
    "QueryBuilder",
    "RAGEvaluator"
]
