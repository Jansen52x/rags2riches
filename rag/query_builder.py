"""
Simple Query Builder for RAG System

Provides structured querying with filters, metadata search, and query templates.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Simple query builder with filtering and templates"""

    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.templates = self._init_templates()

    def _init_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pre-built query templates"""
        return {
            "summary": {
                "description": "Get a summary of a specific document",
                "query_template": "Provide a summary of the main points in this document: {document_name}",
                "k": 10
            },
            "find_facts": {
                "description": "Find specific facts or data points",
                "query_template": "What are the key facts about {topic}?",
                "k": 5
            },
            "compare": {
                "description": "Compare information across documents",
                "query_template": "Compare {aspect} between different sources",
                "k": 8
            },
            "detailed": {
                "description": "Get detailed information on a topic",
                "query_template": "Provide detailed information about {topic}",
                "k": 10
            }
        }

    def build_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Build and execute a structured query with filters

        Args:
            query: The search query
            filters: Metadata filters (e.g., {"document_id": "doc_123", "filename": "report.pdf"})
            k: Number of results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            include_sources: Whether to include source documents

        Returns:
            Query results with answer and sources
        """
        logger.info(f"Building query with filters: {filters}")

        try:
            # Perform search with optional filters
            search_results = self._filtered_search(
                query=query,
                filters=filters,
                k=k,
                score_threshold=score_threshold
            )

            if not search_results:
                return {
                    "answer": "No results found matching your query and filters.",
                    "sources": [] if include_sources else None,
                    "filters_applied": filters or {}
                }

            # Build context from filtered results
            context = "\n\n".join([
                f"Source {i+1}:\n{result['content']}"
                for i, result in enumerate(search_results)
            ])

            # Generate answer using LLM
            answer = self.rag_service.llm_service.generate(query, context)

            # Prepare response
            response = {
                "answer": answer,
                "filters_applied": filters or {},
                "results_count": len(search_results)
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

            logger.info(f"Query completed with {len(search_results)} results")
            return response

        except Exception as e:
            logger.error(f"Error building query: {e}")
            raise

    def _filtered_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search with metadata filters and score threshold

        Args:
            query: Search query
            filters: Metadata filters
            k: Number of results
            score_threshold: Minimum score threshold

        Returns:
            Filtered search results
        """
        # Get initial search results
        results = self.rag_service.search(query, k)

        # Apply metadata filters if provided
        if filters:
            results = self._apply_metadata_filters(results, filters)

        # Apply score threshold if provided
        if score_threshold is not None:
            results = [r for r in results if r.get('score', 0) >= score_threshold]

        logger.info(f"Filtered search returned {len(results)} results")
        return results

    def _apply_metadata_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to search results

        Args:
            results: Search results to filter
            filters: Dictionary of metadata filters

        Returns:
            Filtered results
        """
        filtered_results = []

        for result in results:
            metadata = result.get('metadata', {})
            matches = True

            for key, value in filters.items():
                # Handle nested metadata
                if key == 'custom_metadata' and isinstance(value, dict):
                    custom = metadata.get('custom_metadata', {})
                    for custom_key, custom_value in value.items():
                        if custom.get(custom_key) != custom_value:
                            matches = False
                            break
                # Handle regular metadata
                elif metadata.get(key) != value:
                    matches = False
                    break

            if matches:
                filtered_results.append(result)

        logger.debug(f"Metadata filters reduced results from {len(results)} to {len(filtered_results)}")
        return filtered_results

    def use_template(
        self,
        template_name: str,
        variables: Dict[str, str],
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Use a pre-built query template

        Args:
            template_name: Name of the template to use
            variables: Variables to fill in the template
            filters: Optional metadata filters
            include_sources: Whether to include sources

        Returns:
            Query results
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")

        template = self.templates[template_name]
        query = template['query_template'].format(**variables)
        k = template.get('k')

        logger.info(f"Using template '{template_name}' with query: {query}")

        return self.build_query(
            query=query,
            filters=filters,
            k=k,
            include_sources=include_sources
        )

    def list_templates(self) -> Dict[str, str]:
        """
        List available query templates

        Returns:
            Dictionary of template names and descriptions
        """
        return {
            name: template['description']
            for name, template in self.templates.items()
        }

    def add_custom_template(
        self,
        name: str,
        query_template: str,
        description: str,
        k: int = 5
    ) -> None:
        """
        Add a custom query template

        Args:
            name: Template name
            query_template: Query string with {placeholders}
            description: Template description
            k: Default number of results
        """
        self.templates[name] = {
            "description": description,
            "query_template": query_template,
            "k": k
        }
        logger.info(f"Added custom template: {name}")

    def search_by_document(
        self,
        query: str,
        document_id: str,
        k: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Search within a specific document

        Args:
            query: Search query
            document_id: Document ID to search within
            k: Number of results
            include_sources: Whether to include sources

        Returns:
            Query results from the specified document
        """
        return self.build_query(
            query=query,
            filters={"document_id": document_id},
            k=k,
            include_sources=include_sources
        )

    def search_by_filename(
        self,
        query: str,
        filename: str,
        k: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Search within a specific file

        Args:
            query: Search query
            filename: Filename to search within
            k: Number of results
            include_sources: Whether to include sources

        Returns:
            Query results from the specified file
        """
        return self.build_query(
            query=query,
            filters={"filename": filename},
            k=k,
            include_sources=include_sources
        )
