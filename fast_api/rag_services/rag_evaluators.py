"""
Non-LLM RAG Evaluation Metrics

This module provides automated evaluation metrics that don't require LLM judges
or ground truth answers. These are useful for continuous monitoring and A/B testing.
"""

import re
import logging
import math
from typing import List, Dict, Any
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Non-LLM based RAG evaluation metrics"""

    def __init__(self):
        pass

    def evaluate_all(
        self,
        query: str,
        context: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run all non-LLM evaluation metrics

        Args:
            query: User's question
            context: Retrieved context string
            answer: Generated answer
            retrieved_docs: Optional list of retrieved documents with scores

        Returns:
            Dictionary with all metric scores
        """
        return {
            "answer_length": self.answer_length_metric(answer),
            "lexical_overlap": self.lexical_overlap(query, answer),
            "context_utilization": self.context_utilization(context, answer),
            "answer_completeness": self.answer_completeness(answer),
            "retrieval_metrics": self.retrieval_metrics(retrieved_docs) if retrieved_docs else {},
            "fluency_score": self.fluency_score(answer),
            "specificity_score": self.specificity_score(answer),
        }

    # ==================== Answer Quality Metrics ====================

    def answer_length_metric(self, answer: str) -> Dict[str, Any]:
        """
        Measure answer length characteristics

        Returns:
            - char_count: Number of characters
            - word_count: Number of words
            - sentence_count: Number of sentences
            - is_too_short: Boolean if answer seems incomplete
            - is_too_long: Boolean if answer seems verbose
        """
        char_count = len(answer)
        word_count = len(answer.split())
        sentence_count = len(re.split(r'[.!?]+', answer.strip()))

        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "is_too_short": word_count < 10,
            "is_too_long": word_count > 500,
            "avg_words_per_sentence": word_count / max(sentence_count, 1)
        }

    def lexical_overlap(self, query: str, answer: str) -> Dict[str, float]:
        """
        Measure word overlap between query and answer
        Indicates if answer is addressing the query terms

        Returns:
            - token_overlap_ratio: Percentage of query tokens in answer
            - unique_token_overlap: Percentage of unique query tokens in answer
        """
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())

        # Remove common stopwords for better signal
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_tokens = query_tokens - stopwords
        answer_tokens = answer_tokens - stopwords

        if not query_tokens:
            return {"token_overlap_ratio": 0.0, "unique_token_overlap": 0.0}

        overlap = query_tokens.intersection(answer_tokens)

        return {
            "token_overlap_ratio": len(overlap) / len(query_tokens),
            "unique_token_overlap": len(overlap)
        }

    def context_utilization(self, context: str, answer: str) -> Dict[str, float]:
        """
        Measure how much of the answer is grounded in the context
        Helps detect hallucination vs proper grounding

        Returns:
            - context_token_usage: Percentage of answer tokens that appear in context (answer grounding)
            - context_coverage: Percentage of context tokens used in answer (context utilization)
            - exact_phrase_matches: Number of multi-word phrases from context in answer
        """
        context_tokens = set(context.lower().split())
        answer_tokens = set(answer.lower().split())

        if not answer_tokens:
            return {"context_token_usage": 0.0, "context_coverage": 0.0, "exact_phrase_matches": 0}

        # How many answer tokens come from context (grounding check)
        grounded_tokens = context_tokens.intersection(answer_tokens)
        answer_grounding = len(grounded_tokens) / len(answer_tokens)

        # How much of context was used in answer (utilization check)
        context_coverage = len(grounded_tokens) / len(context_tokens) if context_tokens else 0.0

        # Find exact phrase matches (2+ word sequences)
        context_phrases = self._extract_ngrams(context.lower(), n=3)
        exact_matches = sum(1 for phrase in context_phrases if phrase in answer.lower())

        return {
            "context_token_usage": answer_grounding,  # Returns decimal (0-1), not percentage
            "context_coverage": context_coverage,
            "exact_phrase_matches": exact_matches
        }

    def answer_completeness(self, answer: str) -> Dict[str, Any]:
        """
        Heuristic checks for answer completeness

        Returns:
            - has_numbers: Boolean, contains numerical data
            - has_structure: Boolean, uses bullet points or formatting
            - ends_properly: Boolean, doesn't end abruptly
        """
        has_numbers = bool(re.search(r'\d+', answer))
        has_bullets = bool(re.search(r'[•\-*]\s', answer))
        has_lists = bool(re.search(r'\d+\.\s', answer))

        # Check if answer ends with incomplete sentence
        ends_with_punct = answer.strip()[-1] in '.!?)' if answer.strip() else False

        return {
            "has_numbers": has_numbers,
            "has_structure": has_bullets or has_lists,
            "ends_properly": ends_with_punct,
            "completeness_score": sum([has_numbers, has_bullets or has_lists, ends_with_punct]) / 3.0
        }

    def fluency_score(self, answer: str) -> Dict[str, float]:
        """
        Measure text fluency without LLM

        Returns:
            - avg_word_length: Average word length (longer = more complex)
            - repetition_ratio: Ratio of repeated words (higher = more repetitive)
        """
        words = answer.lower().split()

        if not words:
            return {"avg_word_length": 0.0, "repetition_ratio": 0.0}

        avg_word_length = sum(len(w) for w in words) / len(words)

        # Detect repetition
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words) if words else 0

        return {
            "avg_word_length": avg_word_length,
            "repetition_ratio": repetition_ratio,
            "fluency_score": 1.0 - min(repetition_ratio * 2, 1.0)  # Penalize high repetition
        }

    def specificity_score(self, answer: str) -> Dict[str, Any]:
        """
        Measure answer specificity (numbers, names, dates, facts)

        Returns:
            - has_dates: Boolean
            - has_names: Boolean (capitalized words)
            - has_metrics: Boolean (percentages, currency)
            - specificity_score: Overall 0-1 score (as decimal, not percentage)
        """
        has_dates = bool(re.search(r'\b(19|20)\d{2}\b|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', answer))
        has_names = bool(re.search(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', answer))  # Proper names
        has_metrics = bool(re.search(r'[\d,]+%|[$€£]\s*[\d,]+|[\d,]+\s*(?:million|billion|thousand)', answer))
        has_numbers = bool(re.search(r'\d+', answer))

        specificity_components = [has_dates, has_names, has_metrics, has_numbers]

        return {
            "has_dates": has_dates,
            "has_names": has_names,
            "has_metrics": has_metrics,
            "specificity_score": sum(specificity_components) / len(specificity_components)  # Returns decimal (0-1)
        }

    # ==================== Retrieval Quality Metrics ====================

    def retrieval_metrics(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate retrieval quality based on scores and diversity

        Args:
            retrieved_docs: List of dicts with 'score' and 'content' keys
                           May also contain 'similarity_score' and 'rerank_score'

        Returns:
            Metrics about retrieval quality with normalized scores (0-1 range)
        """
        if not retrieved_docs:
            return {
                "num_retrieved": 0,
                "avg_retrieval_score": 0.0,
                "score_variance": 0.0,
                "content_diversity": 0.0
            }

        # Extract scores - prefer similarity_score if available, otherwise use 'score'
        # This handles both old and new RAG service formats
        scores = []
        for doc in retrieved_docs:
            # Try to get similarity_score first (normalized 0-1)
            if 'similarity_score' in doc:
                scores.append(doc['similarity_score'])
            else:
                scores.append(doc.get('score', 0))
        
        # Detect if scores are cross-encoder logits (typically large negative/positive values)
        # Cross-encoder scores usually range from -150 to +150
        scores_array = np.array(scores) if scores else np.array([])
        
        if len(scores_array) > 0:
            score_min = scores_array.min()
            score_max = scores_array.max()
            score_range = score_max - score_min
            
            # If scores look like cross-encoder logits (outside normal 0-1 range)
            if score_min < -1 or score_max > 10:
                logger.info(f"Detected cross-encoder scores: range=[{score_min:.2f}, {score_max:.2f}], normalizing...")
                # Apply sigmoid normalization to convert to 0-1 range
                normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]
                scores = normalized_scores
                logger.info(f"Normalized score range: [{min(scores):.3f}, {max(scores):.3f}]")

        # Measure content diversity (lexical diversity between docs)
        diversity = self._calculate_document_diversity(retrieved_docs)

        return {
            "num_retrieved": len(retrieved_docs),
            "avg_retrieval_score": np.mean(scores) if scores else 0.0,
            "min_retrieval_score": min(scores) if scores else 0.0,
            "max_retrieval_score": max(scores) if scores else 0.0,
            "score_variance": np.var(scores) if len(scores) > 1 else 0.0,
            "content_diversity": diversity
        }

    def _calculate_document_diversity(self, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate how diverse the retrieved documents are
        Higher diversity = more varied information sources

        Returns:
            Diversity score 0-1 (higher = more diverse)
        """
        if len(docs) < 2:
            return 0.0

        # Calculate pairwise Jaccard distance between documents
        doc_tokens = []
        for doc in docs:
            content = doc.get('content', '')
            tokens = set(content.lower().split())
            doc_tokens.append(tokens)

        similarities = []
        for i in range(len(doc_tokens)):
            for j in range(i + 1, len(doc_tokens)):
                intersection = len(doc_tokens[i].intersection(doc_tokens[j]))
                union = len(doc_tokens[i].union(doc_tokens[j]))
                jaccard_sim = intersection / union if union > 0 else 0
                similarities.append(jaccard_sim)

        # Diversity is inverse of similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity

    # ==================== Helper Methods ====================

    def _extract_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract n-grams from text"""
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def format_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format evaluation metrics into a readable report

        Args:
            metrics: Dictionary returned by evaluate_all()

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("NON-LLM RAG EVALUATION REPORT")
        report.append("=" * 60)

        # Answer Quality
        report.append("\n📝 Answer Quality:")
        length = metrics['answer_length']
        report.append(f"  - Length: {length['word_count']} words, {length['sentence_count']} sentences")
        report.append(f"  - Completeness: {metrics['answer_completeness']['completeness_score']:.2%}")
        report.append(f"  - Fluency: {metrics['fluency_score']['fluency_score']:.2%}")
        report.append(f"  - Specificity: {metrics['specificity_score']['specificity_score']:.2%}")

        # Query Relevance
        report.append("\n🎯 Query Relevance:")
        overlap = metrics['lexical_overlap']
        report.append(f"  - Query token overlap: {overlap['token_overlap_ratio']:.2%}")
        report.append(f"  - Unique tokens matched: {overlap['unique_token_overlap']}")

        # Context Usage
        report.append("\n📄 Context Utilization:")
        context = metrics['context_utilization']
        report.append(f"  - Answer grounded in context: {context['context_token_usage']:.2%}")
        report.append(f"  - Context coverage: {context['context_coverage']:.2%}")
        report.append(f"  - Exact phrase matches: {context['exact_phrase_matches']}")

        # Retrieval Quality
        if metrics['retrieval_metrics']:
            report.append("\n🔍 Retrieval Quality:")
            retrieval = metrics['retrieval_metrics']
            report.append(f"  - Docs retrieved: {retrieval['num_retrieved']}")
            report.append(f"  - Avg retrieval score: {retrieval['avg_retrieval_score']:.3f}")
            report.append(f"  - Score range: [{retrieval['min_retrieval_score']:.3f}, {retrieval['max_retrieval_score']:.3f}]")
            report.append(f"  - Content diversity: {retrieval['content_diversity']:.2%}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    evaluator = RAGEvaluator()

    # Example evaluation
    query = "What was the company's Q4 revenue?"
    context = "According to the Q4 financial report, revenue reached $10 million, representing 20% growth year-over-year."
    answer = "The company's Q4 revenue was $10 million, showing 20% growth compared to the previous year."

    metrics = evaluator.evaluate_all(query, context, answer)
    print(evaluator.format_evaluation_report(metrics))