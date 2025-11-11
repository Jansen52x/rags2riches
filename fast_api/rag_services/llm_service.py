from openai import OpenAI
import logging
import sys
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union, List, Any
import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logger = logging.getLogger(__name__)

# Import RAGEvaluator for non-LLM metrics
try:
    from .rag_evaluators import RAGEvaluator
except ImportError:
    RAGEvaluator = None
    logger.warning("RAGEvaluator not available for non-LLM metrics")


class LLMService:
    """Service for generating answers using NVIDIA NIM LLM"""

    def __init__(self):
        # NVIDIA NIM client for RAG generation
        self.client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY
        )
        self.model = settings.LLM_MODEL

        # Gemini client for independent judging (avoid model bias)
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.judge_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info(f"Initialized Gemini judge model: gemini-2.0-flash-exp")
        else:
            logger.warning("GOOGLE_API_KEY not found. Evaluation will be disabled.")
            self.judge_model = None

        # Setup evaluation output directory and file
        self.eval_dir = Path(getattr(settings, 'EVAL_OUTPUT_DIR', 'evaluations'))
        self.eval_dir.mkdir(exist_ok=True)
        self.eval_file = self.eval_dir / f"rag_evaluations_{datetime.now().strftime('%Y%m%d')}.csv"

        # Initialize CSV file with headers if it doesn't exist
        if not self.eval_file.exists():
            self._initialize_csv()

        # Initialize non-LLM evaluator
        self.non_llm_evaluator = RAGEvaluator() if RAGEvaluator else None

        logger.info(f"Initialized NVIDIA LLM Service with model: {self.model}")
        logger.info(f"Evaluation results will be saved to: {self.eval_file}")
        if self.non_llm_evaluator:
            logger.info("Non-LLM evaluator initialized")

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = None,
        temperature: float = None,
        evaluate: bool = True,
        return_evaluation: bool = False,
        retrieved_docs: List[Dict[str, Any]] = None,
        reranking_info: Dict[str, Any] = None
    ) -> Union[str, Tuple[str, Dict[str, any]]]:
        """
        Generate an answer using the LLM based on query and context
        Automatically evaluates the response by default

        Args:
            query: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            evaluate: Whether to run automatic evaluation (default: True)
            return_evaluation: Whether to return evaluation results (default: False)
            retrieved_docs: Optional list of retrieved documents for retrieval metrics
            reranking_info: Optional dict with reranking information (before/after scores)

        Returns:
            If return_evaluation is False: Generated answer as string
            If return_evaluation is True: Tuple of (answer, evaluation_results)
        """
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE

        system_prompt = """You are an expert sales briefing assistant. You are helping a salesperson prepare for a client meeting.
Use the following pieces of retrieved context to answer the question.

Your goal is to be factual, concise, and directly useful.
- Extract key facts, strategies, names, and numbers.
- Structure your answer clearly. Use bullet points if helpful.
- If the information is not in the context, state that clearly. DO NOT make up information.

Question: {question}

Context: {context}

Answer:"""

        user_prompt = f"""Context: {context}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            answer = response.choices[0].message.content
            logger.info(f"Generated answer for query: {query[:50]}...")

            # Automatically evaluate if requested
            evaluation = None
            if evaluate:
                try:
                    evaluation = self.evaluate_rag_response(
                        query,
                        context,
                        answer,
                        save_to_csv=True,
                        retrieved_docs=retrieved_docs,
                        reranking_info=reranking_info
                    )
                except Exception as e:
                    logger.error(f"Evaluation failed but continuing: {e}")
            elif not self.judge_model and not self.non_llm_evaluator:
                logger.warning("No evaluators available")

            # Return based on return_evaluation flag
            if return_evaluation:
                return answer, evaluation
            else:
                return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _initialize_csv(self):
        """Initialize the CSV file with headers"""
        headers = [
            'timestamp',
            'query',
            'context_preview',
            'answer_preview',
            'faithfulness_score',
            'faithfulness_explanation',
            'answer_relevance_score',
            'answer_relevance_explanation',
            'context_relevance_score',
            'context_relevance_explanation',
            'overall_assessment',
            'error'
        ]

        with open(self.eval_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

        logger.info(f"Initialized evaluation CSV: {self.eval_file}")

    def _save_evaluation_to_csv(
        self,
        query: str,
        context: str,
        answer: str,
        evaluation: Dict[str, any]
    ):
        """
        Save evaluation results to CSV file

        Args:
            query: User's original question
            context: Retrieved context
            answer: Generated answer
            evaluation: Evaluation results dictionary
        """
        try:
            # Truncate long texts for CSV preview (first 200 chars)
            context_preview = context[:200] + "..." if len(context) > 200 else context
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer

            row = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'context_preview': context_preview.replace('\n', ' '),
                'answer_preview': answer_preview.replace('\n', ' '),
                'faithfulness_score': evaluation.get('faithfulness', {}).get('score', 0),
                'faithfulness_explanation': evaluation.get('faithfulness', {}).get('explanation', ''),
                'answer_relevance_score': evaluation.get('answer_relevance', {}).get('score', 0),
                'answer_relevance_explanation': evaluation.get('answer_relevance', {}).get('explanation', ''),
                'context_relevance_score': evaluation.get('context_relevance', {}).get('score', 0),
                'context_relevance_explanation': evaluation.get('context_relevance', {}).get('explanation', ''),
                'overall_assessment': evaluation.get('overall_assessment', ''),
                'error': evaluation.get('error', '')
            }

            # Append to CSV
            with open(self.eval_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

            logger.info(f"Saved evaluation to CSV: {self.eval_file}")

        except Exception as e:
            logger.error(f"Failed to save evaluation to CSV: {e}")

    def evaluate_rag_response(
        self,
        query: str,
        context: str,
        answer: str,
        save_to_csv: bool = True,
        retrieved_docs: List[Dict[str, Any]] = None,
        reranking_info: Dict[str, Any] = None
    ) -> Dict[str, any]:
        """
        Evaluate RAG pipeline using both LLM-as-judge and non-LLM metrics:

        LLM-as-Judge (Gemini):
        1. Faithfulness: Is the answer factually consistent with the context?
        2. Answer Relevance: Does the answer address the user's question?
        3. Context Relevance: Is the retrieved context useful for the question?

        Non-LLM Metrics:
        - Answer quality (length, fluency, specificity)
        - Context utilization
        - Retrieval metrics (if retrieved_docs provided)
        - Reranking effectiveness (if reranking_info provided)

        Args:
            query: User's original question
            context: Retrieved context from RAG
            answer: Generated answer
            save_to_csv: Whether to save results to CSV file
            retrieved_docs: Optional list of retrieved documents with scores
            reranking_info: Optional dict with 'before_rerank' and 'after_rerank' doc lists

        Returns:
            Dictionary with both LLM and non-LLM evaluation scores
        """
        evaluation_prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems. Evaluate the following on three dimensions:

**Question:** {query}

**Retrieved Context:** {context}

**Generated Answer:** {answer}

---

Evaluate on these three dimensions:

1. **Faithfulness (1-5)**: How factually consistent is the generated answer with the retrieved context?
   - 5: All claims in the answer are directly supported by the context
   - 4: Most claims are supported, minor unsupported details
   - 3: Some claims are supported, some are not
   - 2: Few claims are supported by context
   - 1: Answer contradicts or ignores the context

2. **Answer Relevance (1-5)**: How well does the answer address the user's question?
   - 5: Directly and completely answers the question
   - 4: Mostly answers the question, minor gaps
   - 3: Partially answers the question
   - 2: Tangentially related to the question
   - 1: Does not answer the question

3. **Context Relevance (1-5)**: How relevant is the retrieved context for answering the question?
   - 5: Context is highly relevant and contains all needed information
   - 4: Context is mostly relevant with some useful information
   - 3: Context is somewhat relevant
   - 2: Context is barely relevant
   - 1: Context is not relevant to the question

Respond ONLY with valid JSON in this exact format:
{{
  "faithfulness": {{
    "score": <1-5>,
    "explanation": "<brief explanation>"
  }},
  "answer_relevance": {{
    "score": <1-5>,
    "explanation": "<brief explanation>"
  }},
  "context_relevance": {{
    "score": <1-5>,
    "explanation": "<brief explanation>"
  }},
  "overall_assessment": "<brief overall assessment>"
}}"""

        # Initialize combined evaluation result
        combined_evaluation = {}

        # 1. LLM-as-Judge Evaluation (Gemini)
        if self.judge_model:
            try:
                response = self.judge_model.generate_content(
                    f"You are an expert RAG system evaluator. Respond only with valid JSON.\n\n{evaluation_prompt}"
                )

                evaluation_text = response.text

                # Parse JSON response
                if "```json" in evaluation_text:
                    evaluation_text = evaluation_text.split("```json")[1].split("```")[0].strip()
                elif "```" in evaluation_text:
                    evaluation_text = evaluation_text.split("```")[1].split("```")[0].strip()

                llm_evaluation = json.loads(evaluation_text)
                combined_evaluation['llm_judge'] = llm_evaluation

                logger.info(f"LLM Judge - Faithfulness: {llm_evaluation['faithfulness']['score']}/5, "
                           f"Answer Relevance: {llm_evaluation['answer_relevance']['score']}/5")

            except Exception as e:
                logger.error(f"LLM judge evaluation failed: {e}")
                combined_evaluation['llm_judge'] = {
                    "error": str(e),
                    "faithfulness": {"score": 0, "explanation": "Evaluation failed"},
                    "answer_relevance": {"score": 0, "explanation": "Evaluation failed"},
                    "context_relevance": {"score": 0, "explanation": "Evaluation failed"}
                }

        # 2. Non-LLM Metrics
        if self.non_llm_evaluator:
            try:
                non_llm_metrics = self.non_llm_evaluator.evaluate_all(
                    query, context, answer, retrieved_docs
                )
                combined_evaluation['non_llm_metrics'] = non_llm_metrics

                logger.info(f"Non-LLM - Context Utilization: {non_llm_metrics['context_utilization']['context_token_usage']:.1%}, "
                           f"Specificity: {non_llm_metrics['specificity_score']['specificity_score']:.1%}")

            except Exception as e:
                logger.error(f"Non-LLM metrics failed: {e}")
                combined_evaluation['non_llm_metrics'] = {"error": str(e)}

        # 3. Reranking Effectiveness Analysis
        if reranking_info:
            try:
                reranking_metrics = self._analyze_reranking_effectiveness(reranking_info)
                combined_evaluation['reranking_metrics'] = reranking_metrics

                logger.info(f"Reranking - Score improvement: {reranking_metrics.get('avg_score_improvement', 0):.3f}")

            except Exception as e:
                logger.error(f"Reranking analysis failed: {e}")
                combined_evaluation['reranking_metrics'] = {"error": str(e)}

        # Save to CSV if requested
        if save_to_csv:
            self._save_evaluation_to_csv(query, context, answer, combined_evaluation)

        return combined_evaluation

    def _analyze_reranking_effectiveness(self, reranking_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of reranking by comparing before/after

        Args:
            reranking_info: Dict with 'before_rerank' and 'after_rerank' lists
                           Each list contains dicts with 'id', 'score', 'content'

        Returns:
            Dict with reranking metrics
        """
        before_docs = reranking_info.get('before_rerank', [])
        after_docs = reranking_info.get('after_rerank', [])

        if not before_docs or not after_docs:
            return {"error": "Missing before/after rerank data"}

        # Calculate metrics
        before_scores = [doc.get('score', 0) for doc in before_docs]
        after_scores = [doc.get('score', 0) for doc in after_docs]

        # Get top-k docs from both
        k = min(len(before_docs), len(after_docs))
        before_top_k_ids = [doc['id'] for doc in before_docs[:k]]
        after_top_k_ids = [doc['id'] for doc in after_docs[:k]]

        # Calculate overlap (how many docs stayed in top-k)
        top_k_overlap = len(set(before_top_k_ids).intersection(set(after_top_k_ids)))
        overlap_ratio = top_k_overlap / k if k > 0 else 0

        # Calculate rank changes
        rank_changes = []
        for doc_id in after_top_k_ids:
            before_rank = before_top_k_ids.index(doc_id) if doc_id in before_top_k_ids else None
            after_rank = after_top_k_ids.index(doc_id)
            if before_rank is not None:
                rank_changes.append(before_rank - after_rank)  # Positive = moved up

        avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0

        return {
            "before_avg_score": sum(before_scores) / len(before_scores) if before_scores else 0,
            "after_avg_score": sum(after_scores) / len(after_scores) if after_scores else 0,
            "avg_score_improvement": (sum(after_scores) / len(after_scores) - sum(before_scores) / len(before_scores)) if before_scores and after_scores else 0,
            "top_k_overlap_ratio": overlap_ratio,
            "avg_rank_change": avg_rank_change,
            "docs_promoted": sum(1 for change in rank_changes if change > 0),
            "docs_demoted": sum(1 for change in rank_changes if change < 0),
            "reranking_had_effect": overlap_ratio < 1.0
        }

    def health_check(self) -> bool:
        """
        Check if the LLM service is accessible

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Use evaluate=False for health check to avoid unnecessary evaluation
            self.generate("test", "test", max_tokens=10, evaluate=False)
            logger.info("NVIDIA LLM service is healthy")
            return True
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False
