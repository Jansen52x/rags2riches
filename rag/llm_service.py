from openai import OpenAI
import logging
from config import settings
from typing import Dict, Tuple, Optional
import json
import csv
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating answers using NVIDIA NIM LLM"""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY
        )
        self.model = settings.LLM_MODEL
        self.judge_model = settings.JUDGE_MODEL
        
        # Setup evaluation output directory and file
        self.eval_dir = Path(getattr(settings, 'EVAL_OUTPUT_DIR', 'evaluations'))
        self.eval_dir.mkdir(exist_ok=True)
        self.eval_file = self.eval_dir / f"rag_evaluations_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Initialize CSV file with headers if it doesn't exist
        if not self.eval_file.exists():
            self._initialize_csv()
        
        logger.info(f"Initialized NVIDIA LLM Service with model: {self.model}")
        logger.info(f"Initialized Judge model: {self.judge_model}")
        logger.info(f"Evaluation results will be saved to: {self.eval_file}")

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = None,
        temperature: float = None,
        evaluate: bool = True  # Added parameter to control evaluation
    ) -> str:
        """
        Generate an answer using the LLM based on query and context
        Automatically evaluates the response by default

        Args:
            query: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            evaluate: Whether to run automatic evaluation (default: True)

        Returns:
            Generated answer as string
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
            if evaluate:
                try:
                    self.evaluate_rag_response(query, context, answer, save_to_csv=True)
                except Exception as e:
                    logger.error(f"Evaluation failed but continuing: {e}")
            
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
        save_to_csv: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate RAG pipeline using LLM-as-a-judge on three dimensions:
        1. Faithfulness: Is the answer factually consistent with the context?
        2. Answer Relevance: Does the answer address the user's question?
        3. Context Relevance: Is the retrieved context useful for the question?

        Args:
            query: User's original question
            context: Retrieved context from RAG
            answer: Generated answer
            save_to_csv: Whether to save results to CSV file

        Returns:
            Dictionary with scores and explanations for each dimension
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

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert RAG system evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.0,  # Use deterministic evaluation
                max_tokens=1000
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in evaluation_text:
                evaluation_text = evaluation_text.split("```json")[1].split("```")[0].strip()
            elif "```" in evaluation_text:
                evaluation_text = evaluation_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(evaluation_text)
            
            logger.info(f"RAG evaluation completed - Faithfulness: {evaluation['faithfulness']['score']}, "
                       f"Answer Relevance: {evaluation['answer_relevance']['score']}, "
                       f"Context Relevance: {evaluation['context_relevance']['score']}")
            
            # Save to CSV if requested
            if save_to_csv:
                self._save_evaluation_to_csv(query, context, answer, evaluation)
            
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.error(f"Raw response: {evaluation_text}")
            # Return a fallback evaluation
            evaluation = {
                "faithfulness": {"score": 0, "explanation": "Evaluation failed"},
                "answer_relevance": {"score": 0, "explanation": "Evaluation failed"},
                "context_relevance": {"score": 0, "explanation": "Evaluation failed"},
                "overall_assessment": "Evaluation failed due to parsing error",
                "error": str(e)
            }
            if save_to_csv:
                self._save_evaluation_to_csv(query, context, answer, evaluation)
            return evaluation
        except Exception as e:
            logger.error(f"Error during RAG evaluation: {e}")
            evaluation = {
                "faithfulness": {"score": 0, "explanation": "Evaluation failed"},
                "answer_relevance": {"score": 0, "explanation": "Evaluation failed"},
                "context_relevance": {"score": 0, "explanation": "Evaluation failed"},
                "overall_assessment": "Evaluation failed due to error",
                "error": str(e)
            }
            if save_to_csv:
                self._save_evaluation_to_csv(query, context, answer, evaluation)
            return evaluation

    def generate_with_evaluation(
        self,
        query: str,
        context: str,
        max_tokens: int = None,
        temperature: float = None,
        evaluate: bool = True
    ) -> Tuple[str, Dict[str, any]]:
        """
        Generate an answer and optionally evaluate it
        Returns both answer and evaluation results

        Args:
            query: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            evaluate: Whether to run evaluation

        Returns:
            Tuple of (answer, evaluation_results)
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
            
            evaluation = None
            if evaluate:
                evaluation = self.evaluate_rag_response(query, context, answer)
            
            return answer, evaluation
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

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