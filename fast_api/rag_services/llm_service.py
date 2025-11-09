from openai import OpenAI
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating answers using NVIDIA NIM LLM"""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY
        )
        self.model = settings.LLM_MODEL
        logger.info(f"Initialized NVIDIA LLM Service with model: {self.model}")

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate an answer using the LLM based on query and context

        Args:
            query: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens to generate (default from config)
            temperature: Sampling temperature (default from config)

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
            return answer
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
            self.generate("test", "test", max_tokens=10)
            logger.info("NVIDIA LLM service is healthy")
            return True
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False
