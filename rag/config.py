from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for RAG Query Service"""

    # ChromaDB Configuration
    CHROMADB_HOST: str = "chromadb"
    CHROMADB_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # NVIDIA NIM Configuration
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # Models
    LLM_MODEL: str = "meta/llama-3.1-70b-instruct"
    EMBEDDING_MODEL: str = "nvidia/nv-embed-v1"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    JUDGE_MODEL: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    # LLM Generation Settings
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.0

    # Query Defaults
    DEFAULT_K: int = 10

    # Service Configuration
    LOG_LEVEL: str = "INFO"
    SERVICE_PORT: int = 8001

    @property
    def CHROMADB_URL(self) -> str:
        """Get ChromaDB URL"""
        return f"http://{self.CHROMADB_HOST}:{self.CHROMADB_PORT}"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
