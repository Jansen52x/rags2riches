from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for CRUD Service"""

    # ChromaDB Configuration
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # NVIDIA NIM Configuration
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # Models
    EMBEDDING_MODEL: str = "nvidia/nv-embed-v1"

    # Document Processing
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 50

    # Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 100

    # Service Configuration
    LOG_LEVEL: str = "INFO"
    SERVICE_PORT: int = 8002

    @property
    def CHROMADB_URL(self) -> str:
        """Get ChromaDB URL"""
        return f"http://{self.CHROMADB_HOST}:{self.CHROMADB_PORT}"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
