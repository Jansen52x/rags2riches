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
    VISION_MODEL: str = "microsoft/phi-3-vision-128k-instruct"

    # Document Processing
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300

    # Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 100

    # Multimodal Settings
    ENABLE_OCR: bool = True
    ENABLE_IMAGE_CAPTIONING: bool = False  # Set to True to use NVIDIA vision model for image descriptions
    IMAGE_STORAGE_PATH: str = "image_store"
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png", "webp"]
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Path to Tesseract executable

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
