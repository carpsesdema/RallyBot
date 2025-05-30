# config.py
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Attempted to load .env file.")
        except ImportError:
            logger.info(
                ".env file not loaded (python-dotenv not found or no .env file). Relying on environment variables or defaults.")

        self.OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        self.OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL",
                                                     "nomic-embed-text")  # e.g., nomic-embed-text (768), mxbai-embed-large (1024)

        # Example dimension, needs to be accurate for the chosen OLLAMA_EMBEDDING_MODEL
        # For "nomic-embed-text", it's typically 768.
        # For "mxbai-embed-large", it's typically 1024.
        # This should ideally be dynamically determined or carefully configured.
        self.OLLAMA_EMBEDDING_DIMENSION: int = int(os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768"))

        self.KNOWLEDGE_BASE_DIR: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base_docs"))
        self.VECTOR_STORE_DIR: Path = Path(
            os.getenv("VECTOR_STORE_DIR", "./vector_store"))  # Directory for vector store
        self.VECTOR_STORE_FILE_NAME: str = os.getenv("VECTOR_STORE_FILE_NAME",
                                                     "avachat_faiss.index")  # Base name for index
        self.VECTOR_STORE_METADATA_FILE_NAME: str = os.getenv("VECTOR_STORE_METADATA_FILE_NAME",
                                                              "avachat_faiss_metadata.pkl")  # For chunk metadata

        # Full paths for vector store files
        self.VECTOR_STORE_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_FILE_NAME
        self.VECTOR_STORE_METADATA_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_METADATA_FILE_NAME

        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.API_SERVER_HOST: str = os.getenv("API_SERVER_HOST", "127.0.0.1")
        self.API_SERVER_PORT: int = int(os.getenv("API_SERVER_PORT", "8000"))

        self._validate_paths()
        self._initialized = True
        logger.info(
            f"Settings initialized. OLLAMA_EMBEDDING_MODEL: {self.OLLAMA_EMBEDDING_MODEL}, DIMENSION: {self.OLLAMA_EMBEDDING_DIMENSION}")

    def _validate_paths(self):
        """Validates and creates necessary directories."""
        try:
            self.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Knowledge base directory ensured: {self.KNOWLEDGE_BASE_DIR}")

            self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Vector store directory ensured: {self.VECTOR_STORE_DIR}")

        except OSError as e:
            logger.error(f"Error creating directories: {e}")
            # from utils import ConfigurationError # Avoid circular import if utils imports config
            # raise ConfigurationError(f"Could not create required directories: {e}")
            # For now, just log. A proper app might raise a ConfigurationError defined in utils.
            pass


settings = Settings()

if __name__ == '__main__':
    # Example of how to use the settings
    print(f"Ollama API URL: {settings.OLLAMA_API_URL}")
    print(f"Ollama Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
    print(f"Ollama Embedding Dimension: {settings.OLLAMA_EMBEDDING_DIMENSION}")
    print(f"Knowledge Base Dir: {settings.KNOWLEDGE_BASE_DIR}")
    print(f"Vector Store Path: {settings.VECTOR_STORE_PATH}")
    print(f"Vector Store Metadata Path: {settings.VECTOR_STORE_METADATA_PATH}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Knowledge base directory exists: {settings.KNOWLEDGE_BASE_DIR.exists()}")
    print(f"Vector store directory exists: {settings.VECTOR_STORE_DIR.exists()}")