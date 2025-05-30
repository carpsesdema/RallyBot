# config.py - Updated with Gemini support
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
            logger.info("Loaded .env file.")
        except ImportError:
            logger.info(".env file not loaded (python-dotenv not found).")

        # LLM Provider Selection
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "gemini"

        # Ollama Settings
        self.OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "deepseek-llm:7b")
        self.OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.OLLAMA_EMBEDDING_DIMENSION: int = int(os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768"))

        # Google Gemini Settings
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        # Gemini embeddings are 768 dimensions by default
        self.GEMINI_EMBEDDING_DIMENSION: int = 768

        # Paths
        self.KNOWLEDGE_BASE_DIR: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base_docs"))
        self.VECTOR_STORE_DIR: Path = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
        self.VECTOR_STORE_FILE_NAME: str = os.getenv("VECTOR_STORE_FILE_NAME", "tennis_faiss.index")
        self.VECTOR_STORE_METADATA_FILE_NAME: str = os.getenv("VECTOR_STORE_METADATA_FILE_NAME",
                                                              "tennis_faiss_metadata.pkl")

        # Full paths for vector store files
        self.VECTOR_STORE_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_FILE_NAME
        self.VECTOR_STORE_METADATA_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_METADATA_FILE_NAME

        # Server settings
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.API_SERVER_HOST: str = os.getenv("API_SERVER_HOST", "127.0.0.1")
        self.API_SERVER_PORT: int = int(os.getenv("API_SERVER_PORT", "8000"))

        # RAG Settings
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "3"))

        self._validate_paths()
        self._validate_llm_settings()
        self._initialized = True

        logger.info(f"Settings initialized. LLM Provider: {self.LLM_PROVIDER}")

    def _validate_paths(self):
        """Validates and creates necessary directories."""
        try:
            self.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
            self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directories ensured: {self.KNOWLEDGE_BASE_DIR}, {self.VECTOR_STORE_DIR}")
        except OSError as e:
            logger.error(f"Error creating directories: {e}")

    def _validate_llm_settings(self):
        """Validates LLM provider settings."""
        if self.LLM_PROVIDER == "gemini":
            if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your_gemini_api_key_here":
                logger.warning("Gemini selected but no API key configured. Set GOOGLE_API_KEY in .env")
        elif self.LLM_PROVIDER == "ollama":
            logger.info(f"Using Ollama with model: {self.OLLAMA_CHAT_MODEL}")
        else:
            logger.warning(f"Unknown LLM provider: {self.LLM_PROVIDER}. Defaulting to ollama.")
            self.LLM_PROVIDER = "ollama"

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        """Returns the appropriate embedding dimension based on LLM provider."""
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_EMBEDDING_DIMENSION
        else:
            return self.OLLAMA_EMBEDDING_DIMENSION


settings = Settings()

if __name__ == '__main__':
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    if settings.LLM_PROVIDER == "gemini":
        print(f"Gemini Model: {settings.GEMINI_MODEL}")
        print(f"API Key Set: {'Yes' if settings.GOOGLE_API_KEY else 'No'}")
    else:
        print(f"Ollama Model: {settings.OLLAMA_CHAT_MODEL}")
        print(f"Ollama URL: {settings.OLLAMA_API_URL}")