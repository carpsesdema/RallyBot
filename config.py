import os
from pathlib import Path
import logging
from typing import Optional

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

        # --- LLM Provider (Hardcoded to Gemini for backend logic, but GUI might still read this) ---
        # Forcing Gemini in api_server.py's lifespan. This setting in .env is mostly for local GUI context now.
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
        if self.LLM_PROVIDER != "gemini":
            logger.warning(
                f"LLM_PROVIDER in .env is '{self.LLM_PROVIDER}', but backend is hardcoded to Gemini. This setting primarily affects local GUI behavior if it tries to adapt.")
            # self.LLM_PROVIDER = "gemini" # Optionally force it here too for GUI consistency

        # --- Google Gemini Settings ---
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Default if not in .env
        self.GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        self.GEMINI_EMBEDDING_DIMENSION: int = 768  # Standard for text-embedding-004

        # --- Ollama Settings (Present for completeness, but not used by hardcoded Gemini backend) ---
        self.OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "deepseek-llm:7b")
        self.OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.OLLAMA_EMBEDDING_DIMENSION: int = int(os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768"))

        # --- API Server Connection Settings (for GUI's ApiClient) ---
        # For local backend:
        self.LOCAL_API_SERVER_HOST: str = os.getenv("API_SERVER_HOST", "127.0.0.1")  # Renamed to avoid conflict
        self.LOCAL_API_SERVER_PORT: int = int(os.getenv("API_SERVER_PORT", "8000"))  # Renamed

        # For deployed backend on Railway (GUI will use this if set)
        # Example: REMOTE_BACKEND_URL=https://your-app-name.up.railway.app
        self.REMOTE_BACKEND_URL: Optional[str] = os.getenv("REMOTE_BACKEND_URL", None)

        # --- Paths (Primarily for backend, ensure they make sense for Railway if used there) ---
        self.KNOWLEDGE_BASE_DIR: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base_docs"))
        self.VECTOR_STORE_DIR: Path = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
        # Railway might use ephemeral storage unless a volume is attached.
        # For simplicity, if vector store is pre-built and part of repo, these paths are relative.
        self.VECTOR_STORE_FILE_NAME: str = "tennis_faiss.index"
        self.VECTOR_STORE_METADATA_FILE_NAME: str = "tennis_faiss_metadata.pkl"
        self.VECTOR_STORE_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_FILE_NAME
        self.VECTOR_STORE_METADATA_PATH: Path = self.VECTOR_STORE_DIR / self.VECTOR_STORE_METADATA_FILE_NAME

        # --- General Application Settings ---
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # --- RAG Settings ---
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "3"))

        self._validate_paths()
        self._initialized = True

        logger.info(f"Settings initialized. Effective LLM Provider for backend logic: Gemini (Hardcoded).")
        logger.info(f"   GUI will connect to REMOTE_BACKEND_URL='{self.REMOTE_BACKEND_URL}' if set, else local.")
        logger.info(f"   Gemini Model: {self.GEMINI_MODEL}")

    def _validate_paths(self):
        try:
            # These might not be strictly needed if KNOWLEDGE_BASE_DIR isn't used by deployed backend
            # or if vector store is read-only from repo.
            self.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
            self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            # logger.info(f"Directories ensured (if applicable): {self.KNOWLEDGE_BASE_DIR}, {self.VECTOR_STORE_DIR}")
        except OSError as e:
            logger.warning(f"Warning creating directories (might be normal in some environments): {e}")

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        # This property now always reflects Gemini since backend is hardcoded
        return self.GEMINI_EMBEDDING_DIMENSION


settings = Settings()

if __name__ == '__main__':
    print(f"--- Config Test ---")
    print(f"LLM Provider (from .env, GUI might use): {settings.LLM_PROVIDER}")
    print(f"Gemini Model: {settings.GEMINI_MODEL}")
    print(
        f"API Key Set: {'Yes' if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY not in ['dummy_api_key_placeholder', 'your_gemini_api_key_here'] else 'No or Placeholder'}")
    print(f"Embedding Dimension (will be Gemini's): {settings.EMBEDDING_DIMENSION}")
    print(f"Remote Backend URL for GUI: {settings.REMOTE_BACKEND_URL}")
    print(f"Local API Server Host: {settings.LOCAL_API_SERVER_HOST}")
    print(f"Local API Server Port: {settings.LOCAL_API_SERVER_PORT}")
    print(f"Vector Store Path: {settings.VECTOR_STORE_PATH}")