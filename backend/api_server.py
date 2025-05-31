import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import os

try:
    from config import settings
    from utils import (
        setup_logger, AvaChatError, ConfigurationError, VectorStoreError,
        RAGPipelineError, LLMClientError, TextSplittingError,
        EmbeddingGenerationError, DocumentLoadingError
    )
    from backend.api_handlers import router as api_handlers_router
    from llm_interface.gemini_client import GeminiLLMClient
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import FAISSVectorStore, VectorStoreInterface
    from rag.rag_pipeline import RAGPipeline
except ImportError as e:
    # Fallback for when run in environments where full project structure isn't available
    # This allows the file to be parsed, but it won't be fully functional.
    print(f"CRITICAL Backend Import Error in api_server.py: {e}. Using dummy fallbacks.")


    class SettingsClass:
        LOG_LEVEL = "DEBUG"
        LOCAL_API_SERVER_HOST = "127.0.0.1"
        LOCAL_API_SERVER_PORT = 8000
        LLM_PROVIDER = "gemini"  # Default to gemini for fallback consistency
        GOOGLE_API_KEY = "dummy_api_key_placeholder"
        GEMINI_MODEL = "gemini-dummy-model"
        EMBEDDING_DIMENSION = 384  # A common small dimension
        VECTOR_STORE_PATH = Path("./dummy_vector_store/faiss.index")
        VECTOR_STORE_METADATA_PATH = Path("./dummy_vector_store/faiss_metadata.pkl")
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        # Ollama settings, even if not used by hardcoded Gemini, might be expected by config structure
        OLLAMA_API_URL = "http://localhost:11434"
        OLLAMA_CHAT_MODEL = "ollama-dummy-chat"
        OLLAMA_EMBEDDING_MODEL = "ollama-dummy-embed"


    settings = SettingsClass()


    def setup_logger(name, level):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fallback_logger = logging.getLogger(name)
        fallback_logger.info(f"Dummy logger for {name} @ {level} (Backend Fallback)")
        return fallback_logger


    class AvaChatError(Exception):
        pass


    class ConfigurationError(AvaChatError):
        pass


    class VectorStoreError(AvaChatError):
        pass


    class RAGPipelineError(AvaChatError):
        pass


    class LLMClientError(AvaChatError):
        pass


    class TextSplittingError(AvaChatError):
        pass


    class EmbeddingGenerationError(AvaChatError):
        pass


    class DocumentLoadingError(AvaChatError):
        pass


    class APIRouter:
        routes = []  # type: ignore


    api_handlers_router = APIRouter()  # type: ignore


    class GeminiLLMClient:
        def __init__(self, settings_obj):  # Changed parameter name for clarity
            self.settings = settings_obj  # Use passed settings
            print(f"BACKEND WARNING: Using DUMMY GeminiLLMClient with model {self.settings.GEMINI_MODEL}")

        async def close_session(self): pass

        async def list_available_models(self): return [self.settings.GEMINI_MODEL]  # type: ignore


    class DocumentLoader:
        pass


    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap): pass


    class EmbeddingGenerator:
        def __init__(self, llm_client): self.llm_client = llm_client


    class FAISSVectorStore:
        def __init__(self, embedding_dimension, index_file_path, metadata_file_path):
            self.index = None  # FAISS index object
            self.embedding_dimension = embedding_dimension
            self.index_file_path = index_file_path
            self.metadata_file_path = metadata_file_path

        def load(self): print("Dummy FAISSVectorStore: load called")

        def save(self): print("Dummy FAISSVectorStore: save called")

        def is_empty(self): return True

        def add_documents(self, chunk_embeddings): print("Dummy FAISSVectorStore: add_documents called")

        async def search_similar_chunks(self, query_embedding, top_k): return []


    class VectorStoreInterface:
        pass  # Dummy protocol base


    class RAGPipeline:
        def __init__(self, settings_obj, llm_client, document_loader, text_splitter, embedding_generator,
                     vector_store):  # Changed settings param name
            self.settings = settings_obj  # Use passed settings
            print("BACKEND WARNING: Using DUMMY RAGPipeline")

logger = logging.getLogger(__name__)  # Logger for this specific file


@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings instance should be the one imported from config, already loaded with .env
    lifespan_logger = setup_logger("AvaChatServerLifespan", settings.LOG_LEVEL)
    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence initiated.")
    lifespan_logger.info(f"Log level configured to: {settings.LOG_LEVEL}")

    # --- FORCED GEMINI INITIALIZATION ---
    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    lifespan_logger.info("!!! HARDCODING GEMINI CLIENT INITIALIZATION IN LIFESPAN !!!")
    lifespan_logger.info(f"!!! settings.LLM_PROVIDER from config.py initially: '{settings.LLM_PROVIDER}'")
    lifespan_logger.info(
        f"!!! settings.GOOGLE_API_KEY is set and not placeholder: {bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY not in ['your_gemini_api_key_here', 'dummy_api_key_placeholder'])}")
    lifespan_logger.info(f"!!! settings.GEMINI_MODEL from config.py: '{settings.GEMINI_MODEL}'")
    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        lifespan_logger.info("Lifespan: Force initializing GeminiLLMClient...")
        if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY in ['your_gemini_api_key_here',
                                                                      'dummy_api_key_placeholder']:
            error_msg = "CRITICAL CONFIGURATION ERROR: GOOGLE_API_KEY is missing or a placeholder. Gemini WILL fail."
            lifespan_logger.error(error_msg)
            raise ConfigurationError(error_msg)  # Halt server startup

        # Explicitly set LLM_PROVIDER on the settings object if RAG logic elsewhere might read it dynamically
        # This ensures consistency for any component that might inspect settings.LLM_PROVIDER later.
        original_provider_in_settings = settings.LLM_PROVIDER
        settings.LLM_PROVIDER = "gemini"
        lifespan_logger.info(
            f"Lifespan: settings.LLM_PROVIDER was '{original_provider_in_settings}', now forced to '{settings.LLM_PROVIDER}'.")

        app.state.llm_client = GeminiLLMClient(settings=settings)  # Pass the global settings object
        lifespan_logger.info(f"Lifespan: GeminiLLMClient FORCED. Type in app.state: {type(app.state.llm_client)}")

    except Exception as e:  # Catch any exception during Gemini client init
        lifespan_logger.critical(f"Lifespan: Failed to force initialize GeminiLLMClient: {e}", exc_info=True)
        # Re-raise as ConfigurationError to ensure server doesn't start in a broken state
        raise ConfigurationError(f"Hardcoded GeminiLLMClient initialization failed: {e}") from e

    # Initialize RAG Components
    try:
        lifespan_logger.info("Lifespan: Initializing RAG components (with forced Gemini client)...")
        doc_loader = DocumentLoader()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE,
                                                       chunk_overlap=settings.CHUNK_OVERLAP)
        embedding_generator = EmbeddingGenerator(llm_client=app.state.llm_client)

        # EMBEDDING_DIMENSION property in Settings class will use the (now forced) settings.LLM_PROVIDER
        embedding_dimension = settings.EMBEDDING_DIMENSION
        lifespan_logger.info(
            f"Lifespan RAG: Using EMBEDDING_DIMENSION: {embedding_dimension} (derived from LLM_PROVIDER='{settings.LLM_PROVIDER}')")

        vector_store = FAISSVectorStore(
            embedding_dimension=embedding_dimension,
            index_file_path=settings.VECTOR_STORE_PATH,
            metadata_file_path=settings.VECTOR_STORE_METADATA_PATH
        )
        vector_store.load()  # Attempt to load existing store
        app.state.vector_store = vector_store

        app.state.rag_pipeline = RAGPipeline(
            settings=settings,  # Pass the global (and now modified) settings
            llm_client=app.state.llm_client,  # Pass the forced Gemini client
            document_loader=doc_loader,
            text_splitter=text_splitter,
            embedding_generator=embedding_generator,
            vector_store=app.state.vector_store
        )
        lifespan_logger.info("Lifespan: RAGPipeline and components initialized with forced Gemini setup.")
    except Exception as e:
        lifespan_logger.critical(f"Lifespan: Failed to initialize RAG components: {e}", exc_info=True)
        raise RAGPipelineError(f"RAG component initialization failed: {e}") from e

    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence complete. Application ready.")
    yield  # Application runs here

    # === Shutdown ===
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence initiated.")
    if hasattr(app.state, 'vector_store') and app.state.vector_store:
        try:
            app.state.vector_store.save()
            lifespan_logger.info("FAISSVectorStore saved successfully on shutdown.")
        except Exception as e:
            lifespan_logger.error(f"Failed to save FAISSVectorStore during shutdown: {e}", exc_info=True)

    if hasattr(app.state, 'llm_client') and app.state.llm_client:
        try:
            await app.state.llm_client.close_session()
            lifespan_logger.info("LLM client session closed successfully on shutdown.")
        except Exception as e:
            lifespan_logger.error(f"Failed to close LLM client session during shutdown: {e}", exc_info=True)
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence complete.")


# Create FastAPI app instance
app = FastAPI(
    title="AvaChat Backend API",
    description="API server for AvaChat, handling RAG operations and LLM interactions. THIS VERSION IS HARDCODED FOR GEMINI.",
    version="0.1.1",  # Incremented version for clarity
    lifespan=lifespan
)

# Include API routers from api_handlers.py
app.include_router(api_handlers_router, prefix="/api")
logger.info("FastAPI app instance created and API router included under /api prefix.")  # Main module log


# Global exception handler for custom errors
@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError):
    # Use a logger that's guaranteed to be configured (e.g., root or lifespan's)
    logging.getLogger("AvaChatApp").error(f"Unhandled AvaChatError at API level: {exc} for request {request.url.path}",
                                          exc_info=True)
    error_code = "AVACHAT_ERROR"
    error_message = str(exc)
    status_code = 500  # Default
    if isinstance(exc, ConfigurationError):
        error_code = "CONFIGURATION_ERROR"
        status_code = 503  # Service Unavailable
    elif isinstance(exc, VectorStoreError):
        error_code = "VECTOR_STORE_ERROR"
    elif isinstance(exc, LLMClientError):
        error_code = "LLM_CLIENT_ERROR"
        status_code = 502  # Bad Gateway
    # Add more specific error mappings here if needed
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": error_code, "message": error_message}}
    )


# Root path for basic health check or welcome
@app.get("/", include_in_schema=False)
async def root():
    logging.getLogger(__name__).info("Root path '/' accessed.")
    return {"message": "Welcome to the AvaChat API! This version is HARDCODED FOR GEMINI. API docs at /docs or /redoc."}


# Main block for running the server directly (e.g., for local development)
if __name__ == "__main__":
    import uvicorn

    # Fixed: Use LOCAL_API_SERVER_PORT and LOCAL_API_SERVER_HOST instead of the old names
    port_to_use = int(os.getenv("PORT", str(settings.LOCAL_API_SERVER_PORT)))
    host_to_use = os.getenv("HOST", settings.LOCAL_API_SERVER_HOST)

    print(f"Attempting to start Uvicorn server for AvaChat API (Host: {host_to_use}, Port: {port_to_use})...")
    print(f"   >>> Main Block Check before Uvicorn: settings.LLM_PROVIDER from config.py = '{settings.LLM_PROVIDER}'")
    print(
        f"   >>> Main Block Check before Uvicorn: settings.GOOGLE_API_KEY is set = {bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY not in ['your_gemini_api_key_here', 'dummy_api_key_placeholder'])}")

    uvicorn.run(
        "backend.api_server:app",  # Path to the FastAPI app instance
        host=host_to_use,
        port=port_to_use,
        reload=False,  # KEEPING RELOAD FALSE to avoid .env loading issues with reloader
        log_level=settings.LOG_LEVEL.lower()  # Uvicorn's log level
    )
    print("Uvicorn server has been shut down or an attempt to start was made.")