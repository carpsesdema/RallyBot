# backend/api_server.py - COMPLETE WORKING VERSION
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import os

try:
    from config import settings, tennis_config, validate_tennis_config
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
    from rag.vector_store import FAISSVectorStore
    from rag.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"CRITICAL Backend Import Error in api_server.py: {e}. Using dummy fallbacks.")

    # Fallback dummy classes
    from pathlib import Path


    class SettingsClass:
        LOG_LEVEL = "DEBUG"
        LOCAL_API_SERVER_HOST = "0.0.0.0"  # Changed to 0.0.0.0
        LOCAL_API_SERVER_PORT = 8000
        LLM_PROVIDER = "gemini"
        GOOGLE_API_KEY = "dummy_api_key_placeholder"
        GEMINI_MODEL = "gemini-dummy-model"
        EMBEDDING_DIMENSION = 768
        KNOWLEDGE_BASE_DIR = Path("./dummy_kb_dir_on_volume")
        VECTOR_STORE_DIR = Path("./dummy_vector_store_on_volume")
        VECTOR_STORE_PATH = Path("./dummy_vector_store/faiss.index")
        VECTOR_STORE_METADATA_PATH = Path("./dummy_vector_store/faiss_metadata.pkl")
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        OLLAMA_API_URL = "http://localhost:11434"
        OLLAMA_CHAT_MODEL = "ollama-dummy-chat"
        OLLAMA_EMBEDDING_MODEL = "ollama-dummy-embed"


    settings = SettingsClass()
    tennis_config = type('obj', (), {'enable_fallback_data': True})()


    def validate_tennis_config():
        return {"has_primary_api": False}


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


    def setup_logger(name, level):
        logging.basicConfig(level=level)
        return logging.getLogger(name)


    # Dummy router
    from fastapi import APIRouter

    api_handlers_router = APIRouter()


    @api_handlers_router.get("/health")
    async def health():
        return {"status": "fallback_mode", "message": "Running with dummy components"}


    # Dummy classes for missing components
    class GeminiLLMClient:
        def __init__(self, settings): pass

        async def close_session(self): pass


    class DocumentLoader:
        pass


    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap): pass


    class EmbeddingGenerator:
        def __init__(self, llm_client): pass


    class FAISSVectorStore:
        def __init__(self, embedding_dimension, index_file_path, metadata_file_path): pass

        def load(self): pass

        def save(self): pass


    class RAGPipeline:
        def __init__(self, **kwargs): pass

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    lifespan_logger = setup_logger("TennisServerLifespan", settings.LOG_LEVEL)
    lifespan_logger.info("Tennis Server Lifespan: Startup sequence initiated.")
    lifespan_logger.info(f"Log level configured to: {settings.LOG_LEVEL}")

    # Create essential directories based on settings (works with Railway's /data volume)
    try:
        kb_path = Path(settings.KNOWLEDGE_BASE_DIR)
        vs_path = Path(settings.VECTOR_STORE_DIR)

        lifespan_logger.info(f"Ensuring KNOWLEDGE_BASE_DIR exists: {kb_path}")
        kb_path.mkdir(parents=True, exist_ok=True)

        lifespan_logger.info(f"Ensuring VECTOR_STORE_DIR exists: {vs_path}")
        vs_path.mkdir(parents=True, exist_ok=True)

        lifespan_logger.info("Essential directories are ready.")

    except Exception as e:
        lifespan_logger.critical(f"CRITICAL ERROR: Failed to create directories: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to create essential directories: {e}") from e

    # Initialize LLM Client
    try:
        lifespan_logger.info("Initializing LLM Client...")
        app.state.llm_client = GeminiLLMClient(settings=settings)
        lifespan_logger.info(f"LLM Client initialized: {type(app.state.llm_client)}")
    except Exception as e:
        lifespan_logger.critical(f"Failed to initialize LLM Client: {e}", exc_info=True)
        raise ConfigurationError(f"LLM Client initialization failed: {e}") from e

    # Initialize RAG Components
    try:
        lifespan_logger.info("Initializing RAG components...")
        doc_loader = DocumentLoader()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        embedding_generator = EmbeddingGenerator(llm_client=app.state.llm_client)

        vector_store = FAISSVectorStore(
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            index_file_path=settings.VECTOR_STORE_PATH,
            metadata_file_path=settings.VECTOR_STORE_METADATA_PATH
        )
        vector_store.load()
        app.state.vector_store = vector_store

        app.state.rag_pipeline = RAGPipeline(
            settings=settings,
            llm_client=app.state.llm_client,
            document_loader=doc_loader,
            text_splitter=text_splitter,
            embedding_generator=embedding_generator,
            vector_store=app.state.vector_store
        )
        lifespan_logger.info("RAGPipeline and components initialized successfully.")

    except Exception as e:
        lifespan_logger.critical(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise RAGPipelineError(f"RAG component initialization failed: {e}") from e

    lifespan_logger.info("Tennis Server Lifespan: Startup sequence complete. Application ready.")
    yield  # Application runs here

    # Shutdown
    lifespan_logger.info("Tennis Server Lifespan: Shutdown sequence initiated.")

    if hasattr(app.state, 'vector_store') and app.state.vector_store:
        try:
            app.state.vector_store.save()
            lifespan_logger.info("Vector store saved successfully on shutdown.")
        except Exception as e:
            lifespan_logger.error(f"Failed to save vector store during shutdown: {e}", exc_info=True)

    if hasattr(app.state, 'llm_client') and app.state.llm_client:
        try:
            await app.state.llm_client.close_session()
            lifespan_logger.info("LLM client session closed successfully on shutdown.")
        except Exception as e:
            lifespan_logger.error(f"Failed to close LLM client session during shutdown: {e}", exc_info=True)

    lifespan_logger.info("Tennis Server Lifespan: Shutdown sequence complete.")


# Create FastAPI app instance
app = FastAPI(
    title="Tennis Intelligence Backend API",
    description="API server for Tennis Intelligence, handling RAG operations and LLM interactions with professional tennis data.",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routers
app.include_router(api_handlers_router, prefix="/api")
logger.info("FastAPI app instance created and API router included under /api prefix.")


# Global exception handler for custom errors
@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError):
    logging.getLogger("TennisApp").error(f"Unhandled AvaChatError at API level: {exc} for request {request.url.path}",
                                         exc_info=True)

    error_code = "TENNIS_ERROR"
    error_message = str(exc)
    status_code = 500

    if isinstance(exc, ConfigurationError):
        error_code = "CONFIGURATION_ERROR"
        status_code = 503
    elif isinstance(exc, VectorStoreError):
        error_code = "VECTOR_STORE_ERROR"
        status_code = 500
    elif isinstance(exc, RAGPipelineError):
        error_code = "RAG_PIPELINE_ERROR"
        status_code = 500
    elif isinstance(exc, LLMClientError):
        error_code = "LLM_CLIENT_ERROR"
        status_code = 500
    elif isinstance(exc, TextSplittingError):
        error_code = "TEXT_SPLITTING_ERROR"
        status_code = 500
    elif isinstance(exc, EmbeddingGenerationError):
        error_code = "EMBEDDING_GENERATION_ERROR"
        status_code = 500
    elif isinstance(exc, DocumentLoadingError):
        error_code = "DOCUMENT_LOADING_ERROR"
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": error_code, "message": error_message}}
    )


# Root path for basic health check
@app.get("/", include_in_schema=False)
async def root():
    logging.getLogger(__name__).info("Root path '/' accessed.")
    return {
        "message": "Welcome to the Tennis Intelligence API! Professional tennis data and analysis.",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


# Main block for running the server directly (mostly for local dev)
if __name__ == "__main__":
    import uvicorn

    # Production services like Railway provide the PORT env var.
    # Default to 8000 for local development.
    port_to_use = int(os.getenv("PORT", settings.LOCAL_API_SERVER_PORT))

    # In production, HOST should be 0.0.0.0 to accept connections from outside the container.
    # The Procfile with gunicorn handles this, but this makes running locally more flexible.
    host_to_use = os.getenv("HOST", "0.0.0.0")

    print(f"Starting Tennis Intelligence API Server...")
    print(f"Host: {host_to_use}")
    print(f"Port: {port_to_use}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"LLM Provider: {getattr(settings, 'LLM_PROVIDER', 'gemini')}")
    print(f"API Documentation: http://{settings.LOCAL_API_SERVER_HOST}:{port_to_use}/docs")

    uvicorn.run(
        "backend.api_server:app",
        host=host_to_use,
        port=port_to_use,
        reload=False,  # Reload should be False for production/stable testing
        log_level=settings.LOG_LEVEL.lower()
    )