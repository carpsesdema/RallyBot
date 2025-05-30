# backend/api_server.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse  # For custom exception handling if needed

# Attempt to import project-specific modules
try:
    from config import settings  # Global settings instance
    from utils import setup_logger, AvaChatError, ConfigurationError, VectorStoreError, RAGPipelineError, LLMClientError

    # API Handlers
    from backend.api_handlers import router as api_handlers_router

    # LLM Interface - Import both clients
    from llm_interface.ollama_client import OllamaLLMClient
    from llm_interface.gemini_client import GeminiLLMClient

    # RAG Components
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter  # Using the full name here
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import FAISSVectorStore  # Concrete implementation
    from rag.rag_pipeline import RAGPipeline

except ImportError as e:
    print(
        f"CRITICAL Import Error in backend/api_server.py: {e}. Ensure all modules are correctly defined and accessible.")


    # Define dummy classes and objects to allow parsing for incremental dev,
    # though the server won't be functional.
    class Settings:
        LOG_LEVEL = "INFO";
        API_SERVER_HOST = "127.0.0.1";
        API_SERVER_PORT = 8000
        LLM_PROVIDER = "ollama"
        OLLAMA_API_URL = "http://localhost:11434"
        GOOGLE_API_KEY = ""
        EMBEDDING_DIMENSION = 768  # CRITICAL: Must match chosen embedding model
        VECTOR_STORE_PATH = "./dummy_vector_store/faiss.index"
        VECTOR_STORE_METADATA_PATH = "./dummy_vector_store/faiss_metadata.pkl"
        # Add other settings if they are directly accessed in lifespan


    settings = Settings()  # Instantiate dummy


    def setup_logger(name, level):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        l = logging.getLogger(name)
        l.info(f"Dummy logger for {name} @ {level}")
        return l


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


    class APIRouter:
        def __init__(self): self.routes = []
    api_handlers_router = APIRouter()  # Dummy router


    class OllamaLLMClient:
        async def close_session(self):
            pass

    class GeminiLLMClient:
        async def close_session(self):
            pass


    class DocumentLoader:
        pass


    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            pass


    class EmbeddingGenerator:
        pass


    class FAISSVectorStore:
        def __init__(self, embedding_dimension, index_file_path, metadata_file_path):
            pass

        def load(self):
            pass

        def save(self):
            pass


    class RAGPipeline:
        pass

# Setup a logger for the API server module itself
# The root logger is configured by setup_logger called within the lifespan manager.
# This logger instance is for api_server.py specific messages.
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to handle application startup and shutdown logic.
    Initializes and cleans up resources like LLM clients, RAG pipelines, Vector Stores.
    """
    # === Startup ===
    # Configure root logger (and get one for this context)
    # This should be one of the very first things.
    lifespan_logger = setup_logger("AvaChatServerLifespan", settings.LOG_LEVEL)
    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence initiated.")
    lifespan_logger.info(f"Log level set to: {settings.LOG_LEVEL}")

    # Initialize LLM Client based on provider
    try:
        if settings.LLM_PROVIDER == "gemini":
            lifespan_logger.info(f"Initializing GeminiLLMClient...")
            app.state.llm_client = GeminiLLMClient(settings=settings)
            lifespan_logger.info("GeminiLLMClient initialized successfully.")
        else:  # Default to Ollama
            lifespan_logger.info(f"Initializing OllamaLLMClient for URL: {settings.OLLAMA_API_URL}...")
            app.state.llm_client = OllamaLLMClient(settings=settings)
            lifespan_logger.info("OllamaLLMClient initialized successfully.")
    except Exception as e:
        lifespan_logger.critical(f"Failed to initialize LLM client: {e}", exc_info=True)
        # Depending on severity, could raise an error to prevent server start
        raise ConfigurationError(f"LLM client initialization failed: {e}") from e

    # Initialize RAG Components
    try:
        lifespan_logger.info("Initializing RAG components...")
        doc_loader = DocumentLoader()
        # Ensure chunk_size and chunk_overlap are appropriate. These could also come from settings.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedding_generator = EmbeddingGenerator(llm_client=app.state.llm_client)

        # Use the correct embedding dimension based on provider
        embedding_dimension = settings.EMBEDDING_DIMENSION  # This property handles both providers
        lifespan_logger.info(f"Initializing FAISSVectorStore. Dimension: {embedding_dimension}")
        lifespan_logger.info(f"FAISS Index Path: {settings.VECTOR_STORE_PATH}")
        lifespan_logger.info(f"FAISS Metadata Path: {settings.VECTOR_STORE_METADATA_PATH}")

        vector_store = FAISSVectorStore(
            embedding_dimension=embedding_dimension,  # CRITICAL: Must match embedding model
            index_file_path=settings.VECTOR_STORE_PATH,
            metadata_file_path=settings.VECTOR_STORE_METADATA_PATH
        )

        lifespan_logger.info("Attempting to load existing vector store...")
        try:
            vector_store.load()  # Attempt to load existing index
            if vector_store.is_empty():
                lifespan_logger.info("Vector store loaded but is empty (or new).")
            else:
                lifespan_logger.info(
                    f"Successfully loaded vector store with {getattr(vector_store.index, 'ntotal', 0)} items.")
        except FileNotFoundError:
            lifespan_logger.info("Vector store files not found. A new store will be created on first save.")
        except VectorStoreError as vse:  # Catch specific errors from vector_store.load()
            lifespan_logger.warning(
                f"VectorStoreError during load: {vse}. Proceeding with potentially empty/new store.", exc_info=True)
        except Exception as e:  # Catch any other unexpected error during load
            lifespan_logger.error(f"Unexpected error loading vector store: {e}. Proceeding with empty/new store.",
                                  exc_info=True)

        app.state.vector_store = vector_store  # Make vector_store available for shutdown save

        lifespan_logger.info("Initializing RAGPipeline...")
        app.state.rag_pipeline = RAGPipeline(
            settings=settings,
            llm_client=app.state.llm_client,
            document_loader=doc_loader,
            text_splitter=text_splitter,
            embedding_generator=embedding_generator,
            vector_store=app.state.vector_store
        )
        lifespan_logger.info("RAGPipeline and all components initialized successfully.")

    except ConfigurationError as e:  # Catch config errors from component init
        lifespan_logger.critical(f"Configuration error during RAG component initialization: {e}", exc_info=True)
        raise  # Re-raise to stop server if essential config is missing
    except Exception as e:
        lifespan_logger.critical(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise RAGPipelineError(f"RAG component initialization failed: {e}") from e  # Or a more generic server error

    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence complete. Application ready.")
    yield
    # === Shutdown ===
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence initiated.")

    # Save Vector Store
    if hasattr(app.state, 'vector_store') and app.state.vector_store:
        try:
            lifespan_logger.info("Saving FAISSVectorStore...")
            app.state.vector_store.save()
            lifespan_logger.info("FAISSVectorStore saved successfully.")
        except Exception as e:
            lifespan_logger.error(f"Failed to save FAISSVectorStore during shutdown: {e}", exc_info=True)
    else:
        lifespan_logger.warning("Vector store not found in app.state during shutdown. Nothing to save.")

    # Close LLM Client Session
    if hasattr(app.state, 'llm_client') and app.state.llm_client:
        try:
            lifespan_logger.info("Closing LLM client session...")
            await app.state.llm_client.close_session()
            lifespan_logger.info("LLM client session closed successfully.")
        except Exception as e:
            lifespan_logger.error(f"Failed to close LLM client session during shutdown: {e}", exc_info=True)
    else:
        lifespan_logger.warning("LLM client not found in app.state during shutdown.")

    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence complete.")


# Initialize FastAPI Application with the lifespan context manager
app = FastAPI(
    title="AvaChat Backend API",
    description="API server for AvaChat, handling RAG operations and LLM interactions.",
    version="0.1.0",
    lifespan=lifespan
)

# Include API routers
app.include_router(api_handlers_router, prefix="/api")
logger.info("API router included with prefix /api.")


# Optional: Define a custom global exception handler for AvaChatError or others
# This can be useful to ensure all errors are returned in a consistent format
# if HTTPErrors from handlers aren't sufficient.
@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError):
    logger.error(f"Unhandled AvaChatError at API level: {exc}", exc_info=True)
    # Default error code and message if not more specific from the exception
    error_code = "AVACHAT_ERROR"
    error_message = str(exc)
    status_code = 500  # Internal Server Error by default

    if isinstance(exc, ConfigurationError):
        error_code = "CONFIGURATION_ERROR"
        status_code = 503  # Service Unavailable
    elif isinstance(exc, VectorStoreError):
        error_code = "VECTOR_STORE_ERROR"
    elif isinstance(exc, LLMClientError):
        error_code = "LLM_CLIENT_ERROR"
        status_code = 502  # Bad Gateway
    # Add more specific error mapping here if needed

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {  # Matches ApiErrorResponse structure
                "code": error_code,
                "message": error_message
            }
        }
    )


@app.get("/", include_in_schema=False)
async def root():
    logger.info("Root path '/' accessed.")
    return {"message": "Welcome to the AvaChat API! Documentation available at /docs or /redoc."}


if __name__ == "__main__":
    import uvicorn

    # This block allows running the server directly using `python backend/api_server.py`
    # The logger here is the module-level logger defined at the top of this file.
    logger.info("Attempting to start Uvicorn server for AvaChat API...")
    uvicorn.run(
        "backend.api_server:app",  # Path to the FastAPI app instance
        host=settings.API_SERVER_HOST,
        port=settings.API_SERVER_PORT,
        reload=True,  # Enable auto-reload for development (True might be problematic if not run with watchfiles)
        # Consider `reload_dirs` if you want to specify directories to watch.
        # Often set to False for production or when using external process managers.
        log_level=settings.LOG_LEVEL.lower()  # Uvicorn's log level
    )
    logger.info("Uvicorn server has been started (or an attempt was made).")