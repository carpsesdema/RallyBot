# backend/api_server.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path # Make sure Path is imported
import os

try:
    from config import settings # Your global settings object
    from utils import (
        setup_logger, AvaChatError, ConfigurationError, VectorStoreError,
        RAGPipelineError, LLMClientError, TextSplittingError,
        EmbeddingGenerationError, DocumentLoadingError
    )
    from backend.api_handlers import router as api_handlers_router
    from llm_interface.gemini_client import GeminiLLMClient
    # Ollama client might be needed if you switch, keep imports if they were there
    # from llm_interface.ollama_client import OllamaLLMClient
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import FAISSVectorStore, VectorStoreInterface # Assuming FAISSVectorStore is your choice
    from rag.rag_pipeline import RAGPipeline
except ImportError as e:
    # Fallback for when run in environments where full project structure isn't available
    # This allows the file to be parsed, but it won't be fully functional.
    print(f"CRITICAL Backend Import Error in api_server.py: {e}. Using dummy fallbacks.")
    # ... (your existing fallback dummy classes from api_server.py) ...
    # Ensure Path is available in fallback if needed for dummy settings
    from pathlib import Path as DummyPath # type: ignore

    class SettingsClass:
        LOG_LEVEL = "DEBUG"
        LOCAL_API_SERVER_HOST = "127.0.0.1"
        LOCAL_API_SERVER_PORT = 8000
        LLM_PROVIDER = "gemini"
        GOOGLE_API_KEY = "dummy_api_key_placeholder"
        GEMINI_MODEL = "gemini-dummy-model"
        EMBEDDING_DIMENSION = 384
        # --- THESE ARE IMPORTANT FOR THE LIFESPAN LOGIC ---
        KNOWLEDGE_BASE_DIR = DummyPath("./dummy_kb_dir_on_volume")
        VECTOR_STORE_DIR = DummyPath("./dummy_vector_store_on_volume")
        # ---
        VECTOR_STORE_PATH = DummyPath("./dummy_vector_store/faiss.index") # This should use VECTOR_STORE_DIR
        VECTOR_STORE_METADATA_PATH = DummyPath("./dummy_vector_store/faiss_metadata.pkl") # This should use VECTOR_STORE_DIR
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        OLLAMA_API_URL = "http://localhost:11434"
        OLLAMA_CHAT_MODEL = "ollama-dummy-chat"
        OLLAMA_EMBEDDING_MODEL = "ollama-dummy-embed"

    settings = SettingsClass()
    # ... (rest of your fallback dummy classes)


logger = logging.getLogger(__name__)  # Logger for this specific file

@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings instance should be the one imported from config, already loaded with .env
    lifespan_logger = setup_logger("AvaChatServerLifespan", settings.LOG_LEVEL)
    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence initiated.")
    lifespan_logger.info(f"Log level configured to: {settings.LOG_LEVEL}")

    # --- STEP 1: ENSURE DIRECTORIES ON VOLUME EXIST ---
    # This should happen BEFORE RAG components try to use these paths.
    try:
        # These paths come from your config.py (via .env on Railway, pointing to the volume mount)
        # e.g., KNOWLEDGE_BASE_DIR=/data/tennis_kb, VECTOR_STORE_DIR=/data/vector_store
        kb_path_on_volume = Path(settings.KNOWLEDGE_BASE_DIR)
        vs_path_on_volume = Path(settings.VECTOR_STORE_DIR)

        lifespan_logger.info(f"Ensuring KNOWLEDGE_BASE_DIR exists on volume: {kb_path_on_volume}")
        kb_path_on_volume.mkdir(parents=True, exist_ok=True)

        lifespan_logger.info(f"Ensuring VECTOR_STORE_DIR exists on volume: {vs_path_on_volume}")
        vs_path_on_volume.mkdir(parents=True, exist_ok=True)
        # This also implicitly ensures the parent directory for
        # settings.VECTOR_STORE_PATH and settings.VECTOR_STORE_METADATA_PATH exists,
        # as they are defined relative to VECTOR_STORE_DIR in your config.py.

        lifespan_logger.info("Essential directories for knowledge base and vector store ensured on volume.")

    except Exception as e:
        lifespan_logger.critical(
            f"CRITICAL ERROR: Failed to create essential directories on volume "
            f"(KB: {settings.KNOWLEDGE_BASE_DIR}, VS: {settings.VECTOR_STORE_DIR}). "
            f"Error: {e}",
            exc_info=True
        )
        # This is a fatal error for the app if these paths are needed.
        # Depending on your desired behavior, you might want to raise an error to stop startup.
        raise ConfigurationError(
            f"Failed to create essential directories on volume. KB='{settings.KNOWLEDGE_BASE_DIR}', VS='{settings.VECTOR_STORE_DIR}'. Error: {e}"
        ) from e
    # --- END OF STEP 1 ---

    # --- FORCED GEMINI INITIALIZATION ---
    # (Your existing logic for hardcoding Gemini)
    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    lifespan_logger.info("!!! HARDCODING GEMINI CLIENT INITIALIZATION IN LIFESPAN !!!")
    # ... (rest of your Gemini hardcoding info logs) ...
    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        lifespan_logger.info("Lifespan: Force initializing GeminiLLMClient...")
        # ... (your Gemini client initialization logic) ...
        app.state.llm_client = GeminiLLMClient(settings=settings)
        lifespan_logger.info(f"Lifespan: GeminiLLMClient FORCED. Type in app.state: {type(app.state.llm_client)}")
    except Exception as e:
        lifespan_logger.critical(f"Lifespan: Failed to force initialize GeminiLLMClient: {e}", exc_info=True)
        raise ConfigurationError(f"Hardcoded GeminiLLMClient initialization failed: {e}") from e

    # Initialize RAG Components
    try:
        lifespan_logger.info("Lifespan: Initializing RAG components (with forced Gemini client)...")
        doc_loader = DocumentLoader()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        embedding_generator = EmbeddingGenerator(llm_client=app.state.llm_client)
        embedding_dimension = settings.EMBEDDING_DIMENSION
        lifespan_logger.info(
            f"Lifespan RAG: Using EMBEDDING_DIMENSION: {embedding_dimension} (derived from LLM_PROVIDER='{settings.LLM_PROVIDER}')"
        )

        # FAISSVectorStore will use paths like /data/vector_store/tennis_faiss.index
        # which are now ensured to have their parent directory /data/vector_store created.
        vector_store = FAISSVectorStore(
            embedding_dimension=embedding_dimension,
            index_file_path=settings.VECTOR_STORE_PATH, # e.g., /data/vector_store/tennis_faiss.index
            metadata_file_path=settings.VECTOR_STORE_METADATA_PATH # e.g., /data/vector_store/tennis_faiss_metadata.pkl
        )
        vector_store.load()  # Attempt to load existing store from volume
        app.state.vector_store = vector_store

        app.state.rag_pipeline = RAGPipeline(
            settings=settings,
            llm_client=app.state.llm_client,
            document_loader=doc_loader,
            text_splitter=text_splitter,
            embedding_generator=embedding_generator,
            vector_store=app.state.vector_store
        )
        lifespan_logger.info("Lifespan: RAGPipeline and components initialized with forced Gemini setup.")

        # --- Optional: Automatic Ingestion if Vector Store is Empty ---
        # Be cautious with this, especially if ingestion is slow or resource-intensive.
        # You might only want this for the very first startup.
        # if app.state.vector_store.is_empty() and Path(settings.KNOWLEDGE_BASE_DIR).exists() and any(Path(settings.KNOWLEDGE_BASE_DIR).iterdir()):
        #     lifespan_logger.info(f"Vector store is empty and KNOWLEDGE_BASE_DIR ({settings.KNOWLEDGE_BASE_DIR}) has content. Triggering initial ingestion...")
        #     try:
        #         # This directory_path_str should be settings.KNOWLEDGE_BASE_DIR
        #         # which points to the path on the volume (e.g., /data/tennis_kb)
        #         await app.state.rag_pipeline.ingest_documents_from_directory(str(settings.KNOWLEDGE_BASE_DIR))
        #         lifespan_logger.info(f"Automatic initial ingestion from '{settings.KNOWLEDGE_BASE_DIR}' completed.")
        #     except Exception as ingest_err:
        #         lifespan_logger.error(f"Automatic initial ingestion from '{settings.KNOWLEDGE_BASE_DIR}' failed: {ingest_err}", exc_info=True)
        # else:
        #     if not app.state.vector_store.is_empty():
        #         lifespan_logger.info("Vector store is not empty. Skipping automatic initial ingestion.")
        #     else:
        #         lifespan_logger.info(f"Vector store is empty, but KNOWLEDGE_BASE_DIR ({settings.KNOWLEDGE_BASE_DIR}) is empty or doesn't exist. Skipping auto-ingestion.")
        # --- End Optional Automatic Ingestion ---

    except Exception as e:
        lifespan_logger.critical(f"Lifespan: Failed to initialize RAG components: {e}", exc_info=True)
        raise RAGPipelineError(f"RAG component initialization failed: {e}") from e

    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence complete. Application ready.")
    yield  # Application runs here

    # === Shutdown ===
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence initiated.")
    if hasattr(app.state, 'vector_store') and app.state.vector_store:
        try:
            app.state.vector_store.save() # Save to the volume path
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
    version="0.1.1",
    lifespan=lifespan # Assign the lifespan context manager
)

# Include API routers from api_handlers.py
app.include_router(api_handlers_router, prefix="/api")
logger.info("FastAPI app instance created and API router included under /api prefix.")


# Global exception handler for custom errors
@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError):
    # ... (your existing exception handler logic) ...
    logging.getLogger("AvaChatApp").error(f"Unhandled AvaChatError at API level: {exc} for request {request.url.path}",
                                          exc_info=True)
    error_code = "AVACHAT_ERROR"
    error_message = str(exc)
    status_code = 500  # Default
    if isinstance(exc, ConfigurationError):
        error_code = "CONFIGURATION_ERROR"
        status_code = 503
    # ... (other error type checks) ...
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": error_code, "message": error_message}}
    )


# Root path for basic health check or welcome
@app.get("/", include_in_schema=False)
async def root():
    logging.getLogger(__name__).info("Root path '/' accessed.")
    return {"message": "Welcome to the AvaChat API! This version is HARDCODED FOR GEMINI. API docs at /docs or /redoc."}


# Main block for running the server directly
if __name__ == "__main__":
    import uvicorn
    port_to_use = int(os.getenv("PORT", str(settings.LOCAL_API_SERVER_PORT)))
    host_to_use = os.getenv("HOST", settings.LOCAL_API_SERVER_HOST)
    print(f"Attempting to start Uvicorn server for AvaChat API (Host: {host_to_use}, Port: {port_to_use})...")
    # ... (your existing main block logs) ...
    uvicorn.run(
        "backend.api_server:app",
        host=host_to_use,
        port=port_to_use,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
    print("Uvicorn server has been shut down or an attempt to start was made.")