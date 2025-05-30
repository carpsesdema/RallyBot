import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path

try:
    from config import settings
    from utils import setup_logger, AvaChatError, ConfigurationError, VectorStoreError, RAGPipelineError, \
        LLMClientError, TextSplittingError, EmbeddingGenerationError, DocumentLoadingError
    from backend.api_handlers import router as api_handlers_router
    from llm_interface.gemini_client import GeminiLLMClient
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import FAISSVectorStore, VectorStoreInterface
    from rag.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"CRITICAL Backend Import Error in api_server.py: {e}. Using dummy fallbacks.")


    class SettingsClass:
        LOG_LEVEL = "DEBUG"
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000
        LLM_PROVIDER = "gemini"
        GOOGLE_API_KEY = "dummy_api_key_placeholder"
        GEMINI_MODEL = "gemini-dummy-model"
        EMBEDDING_DIMENSION = 384
        VECTOR_STORE_PATH = Path("./dummy_vector_store/faiss.index")
        VECTOR_STORE_METADATA_PATH = Path("./dummy_vector_store/faiss_metadata.pkl")
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        OLLAMA_API_URL = ""
        OLLAMA_CHAT_MODEL = ""
        OLLAMA_EMBEDDING_MODEL = ""


    settings = SettingsClass()


    def setup_logger(name, level):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        l = logging.getLogger(name)
        l.info(f"Dummy logger for {name} @ {level} (Backend Fallback)")
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


    class TextSplittingError(AvaChatError):
        pass


    class EmbeddingGenerationError(AvaChatError):
        pass


    class DocumentLoadingError(AvaChatError):
        pass


    class APIRouter:
        routes = []


    api_handlers_router = APIRouter()


    class GeminiLLMClient:
        def __init__(self, settings):
            self.settings = settings
            print("BACKEND WARNING: Using DUMMY GeminiLLMClient")

        async def close_session(self): pass

        async def list_available_models(self): return [self.settings.GEMINI_MODEL]


    class DocumentLoader:
        pass


    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap): pass


    class EmbeddingGenerator:
        def __init__(self, llm_client): self.llm_client = llm_client


    class FAISSVectorStore:
        def __init__(self, embedding_dimension, index_file_path, metadata_file_path): self.index = None

        def load(self): pass

        def save(self): pass

        def is_empty(self): return True


    class VectorStoreInterface:
        pass


    class RAGPipeline:
        def __init__(self, settings, llm_client, document_loader, text_splitter, embedding_generator,
                     vector_store): pass

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lifespan_logger = setup_logger("AvaChatServerLifespan", settings.LOG_LEVEL)
    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence initiated.")
    lifespan_logger.info(f"Log level configured to: {settings.LOG_LEVEL}")

    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    lifespan_logger.info(f"!!! HARDCODING GEMINI CLIENT INITIALIZATION !!!")
    lifespan_logger.info(
        f"!!! Original settings.LLM_PROVIDER = '{settings.LLM_PROVIDER}' (will be overridden for client choice)")
    lifespan_logger.info(
        f"!!! Using settings.GOOGLE_API_KEY set = {bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != 'your_gemini_api_key_here' and settings.GOOGLE_API_KEY != 'dummy_api_key_placeholder')}")
    lifespan_logger.info(f"!!! Using settings.GEMINI_MODEL = '{settings.GEMINI_MODEL}'")
    lifespan_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        lifespan_logger.info(f"Force initializing GeminiLLMClient...")
        if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == 'your_gemini_api_key_here' or settings.GOOGLE_API_KEY == 'dummy_api_key_placeholder':
            lifespan_logger.error(
                "CRITICAL CONFIGURATION ERROR: GOOGLE_API_KEY is missing or a placeholder. Gemini WILL fail.")
            raise ConfigurationError("GOOGLE_API_KEY not properly set for hardcoded Gemini provider.")

        current_provider_before_force = settings.LLM_PROVIDER
        settings.LLM_PROVIDER = "gemini"
        lifespan_logger.info(
            f"Settings.LLM_PROVIDER was '{current_provider_before_force}', now forced to '{settings.LLM_PROVIDER}' for client and RAG init.")

        app.state.llm_client = GeminiLLMClient(settings=settings)
        lifespan_logger.info(f"GeminiLLMClient FORCED. Type in app.state: {type(app.state.llm_client)}")

    except Exception as e:
        lifespan_logger.critical(f"Failed to force initialize GeminiLLMClient: {e}", exc_info=True)
        raise ConfigurationError(f"Hardcoded GeminiLLMClient initialization failed: {e}") from e

    try:
        lifespan_logger.info("Initializing RAG components (with forced Gemini client)...")
        doc_loader = DocumentLoader()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE,
                                                       chunk_overlap=settings.CHUNK_OVERLAP)
        embedding_generator = EmbeddingGenerator(llm_client=app.state.llm_client)

        embedding_dimension = settings.EMBEDDING_DIMENSION
        lifespan_logger.info(
            f"RAG: Using EMBEDDING_DIMENSION: {embedding_dimension} (derived from LLM_PROVIDER='{settings.LLM_PROVIDER}')")

        vector_store = FAISSVectorStore(
            embedding_dimension=embedding_dimension,
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
        lifespan_logger.info("RAGPipeline and components initialized with forced Gemini setup.")
    except Exception as e:
        lifespan_logger.critical(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise RAGPipelineError(f"RAG component initialization failed: {e}") from e

    lifespan_logger.info("AvaChat Server Lifespan: Startup sequence complete. Application ready.")
    yield
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence initiated.")
    if hasattr(app.state, 'vector_store') and app.state.vector_store:
        try:
            app.state.vector_store.save()
            lifespan_logger.info("FAISSVectorStore saved.")
        except Exception as e:
            lifespan_logger.error(f"Failed to save FAISSVectorStore: {e}", exc_info=True)
    if hasattr(app.state, 'llm_client') and app.state.llm_client:
        try:
            await app.state.llm_client.close_session()
            lifespan_logger.info("LLM client session closed.")
        except Exception as e:
            lifespan_logger.error(f"Failed to close LLM client session: {e}", exc_info=True)
    lifespan_logger.info("AvaChat Server Lifespan: Shutdown sequence complete.")


app = FastAPI(
    title="AvaChat Backend API",
    description="API server for AvaChat, handling RAG operations and LLM interactions. HARDCODED FOR GEMINI.",
    version="0.1.0",
    lifespan=lifespan
)
app.include_router(api_handlers_router, prefix="/api")
logger.info("FastAPI app created and API router included under /api prefix.")


@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError):
    logging.getLogger("AvaChatServerLifespan").error(f"Unhandled AvaChatError at API level: {exc}", exc_info=True)
    error_code = "AVACHAT_ERROR"
    error_message = str(exc)
    status_code = 500
    if isinstance(exc, ConfigurationError):
        error_code = "CONFIGURATION_ERROR"
        status_code = 503
    elif isinstance(exc, VectorStoreError):
        error_code = "VECTOR_STORE_ERROR"
    elif isinstance(exc, LLMClientError):
        error_code = "LLM_CLIENT_ERROR"
        status_code = 502
    return JSONResponse(status_code=status_code, content={"error": {"code": error_code, "message": error_message}})


@app.get("/", include_in_schema=False)
async def root():
    logging.getLogger(__name__).info("Root path '/' accessed.")
    return {"message": "Welcome to the AvaChat API! This version is HARDCODED FOR GEMINI. Docs at /docs or /redoc."}


if __name__ == "__main__":
    import uvicorn

    print(
        f"Attempting to start Uvicorn server for AvaChat API (Host: {settings.API_SERVER_HOST}, Port: {settings.API_SERVER_PORT})...")
    print(f"   >>> Main Block Check: settings.LLM_PROVIDER = {settings.LLM_PROVIDER}")
    print(
        f"   >>> Main Block Check: settings.GOOGLE_API_KEY set = {bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != 'your_gemini_api_key_here' and settings.GOOGLE_API_KEY != 'dummy_api_key_placeholder')}")
    uvicorn.run(
        "backend.api_server:app",
        host=settings.API_SERVER_HOST,
        port=settings.API_SERVER_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
    print("Uvicorn server has been started (or an attempt was made).")