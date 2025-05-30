import logging
from fastapi import APIRouter, Request, HTTPException

try:
    from config import settings
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse, ApiErrorResponse
    )
    from utils import RAGPipelineError, LLMClientError, ConfigurationError
    # GeminiLLMClient might be typed in app.state.llm_client
    # from llm_interface.gemini_client import GeminiLLMClient
except ImportError as e:
    print(f"HANDLER Import Error: {e}. Using dummy fallbacks.")


    class SettingsClass:
        LLM_PROVIDER = "gemini"; GEMINI_MODEL = "gemini-dummy-handler"  # Hardcode for dummy too


    settings = SettingsClass()


    class PydanticBaseModel:
        def model_dump(self, **kwargs): return {}


    class QueryRequest(PydanticBaseModel):
        query_text: str = ""; top_k_chunks: int = 3; model_name: str = ""


    class QueryResponse(PydanticBaseModel):
        answer: str = "dummy_answer"; retrieved_chunks_details = []


    class IngestDirectoryRequest(PydanticBaseModel):
        directory_path: str = ""


    class IngestDirectoryResponse(PydanticBaseModel):
        status: str = "dummy_status"; documents_processed: int = 0; chunks_created: int = 0


    class AvailableModelsResponse(PydanticBaseModel):
        models: list = [settings.GEMINI_MODEL]


    class ApiErrorResponse(PydanticBaseModel):
        pass


    class RAGPipelineError(Exception):
        pass


    class LLMClientError(Exception):
        pass


    class ConfigurationError(Exception):
        pass

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=QueryResponse)
async def handle_chat_query(request: Request, query_request: QueryRequest):
    # query_request.model_name will be ignored as we are hardcoding Gemini.
    # The RAGPipeline will use the llm_client from app.state (forced Gemini)
    # and the model from settings.GEMINI_MODEL.
    logger.info(f"Received chat query: '{query_request.query_text[:50]}...'. Using HARDCODED GEMINI.")
    try:
        if not hasattr(request.app.state, 'rag_pipeline') or not request.app.state.rag_pipeline:
            logger.error("RAG pipeline not initialized in app state.")
            raise HTTPException(status_code=503, detail="Service not fully initialized (RAG pipeline missing).")

        # Pass None for model_name_override to ensure RAG pipeline uses its configured default (which will be Gemini)
        answer, retrieved_chunks = await request.app.state.rag_pipeline.query_with_rag(
            query_text=query_request.query_text,
            top_k_chunks=query_request.top_k_chunks,
            model_name_override=None  # Explicitly use the pipeline's default (Gemini)
        )
        return QueryResponse(answer=answer, retrieved_chunks_details=retrieved_chunks)
    except RAGPipelineError as e:
        logger.error(f"RAG pipeline error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in RAG pipeline: {str(e)}")
    except LLMClientError as e:
        logger.error(f"LLM client error during query processing: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM communication error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@router.post("/ingest", response_model=IngestDirectoryResponse)
async def handle_ingest_directory(request: Request, ingest_request: IngestDirectoryRequest):
    logger.info(f"Received request to ingest documents from: {ingest_request.directory_path}")
    try:
        # RAG pipeline will use the forced Gemini client for embeddings
        docs_processed, chunks_created = await request.app.state.rag_pipeline.ingest_documents_from_directory(
            ingest_request.directory_path
        )
        return IngestDirectoryResponse(
            status="Ingestion process completed.",
            documents_processed=docs_processed,
            chunks_created=chunks_created
        )
    except RAGPipelineError as e:
        logger.error(f"RAG pipeline error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during document ingestion: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during document ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error during ingestion: {str(e)}")


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models_endpoint(request: Request):
    logger.info(f"Request received for available models. HARDCODING TO GEMINI.")
    try:
        if not hasattr(request.app.state, 'llm_client') or not request.app.state.llm_client:
            logger.error("LLM client (expected Gemini) not initialized in app state for listing models.")
            raise HTTPException(status_code=503, detail="Service not fully initialized (LLM client missing).")

        # app.state.llm_client is now guaranteed to be GeminiLLMClient by api_server.py
        # Its list_available_models() method returns the configured Gemini model.
        client_type = type(request.app.state.llm_client).__name__
        logger.info(f"   Handler /models: app.state.llm_client type = {client_type}")

        models = await request.app.state.llm_client.list_available_models()

        logger.info(f"Returning HARDCODED Gemini model(s) via llm_client ({client_type}): {models}")
        return AvailableModelsResponse(models=models)

    except LLMClientError as e:  # Should not happen if Gemini client init was ok
        logger.error(f"LLMClientError when fetching hardcoded Gemini models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error with Gemini client for models: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching hardcoded Gemini models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch hardcoded Gemini models: {str(e)}")