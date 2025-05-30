# backend/api_handlers.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import List # Added for type hinting

# Attempt to import project-specific modules
try:
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse, # Added new response model
        ApiErrorResponse, ApiErrorDetail
    )
    # RAGPipeline and OllamaLLMClient types for dependency injection hints
    from rag.rag_pipeline import RAGPipeline
    from llm_interface.ollama_client import OllamaLLMClient # Ensure this is the correct client
    from utils import RAGPipelineError, DocumentLoadingError, LLMClientError, ConfigurationError
except ImportError as e:
    print(f"Import Error in backend/api_handlers.py: {e}. Ensure models.py, rag_pipeline.py, etc., are correct.")

    # Define dummy classes if imports fail for standalone parsing/early dev
    class PydanticBaseModel:
        def model_dump(self, **kwargs): return {}
        @classmethod
        def model_validate(cls, data, **kwargs): return cls()

    class QueryRequest(PydanticBaseModel): query_text: str = ""; top_k_chunks: int = 3; model_name: str = "default"
    class QueryResponse(PydanticBaseModel): answer: str = ""; retrieved_chunks_details: list = []
    class IngestDirectoryRequest(PydanticBaseModel): directory_path: str = ""
    class IngestDirectoryResponse(PydanticBaseModel): status: str = ""; documents_processed: int = 0; chunks_created: int = 0
    class AvailableModelsResponse(PydanticBaseModel): models: List[str] = [] # Added dummy
    class ApiErrorDetail(PydanticBaseModel): code: str = ""; message: str = ""
    class ApiErrorResponse(PydanticBaseModel): error: ApiErrorDetail = ApiErrorDetail()
    class RAGPipeline: pass
    class OllamaLLMClient:
        async def list_available_models(self) -> List[str]: return ["dummy_model:latest"] # Dummy method
    class RAGPipelineError(Exception): pass
    class DocumentLoadingError(Exception): pass
    class LLMClientError(Exception): pass
    class ConfigurationError(Exception): pass

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Functions ---

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Dependency to get the RAGPipeline instance from app state."""
    if not hasattr(request.app.state, 'rag_pipeline') or request.app.state.rag_pipeline is None:
        logger.error("RAGPipeline not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized or available."
        )
    return request.app.state.rag_pipeline

def get_llm_client(request: Request) -> OllamaLLMClient: # Assuming OllamaLLMClient for now
    """Dependency to get the LLMClient instance from app state."""
    # This needs to be robust if you support multiple LLM clients (Ollama, Gemini)
    # For now, let's assume app.state.llm_client is the primary one (e.g. Ollama)
    # or the one capable of listing its own models.
    # If you want to list models from a specific provider, you might need separate dependencies
    # or a way to specify the provider.
    if not hasattr(request.app.state, 'llm_client') or request.app.state.llm_client is None:
        logger.error("LLMClient not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM client service is not initialized or available."
        )
    # Ensure the client has the 'list_available_models' method if it's specific
    if not hasattr(request.app.state.llm_client, 'list_available_models'):
        logger.error("LLMClient in application state does not support listing models.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, # Or 503 if it should be there
            detail="The configured LLM client cannot list available models."
        )
    return request.app.state.llm_client

# --- API Endpoints ---

@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse}
    },
    summary="List available LLM models",
    description="Fetches a list of available LLM model names from the configured provider (e.g., Ollama)."
)
async def list_available_llm_models(
    llm_client: OllamaLLMClient = Depends(get_llm_client) # Specify OllamaLLMClient for clarity if that's what get_llm_client returns
):
    logger.info("Request received for /models endpoint.")
    try:
        # The llm_client (OllamaLLMClient) should have the list_available_models method
        model_names = await llm_client.list_available_models()
        logger.info(f"Successfully fetched {len(model_names)} models.")
        return AvailableModelsResponse(models=model_names)
    except LLMClientError as e:
        logger.error(f"LLMClientError when fetching models: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="LLM_SERVICE_ERROR", message=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, # LLM service is often external
            detail=error_detail.model_dump()
        )
    except Exception as e:
        logger.critical(f"Unexpected error in /models endpoint: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR", message="An unexpected error occurred while fetching models.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail.model_dump()
        )

@router.post(
    "/chat",
    response_model=QueryResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse}
    },
    summary="Process a chat query using RAG",
    description="Receives a user query, retrieves relevant context using RAG, and generates a response from the LLM."
)
async def handle_chat_query(
        payload: QueryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    logger.info(f"Received chat query: '{payload.query_text[:50]}...', top_k: {payload.top_k_chunks}, model: {payload.model_name}")
    try:
        answer, sources = await rag_pipeline.query_with_rag(
            query_text=payload.query_text,
            top_k_chunks=payload.top_k_chunks,
            model_name_override=payload.model_name # Pass the model name to the RAG pipeline
        )
        logger.info(f"Successfully generated chat response. Answer length: {len(answer)}, Sources found: {len(sources)}")
        return QueryResponse(answer=answer, retrieved_chunks_details=sources)
    except RAGPipelineError as e:
        logger.error(f"RAGPipelineError during chat query: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="RAG_PROCESSING_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())
    except LLMClientError as e:
        logger.error(f"LLMClientError during chat query: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="LLM_CLIENT_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=error_detail.model_dump())
    except Exception as e:
        logger.critical(f"Unexpected error in /chat endpoint: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR", message="An unexpected error occurred.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())

@router.post(
    "/ingest",
    response_model=IngestDirectoryResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse}
    },
    summary="Ingest documents from a directory",
    description="Triggers the RAG pipeline to load, process, and store documents from the specified server-side directory path."
)
async def handle_ingest_directory(
        payload: IngestDirectoryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    logger.info(f"Received request to ingest documents from directory: {payload.directory_path}")
    try:
        num_docs, num_chunks = await rag_pipeline.ingest_documents_from_directory(
            directory_path_str=payload.directory_path # Ensure RAG pipeline expects directory_path_str
        )
        logger.info(f"Successfully ingested documents. Processed: {num_docs} docs, Created: {num_chunks} chunks.")
        return IngestDirectoryResponse(status="success", documents_processed=num_docs, chunks_created=num_chunks)
    except DocumentLoadingError as e:
        logger.error(f"DocumentLoadingError during ingestion for path '{payload.directory_path}': {e}", exc_info=True)
        error_code = "DOCUMENT_PATH_INVALID" if "not found" in str(e).lower() or "is not a directory" in str(e).lower() else "DOCUMENT_LOADING_FAILED"
        error_detail = ApiErrorDetail(code=error_code, message=str(e))
        http_status = status.HTTP_404_NOT_FOUND if error_code == "DOCUMENT_PATH_INVALID" else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=http_status, detail=error_detail.model_dump())
    except RAGPipelineError as e:
        logger.error(f"RAGPipelineError during ingestion for path '{payload.directory_path}': {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="INGESTION_PROCESSING_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())
    except ConfigurationError as e:
        logger.error(f"ConfigurationError affecting ingestion: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="SERVER_CONFIGURATION_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_detail.model_dump())
    except Exception as e:
        logger.critical(f"Unexpected error in /ingest endpoint for path '{payload.directory_path}': {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR", message="An unexpected error occurred during document ingestion.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())

if __name__ == "__main__":
    logger.info("backend/api_handlers.py executed directly (for info only).")
    # Example of how an error response would look
    try:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiErrorDetail(code="VALIDATION_ERROR", message="Invalid input provided.").model_dump()
        )
    except HTTPException as h:
        print("\nExample HTTPException structure:")
        print(f"Status Code: {h.status_code}")
        print(f"Detail: {h.detail}")