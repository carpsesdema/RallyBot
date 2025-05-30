# backend/api_handlers.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

# Attempt to import project-specific modules
# These will be fully available when the project is complete.
try:
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        ApiErrorResponse, ApiErrorDetail
    )
    # RAGPipeline and OllamaLLMClient types for dependency injection hints
    from rag.rag_pipeline import RAGPipeline
    from llm_interface.ollama_client import OllamaLLMClient
    from utils import RAGPipelineError, DocumentLoadingError, LLMClientError, ConfigurationError
except ImportError as e:
    print(f"Import Error in backend/api_handlers.py: {e}. Ensure models.py, rag_pipeline.py, etc., are correct.")


    # Define dummy classes if imports fail for standalone parsing/early dev
    class PydanticBaseModel:  # Dummy base for other Pydantic models
        def model_dump(self, **kwargs): return {}

        @classmethod
        def model_validate(cls, data, **kwargs): return cls()


    class QueryRequest(PydanticBaseModel):
        query_text: str = ""; top_k_chunks: int = 3


    class QueryResponse(PydanticBaseModel):
        answer: str = ""; retrieved_chunks_details: list = []


    class IngestDirectoryRequest(PydanticBaseModel):
        directory_path: str = ""


    class IngestDirectoryResponse(PydanticBaseModel):
        status: str = ""; documents_processed: int = 0; chunks_created: int = 0


    class ApiErrorDetail(PydanticBaseModel):
        code: str = ""; message: str = ""


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class RAGPipeline:
        pass  # Dummy for type hint


    class OllamaLLMClient:
        pass  # Dummy for type hint


    class RAGPipelineError(Exception):
        pass


    class DocumentLoadingError(Exception):
        pass


    class LLMClientError(Exception):
        pass


    class ConfigurationError(Exception):
        pass

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Dependency Injection Functions ---

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Dependency to get the RAGPipeline instance from app state."""
    if not hasattr(request.app.state, 'rag_pipeline') or request.app.state.rag_pipeline is None:
        logger.error("RAGPipeline not found in application state. FastAPI server setup might be incomplete.")
        # This error should ideally not happen if lifespan context manager in api_server.py works.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized or available."
        )
    return request.app.state.rag_pipeline


def get_llm_client(request: Request) -> OllamaLLMClient:
    """Dependency to get the OllamaLLMClient instance from app state."""
    if not hasattr(request.app.state, 'llm_client') or request.app.state.llm_client is None:
        logger.error("OllamaLLMClient not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM client service is not initialized or available."
        )
    return request.app.state.llm_client


# --- API Endpoints ---

@router.post(
    "/chat",
    response_model=QueryResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse}  # For input validation errors by Pydantic
    },
    summary="Process a chat query using RAG",
    description="Receives a user query, retrieves relevant context using RAG, and generates a response from the LLM."
)
async def handle_chat_query(
        payload: QueryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Handles chat queries. Uses the RAG pipeline to generate a response.
    """
    logger.info(f"Received chat query: '{payload.query_text[:50]}...', top_k: {payload.top_k_chunks}")
    try:
        answer, sources = await rag_pipeline.query_with_rag(
            query_text=payload.query_text,
            top_k_chunks=payload.top_k_chunks
        )
        logger.info(
            f"Successfully generated chat response. Answer length: {len(answer)}, Sources found: {len(sources)}")
        return QueryResponse(answer=answer, retrieved_chunks_details=sources)

    except RAGPipelineError as e:
        logger.error(f"RAGPipelineError during chat query: {e}", exc_info=True)
        # More specific error codes could be set based on the type of RAGPipelineError
        error_detail = ApiErrorDetail(code="RAG_PROCESSING_ERROR", message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail.model_dump()  # FastAPI expects dict here for detail if using HTTPException this way
        )
    except LLMClientError as e:  # If RAGPipeline re-raises this or it's caught separately
        logger.error(f"LLMClientError during chat query: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="LLM_CLIENT_ERROR", message=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,  # LLM service is likely external or acting as a gateway
            detail=error_detail.model_dump()
        )
    except Exception as e:  # Catch-all for unexpected errors
        logger.critical(f"Unexpected error in /chat endpoint: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR",
                                      message="An unexpected error occurred processing your chat query.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail.model_dump()
        )


@router.post(
    "/ingest",
    response_model=IngestDirectoryResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse},  # e.g., if directory_path is invalid before RAG logic
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse}  # if directory path doesn't exist
    },
    summary="Ingest documents from a directory",
    description="Triggers the RAG pipeline to load, process, and store documents from the specified server-side directory path."
)
async def handle_ingest_directory(
        payload: IngestDirectoryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Handles requests to ingest documents from a directory.
    """
    logger.info(f"Received request to ingest documents from directory: {payload.directory_path}")
    try:
        # Note: Path validation (existence, readability) should ideally happen within
        # the DocumentLoader or RAGPipeline for better encapsulation.
        # If DocumentLoadingError is raised for path issues, it will be caught below.

        num_docs, num_chunks = await rag_pipeline.ingest_documents_from_directory(
            directory_path=payload.directory_path
        )
        logger.info(f"Successfully ingested documents. Processed: {num_docs} docs, Created: {num_chunks} chunks.")
        return IngestDirectoryResponse(
            status="success",
            documents_processed=num_docs,
            chunks_created=num_chunks
        )

    except DocumentLoadingError as e:  # Specific error for path issues or file reading problems
        logger.error(f"DocumentLoadingError during ingestion for path '{payload.directory_path}': {e}", exc_info=True)
        error_code = "DOCUMENT_PATH_INVALID" if "not found" in str(e).lower() or "is not a directory" in str(
            e).lower() else "DOCUMENT_LOADING_FAILED"
        error_detail = ApiErrorDetail(code=error_code, message=str(e))
        # Determine appropriate status code
        http_status = status.HTTP_404_NOT_FOUND if error_code == "DOCUMENT_PATH_INVALID" else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=http_status, detail=error_detail.model_dump())

    except RAGPipelineError as e:
        logger.error(f"RAGPipelineError during ingestion for path '{payload.directory_path}': {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="INGESTION_PROCESSING_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())

    except ConfigurationError as e:  # E.g. if a required path in settings is bad during pipeline setup
        logger.error(f"ConfigurationError affecting ingestion: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="SERVER_CONFIGURATION_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_detail.model_dump())

    except Exception as e:  # Catch-all for unexpected errors
        logger.critical(f"Unexpected error in /ingest endpoint for path '{payload.directory_path}': {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR",
                                      message="An unexpected error occurred during document ingestion.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())


if __name__ == "__main__":
    # This block is for illustration and won't run as part of the FastAPI app.
    # To test these endpoints, you'd run the FastAPI server (api_server.py) and use an HTTP client (like Postman, curl, or httpx).
    logger.info("backend/api_handlers.py executed directly (for info only).")
    logger.info("This module defines FastAPI routes and is intended to be included by api_server.py.")

    # Example of how an error response would look (for documentation or understanding)
    try:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiErrorDetail(code="VALIDATION_ERROR", message="Invalid input provided.").model_dump()
        )
    except HTTPException as h:
        print("\nExample HTTPException structure:")
        print(f"Status Code: {h.status_code}")
        print(f"Detail: {h.detail}")
        # In FastAPI, this would be automatically converted to a JSON response:
        # {"detail": {"code": "VALIDATION_ERROR", "message": "Invalid input provided."}}
        # Or, if detail is a string: {"detail": "Error message"}

        # To match the ApiErrorResponse structure for clients expecting it:
        # One way is to structure the `detail` of HTTPException to match the *value* of the `error` field in ApiErrorResponse.
        # So, `detail` would be `ApiErrorDetail(...).model_dump()`.
        # The client would then parse `response.json()['detail']` into an `ApiErrorDetail`.

        # If the client strictly expects `{"error": {"code": "...", "message": "..."}}`,
        # then you might need custom exception handlers in FastAPI (see FastAPI docs on "Handling Errors")
        # to transform HTTPErrors into this specific JSON structure.
        # For now, the `detail` field of HTTPException will carry the `ApiErrorDetail` contents.