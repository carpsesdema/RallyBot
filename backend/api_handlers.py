# backend/api_handlers.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import List, Optional
import httpx
import asyncio

# Attempt to import project-specific modules
try:
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse,
        ApiErrorResponse, ApiErrorDetail
    )
    # RAGPipeline and LLMClient types for dependency injection hints
    from rag.rag_pipeline import RAGPipeline
    from llm_interface.ollama_client import OllamaLLMClient
    from llm_interface.gemini_client import GeminiLLMClient
    from utils import RAGPipelineError, DocumentLoadingError, LLMClientError, ConfigurationError
    from config import settings
except ImportError as e:
    print(f"Import Error in backend/api_handlers.py: {e}. Ensure models.py, rag_pipeline.py, etc., are correct.")


    # Define dummy classes if imports fail for standalone parsing/early dev
    class PydanticBaseModel:
        def model_dump(self, **kwargs): return {}

        @classmethod
        def model_validate(cls, data, **kwargs): return cls()


    class QueryRequest(PydanticBaseModel):
        query_text: str = ""
        top_k_chunks: int = 3
        model_name: Optional[str] = None


    class QueryResponse(PydanticBaseModel):
        answer: str = ""
        retrieved_chunks_details: list = []
        used_web_search: bool = False


    class IngestDirectoryRequest(PydanticBaseModel):
        directory_path: str = ""


    class IngestDirectoryResponse(PydanticBaseModel):
        status: str = ""; documents_processed: int = 0; chunks_created: int = 0


    class AvailableModelsResponse(PydanticBaseModel):
        models: List[str] = []


    class ApiErrorDetail(PydanticBaseModel):
        code: str = ""; message: str = ""


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class RAGPipeline:
        pass


    class OllamaLLMClient:
        async def list_available_models(self) -> List[str]: return ["dummy_model:latest"]


    class GeminiLLMClient:
        async def list_available_models(self) -> List[str]: return ["gemini-1.5-flash"]


    class RAGPipelineError(Exception):
        pass


    class DocumentLoadingError(Exception):
        pass


    class LLMClientError(Exception):
        pass


    class ConfigurationError(Exception):
        pass


    # Dummy settings
    class Settings:
        LLM_PROVIDER = "ollama"


    settings = Settings()

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Web Search Functionality ---

async def search_web_for_tennis_info(query: str, max_results: int = 3) -> List[dict]:
    """
    Performs web search for tennis-related information using a simple search API.
    Returns a list of search results with title, snippet, and URL.
    """
    search_results = []

    try:
        # Using DuckDuckGo Instant Answer API as a fallback search
        # You could also integrate with Google Custom Search, Bing, or other APIs
        search_query = f"tennis {query}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try DuckDuckGo search
            try:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": search_query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract instant answer if available
                    if data.get("AbstractText"):
                        search_results.append({
                            "title": data.get("AbstractSource", "Tennis Information"),
                            "snippet": data["AbstractText"][:300] + "..." if len(data["AbstractText"]) > 300 else data[
                                "AbstractText"],
                            "url": data.get("AbstractURL", ""),
                            "source": "DuckDuckGo"
                        })

                    # Extract related topics
                    for topic in data.get("RelatedTopics", [])[:2]:  # Limit to 2 topics
                        if isinstance(topic, dict) and topic.get("Text"):
                            search_results.append({
                                "title": f"Related: {topic.get('FirstURL', '').split('/')[-1].replace('_', ' ').title()}",
                                "snippet": topic["Text"][:300] + "..." if len(topic["Text"]) > 300 else topic["Text"],
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo"
                            })

            except Exception as ddg_error:
                logger.warning(f"DuckDuckGo search failed: {ddg_error}")

        # If we don't have enough results, add some fallback tennis information
        if len(search_results) == 0:
            search_results = [
                {
                    "title": "Tennis Information",
                    "snippet": f"Tennis information related to: {query}. This is a fallback response when web search is unavailable.",
                    "url": "",
                    "source": "Fallback"
                }
            ]

        logger.info(f"Web search returned {len(search_results)} results for query: {query}")
        return search_results[:max_results]

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{
            "title": "Search Unavailable",
            "snippet": "Web search is currently unavailable. Please try again later or rephrase your question.",
            "url": "",
            "source": "Error"
        }]


async def should_use_web_search(query: str, rag_sources: List[dict]) -> bool:
    """
    Determines if web search should be used as a fallback.
    Returns True if RAG results are insufficient.
    """
    # Use web search if:
    # 1. No sources found from RAG
    # 2. Very few sources (less than 2)
    # 3. Query seems to be asking for recent/current information

    if not rag_sources or len(rag_sources) < 2:
        return True

    # Keywords that might indicate need for current information
    current_info_keywords = [
        "current", "latest", "recent", "now", "today", "2024", "2025",
        "ranking", "standings", "schedule", "upcoming", "news"
    ]

    query_lower = query.lower()
    if any(keyword in query_lower for keyword in current_info_keywords):
        return True

    return False


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


def get_llm_client(request: Request):
    """Dependency to get the LLMClient instance from app state."""
    if not hasattr(request.app.state, 'llm_client') or request.app.state.llm_client is None:
        logger.error("LLMClient not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM client service is not initialized or available."
        )

    # Ensure the client has the 'list_available_models' method
    if not hasattr(request.app.state.llm_client, 'list_available_models'):
        logger.error("LLMClient in application state does not support listing models.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
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
    description="Fetches a list of available LLM model names from the configured provider (e.g., Ollama, Gemini)."
)
async def list_available_llm_models(
        llm_client=Depends(get_llm_client)
):
    logger.info("Request received for /models endpoint.")
    try:
        model_names = await llm_client.list_available_models()
        logger.info(f"Successfully fetched {len(model_names)} models.")
        return AvailableModelsResponse(models=model_names)
    except LLMClientError as e:
        logger.error(f"LLMClientError when fetching models: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="LLM_SERVICE_ERROR", message=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_detail.model_dump()
        )
    except Exception as e:
        logger.critical(f"Unexpected error in /models endpoint: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR",
                                      message="An unexpected error occurred while fetching models.")
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
    summary="Process a chat query using RAG with web search fallback",
    description="Receives a user query, retrieves relevant context using RAG, and generates a response from the LLM. Falls back to web search if RAG results are insufficient."
)
async def handle_chat_query(
        payload: QueryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    logger.info(
        f"Received chat query: '{payload.query_text[:50]}...', top_k: {payload.top_k_chunks}, model: {payload.model_name}")

    used_web_search = False
    web_search_results = []

    try:
        # First, try RAG pipeline
        answer, sources = await rag_pipeline.query_with_rag(
            query_text=payload.query_text,
            top_k_chunks=payload.top_k_chunks,
            model_name_override=payload.model_name
        )

        # Check if we should use web search as fallback
        if await should_use_web_search(payload.query_text, sources):
            logger.info("RAG results insufficient, attempting web search fallback...")
            used_web_search = True

            try:
                # Perform web search
                web_search_results = await search_web_for_tennis_info(payload.query_text)

                # Enhance the answer with web search results
                if web_search_results:
                    web_context = "\n\nAdditional information from web search:\n"
                    for i, result in enumerate(web_search_results, 1):
                        web_context += f"{i}. {result['title']}: {result['snippet']}\n"

                    # Re-generate answer with enhanced context
                    enhanced_prompt = f"""Based on the following information, please provide a comprehensive answer about tennis.

Knowledge Base Information:
{chr(10).join([chunk.get('text_preview', '') for chunk in sources]) if sources else 'No specific information found in knowledge base.'}

Web Search Results:
{web_context}

Question: {payload.query_text}

Please provide a complete and accurate answer, citing both knowledge base and web sources where appropriate."""

                    # Get LLM client from the RAG pipeline
                    llm_client = rag_pipeline.llm_client

                    # Determine model name
                    final_model_name = payload.model_name
                    if not final_model_name:
                        if settings.LLM_PROVIDER == "gemini":
                            final_model_name = settings.GEMINI_MODEL
                        else:
                            final_model_name = settings.OLLAMA_CHAT_MODEL

                    enhanced_answer = await llm_client.generate_response(
                        prompt=enhanced_prompt,
                        model_name=final_model_name
                    )

                    # Combine sources
                    combined_sources = sources + [
                        {
                            "source_file": f"Web Search: {result['source']}",
                            "chunk_id": f"web_{i}",
                            "text_preview": f"{result['title']}: {result['snippet'][:100]}..."
                        }
                        for i, result in enumerate(web_search_results)
                    ]

                    logger.info(f"Enhanced answer generated with web search. Sources: {len(combined_sources)}")
                    return QueryResponse(
                        answer=enhanced_answer,
                        retrieved_chunks_details=combined_sources,
                        used_web_search=True
                    )

            except Exception as web_error:
                logger.warning(f"Web search failed, using original RAG answer: {web_error}")
                # Fall back to original RAG answer if web search fails

        logger.info(
            f"Successfully generated chat response. Answer length: {len(answer)}, Sources found: {len(sources)}, Used web search: {used_web_search}")
        return QueryResponse(
            answer=answer,
            retrieved_chunks_details=sources,
            used_web_search=used_web_search
        )

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
            directory_path_str=payload.directory_path
        )
        logger.info(f"Successfully ingested documents. Processed: {num_docs} docs, Created: {num_chunks} chunks.")
        return IngestDirectoryResponse(status="success", documents_processed=num_docs, chunks_created=num_chunks)
    except DocumentLoadingError as e:
        logger.error(f"DocumentLoadingError during ingestion for path '{payload.directory_path}': {e}", exc_info=True)
        error_code = "DOCUMENT_PATH_INVALID" if "not found" in str(e).lower() or "is not a directory" in str(
            e).lower() else "DOCUMENT_LOADING_FAILED"
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
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR",
                                      message="An unexpected error occurred during document ingestion.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())


# --- Health Check Endpoint ---

@router.get(
    "/health",
    summary="Health check",
    description="Returns the health status of the API and its dependencies."
)
async def health_check(request: Request):
    """Basic health check endpoint."""
    try:
        # Check if RAG pipeline is available
        rag_status = "ok" if hasattr(request.app.state,
                                     'rag_pipeline') and request.app.state.rag_pipeline else "unavailable"

        # Check if LLM client is available
        llm_status = "ok" if hasattr(request.app.state,
                                     'llm_client') and request.app.state.llm_client else "unavailable"

        return {
            "status": "healthy",
            "timestamp": "2025-01-31T00:00:00Z",  # You'd use actual timestamp
            "services": {
                "rag_pipeline": rag_status,
                "llm_client": llm_status,
                "web_search": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


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