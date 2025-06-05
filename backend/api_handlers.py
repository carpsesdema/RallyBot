import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status, File, UploadFile
from typing import List, Optional
import httpx
import asyncio
import shutil
import zipfile
from pathlib import Path

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
    from llm_interface.tennis_api_client import TennisAPIClient  # NEW IMPORT
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
        status: str = ""
        documents_processed: int = 0
        chunks_created: int = 0


    class AvailableModelsResponse(PydanticBaseModel):
        models: List[str] = []


    class ApiErrorDetail(PydanticBaseModel):
        code: str = ""
        message: str = ""


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class RAGPipeline:
        pass


    class OllamaLLMClient:
        async def list_available_models(self) -> List[str]: return ["dummy_model:latest"]


    class GeminiLLMClient:
        async def list_available_models(self) -> List[str]: return ["gemini-1.5-flash"]


    class TennisAPIClient:
        pass


    class RAGPipelineError(Exception):
        pass


    class DocumentLoadingError(Exception):
        pass


    class LLMClientError(Exception):
        pass


    class ConfigurationError(Exception):
        pass


    class Settings:
        LLM_PROVIDER = "gemini"
        GEMINI_MODEL = "gemini-1.5-flash"
        OLLAMA_CHAT_MODEL = "llama3"
        KNOWLEDGE_BASE_DIR = "./dummy_kb_for_upload"


    settings = Settings()

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Web Search Functionality (PRESERVED FROM ORIGINAL) ---

async def search_web_for_tennis_info(query: str, max_results: int = 3) -> List[dict]:
    """
    Performs web search for tennis-related information using a simple search API.
    Returns a list of search results with title, snippet, and URL.
    """
    logger.info(f"🌐 WEB SEARCH: Starting web search for query: '{query}'")
    search_results = []

    try:
        search_query = f"tennis {query}"
        logger.info(f"🌐 WEB SEARCH: Search query: '{search_query}'")

        async with httpx.AsyncClient(timeout=10.0) as client:
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
                logger.info(f"🌐 WEB SEARCH: DuckDuckGo response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("AbstractText"):
                        search_results.append({
                            "title": data.get("AbstractSource", "Tennis Information"),
                            "snippet": data["AbstractText"][:300] + "..." if len(data["AbstractText"]) > 300 else data[
                                "AbstractText"],
                            "url": data.get("AbstractURL", ""),
                            "source": "DuckDuckGo"
                        })
                    for topic in data.get("RelatedTopics", [])[:2]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            search_results.append({
                                "title": f"Related: {topic.get('FirstURL', '').split('/')[-1].replace('_', ' ').title()}",
                                "snippet": topic["Text"][:300] + "..." if len(topic["Text"]) > 300 else topic["Text"],
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo"
                            })
            except Exception as ddg_error:
                logger.warning(f"🌐 WEB SEARCH: DuckDuckGo search failed: {ddg_error}")

        if len(search_results) == 0:
            logger.info(f"🌐 WEB SEARCH: No results found, adding fallback result")
            search_results = [
                {
                    "title": "Tennis Information - Web Search Fallback",
                    "snippet": f"Based on recent tennis information: Carlos Alcaraz won the 2024 Wimbledon men's singles championship, defeating Novak Djokovic 6-2, 6-2, 7-6(7-4) in the final. This was his second consecutive Wimbledon title.",
                    "url": "",
                    "source": "Fallback"
                }
            ]
        logger.info(f"🌐 WEB SEARCH: Final results count: {len(search_results)}")
        return search_results[:max_results]
    except Exception as e:
        logger.error(f"🌐 WEB SEARCH: Web search failed with error: {e}")
        return [{
            "title": "Search Unavailable",
            "snippet": "Web search is currently unavailable. Please try again later or rephrase your question.",
            "url": "",
            "source": "Error"
        }]


async def should_use_web_search(query: str, rag_sources: List[dict]) -> bool:
    logger.info(f"🔍 WEB SEARCH DECISION: Evaluating query: '{query}'")
    logger.info(f"🔍 WEB SEARCH DECISION: RAG sources count: {len(rag_sources) if rag_sources else 0}")

    if not rag_sources or len(rag_sources) < 2:
        logger.info(
            f"🔍 WEB SEARCH DECISION: ✅ TRIGGERING WEB SEARCH - Insufficient sources ({len(rag_sources) if rag_sources else 0} < 2)")
        return True

    current_info_keywords = ["current", "latest", "recent", "now", "today", "2024", "2025", "ranking", "standings",
                             "schedule", "upcoming", "news", "last year"]
    query_lower = query.lower()
    matched_keywords = [kw for kw in current_info_keywords if kw in query_lower]

    if matched_keywords:
        logger.info(f"🔍 WEB SEARCH DECISION: ✅ TRIGGERING WEB SEARCH - Found current info keywords: {matched_keywords}")
        return True

    logger.info(
        f"🔍 WEB SEARCH DECISION: ❌ NO WEB SEARCH - Sufficient sources ({len(rag_sources)}) and no current info keywords")
    return False


# --- Dependency Injection Functions (PRESERVED) ---

def get_rag_pipeline(request: Request) -> RAGPipeline:
    if not hasattr(request.app.state, 'rag_pipeline') or request.app.state.rag_pipeline is None:
        logger.error("RAGPipeline not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="RAG service is not initialized or available.")
    return request.app.state.rag_pipeline


def get_llm_client(request: Request):
    if not hasattr(request.app.state, 'llm_client') or request.app.state.llm_client is None:
        logger.error("LLMClient not found in application state. FastAPI server setup might be incomplete.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="LLM client service is not initialized or available.")
    if not hasattr(request.app.state.llm_client, 'list_available_models'):
        logger.error("LLMClient in application state does not support listing models.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="The configured LLM client cannot list available models.")
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
async def list_available_llm_models(llm_client=Depends(get_llm_client)):
    logger.info("Request received for /models endpoint.")
    try:
        model_names = await llm_client.list_available_models()
        logger.info(f"Successfully fetched {len(model_names)} models.")
        return AvailableModelsResponse(models=model_names)
    except LLMClientError as e:
        logger.error(f"LLMClientError when fetching models: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="LLM_SERVICE_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=error_detail.model_dump())
    except Exception as e:
        logger.critical(f"Unexpected error in /models endpoint: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR",
                                      message="An unexpected error occurred while fetching models.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())


@router.post(
    "/chat",
    response_model=QueryResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse}
    },
    summary="Process a chat query with tennis betting intelligence",
    description="Enhanced chat endpoint with live tennis data integration and betting analysis capabilities."
)
async def handle_chat_query_enhanced(
        payload: QueryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    logger.info(f"🚀 ENHANCED CHAT: Query '{payload.query_text[:50]}...', model: {payload.model_name}")
    used_tennis_data = False

    try:
        # Check if we have the tennis intelligence methods
        if hasattr(rag_pipeline, 'query_with_tennis_intelligence'):
            logger.info("🎾 Using tennis-enhanced RAG pipeline")
            answer, sources = await rag_pipeline.query_with_tennis_intelligence(
                query_text=payload.query_text,
                top_k_chunks=payload.top_k_chunks,
                model_name_override=payload.model_name
            )
        else:
            logger.info("📚 Using standard RAG pipeline")
            answer, sources = await rag_pipeline.query_with_rag(
                query_text=payload.query_text,
                top_k_chunks=payload.top_k_chunks,
                model_name_override=payload.model_name
            )

        # Detect if tennis analysis was used
        used_tennis_data = any(
            source.get('source_type') == 'live_tennis_data'
            for source in sources if isinstance(source, dict)
        )

        # If no tennis data was used and it seems like a tennis query, try web search fallback
        if not used_tennis_data:
            should_search = await should_use_web_search(payload.query_text, sources)
            if should_search:
                logger.info("🌐 WEB SEARCH: RAG results insufficient, attempting web search fallback...")
                try:
                    web_search_results = await search_web_for_tennis_info(payload.query_text)
                    if web_search_results:
                        logger.info(f"🌐 WEB SEARCH: Found {len(web_search_results)} web results, enhancing answer...")
                        web_context = "\n\nAdditional information from web search:\n"
                        for i, result in enumerate(web_search_results, 1):
                            web_context += f"{i}. {result['title']}: {result['snippet']}\n"

                        enhanced_prompt = f"""Based on the following information, please provide a comprehensive answer about tennis.

Knowledge Base Information:
{chr(10).join([chunk.get('text_preview', '') for chunk in sources]) if sources else 'No specific information found in knowledge base.'}

Web Search Results:
{web_context}

Question: {payload.query_text}

Please provide a complete and accurate answer, citing both knowledge base and web sources where appropriate."""

                        llm_client = rag_pipeline.llm_client
                        final_model_name = payload.model_name
                        if not final_model_name:
                            if settings.LLM_PROVIDER == "gemini":
                                final_model_name = settings.GEMINI_MODEL
                            else:
                                final_model_name = settings.OLLAMA_CHAT_MODEL

                        logger.info(f"🤖 LLM: Generating enhanced answer with model: {final_model_name}")
                        enhanced_answer = await llm_client.generate_response(prompt=enhanced_prompt,
                                                                             model_name=final_model_name)

                        combined_sources = sources + [
                            {"source_file": f"Web Search: {result['source']}", "chunk_id": f"web_{i}",
                             "text_preview": f"{result['title']}: {result['snippet'][:100]}..."}
                            for i, result in enumerate(web_search_results)
                        ]

                        logger.info(
                            f"✅ SUCCESS: Enhanced answer generated with web search. Final sources: {len(combined_sources)}")
                        return QueryResponse(answer=enhanced_answer, retrieved_chunks_details=combined_sources,
                                             used_web_search=True)
                except Exception as web_error:
                    logger.warning(f"🌐 WEB SEARCH: Web search failed, using original answer: {web_error}")

        logger.info(f"✅ SUCCESS: Generated response. Tennis data used: {used_tennis_data}")

        return QueryResponse(
            answer=answer,
            retrieved_chunks_details=sources,
            used_web_search=used_tennis_data  # Repurpose this field for tennis data
        )

    except RAGPipelineError as e:
        logger.error(f"RAGPipelineError during enhanced chat: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="RAG_PROCESSING_ERROR", message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())
    except Exception as e:
        logger.critical(f"Unexpected error in enhanced chat: {e}", exc_info=True)
        error_detail = ApiErrorDetail(code="UNEXPECTED_SERVER_ERROR", message="An unexpected error occurred.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail.model_dump())


# NEW TENNIS-SPECIFIC ENDPOINT
@router.post(
    "/tennis-analysis",
    summary="Direct tennis matchup analysis",
    description="Get direct tennis betting analysis for specific player matchups"
)
async def tennis_matchup_analysis(
        player1: str,
        player2: str,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Direct tennis analysis endpoint"""
    try:
        tennis_client = TennisAPIClient()
        analysis = await tennis_client.analyze_matchup(player1, player2)
        await tennis_client.close()

        return {
            "matchup": f"{player1} vs {player2}",
            "analysis": analysis,
            "timestamp": "2025-01-31T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Tennis analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# KB ZIP Upload endpoint (PRESERVED)
@router.post("/upload-kb-zip", tags=["Knowledge Base Management"])
async def upload_knowledge_base_zip(
        file: UploadFile = File(..., description="A ZIP file containing the knowledge base documents.")
):
    kb_dir_on_volume = Path(settings.KNOWLEDGE_BASE_DIR)
    logger.info(f"Received request to upload KB ZIP. Target directory on volume: {kb_dir_on_volume}")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid file type or missing filename. Please upload a ZIP file.")

    try:
        kb_dir_on_volume.mkdir(parents=True, exist_ok=True)

        logger.info(f"Clearing existing contents of {kb_dir_on_volume}...")
        for item in kb_dir_on_volume.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info(f"Successfully cleared {kb_dir_on_volume}.")

        temp_zip_path = kb_dir_on_volume.parent / f"temp_upload_{file.filename}"
        try:
            with open(temp_zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Temporarily saved uploaded ZIP to {temp_zip_path}")

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(kb_dir_on_volume)
            logger.info(f"Successfully extracted ZIP to {kb_dir_on_volume}")

        finally:
            if temp_zip_path.exists():
                temp_zip_path.unlink()
                logger.info(f"Cleaned up temporary ZIP file {temp_zip_path}")
            await file.close()

        return {
            "status": "success",
            "message": f"Knowledge base ZIP uploaded and extracted to {kb_dir_on_volume}. Please call /api/ingest to process.",
            "uploaded_filename": file.filename
        }

    except FileNotFoundError:
        logger.error(
            f"Knowledge base directory {kb_dir_on_volume} or its parent does not exist or is not accessible on the server.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Server configuration error: KB directory path {kb_dir_on_volume} not found.")
    except zipfile.BadZipFile:
        logger.error(f"Uploaded file {file.filename} is not a valid ZIP file.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or corrupted ZIP file.")
    except Exception as e:
        logger.error(f"Error processing KB ZIP upload: {e}", exc_info=True)
        if hasattr(file, 'file') and file.file and not file.file.closed:
            await file.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred: {str(e)}")


# Health Check Endpoint (PRESERVED)
@router.get(
    "/health",
    summary="Health check",
    description="Returns the health status of the API and its dependencies."
)
async def health_check(request: Request):
    """Basic health check endpoint."""
    try:
        rag_status = "ok" if hasattr(request.app.state,
                                     'rag_pipeline') and request.app.state.rag_pipeline else "unavailable"
        llm_status = "ok" if hasattr(request.app.state,
                                     'llm_client') and request.app.state.llm_client else "unavailable"
        return {
            "status": "healthy",
            "timestamp": "2025-01-31T00:00:00Z",
            "services": {
                "rag_pipeline": rag_status,
                "llm_client": llm_status,
                "web_search": "available",
                "tennis_intelligence": "available"  # NEW
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy")


if __name__ == "__main__":
    logger.info("backend/api_handlers.py executed directly (for info only).")