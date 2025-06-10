# backend/api_handlers.py - COMPLETE VERSION WITH UPLOAD & INGEST
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, status, BackgroundTasks, UploadFile, File
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

try:
    from config import settings, tennis_config, validate_tennis_config
    from llm_interface.tennis_api_client import ProfessionalTennisAPIClient as TennisAPIClient
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse
    )
    from utils import RAGPipelineError
except ImportError as e:
    print(f"Import error in API handlers: {e}")


    # Fallback classes
    class MockConfig:
        def __init__(self):
            self.intelligence = type('obj', (), {'enable_betting_intelligence': True})()
            self.database = type('obj', (), {'enable_caching': True})()
            self.rate_limit_calls_per_minute = 60
            self.request_timeout_seconds = 30
            self.enable_fallback_data = True


    settings = MockConfig()
    tennis_config = MockConfig()


    def validate_tennis_config():
        return {"has_api_credentials": bool(tennis_config.credentials.rapidapi_key)}


    class QueryRequest(BaseModel):
        query_text: str;
        top_k_chunks: int = 3;
        model_name: Optional[str] = None


    class QueryResponse(BaseModel):
        answer: str;
        retrieved_chunks_details: List[Dict[str, Any]] = [];
        used_web_search: bool = False


    class IngestDirectoryRequest(BaseModel):
        directory_path: str


    class IngestDirectoryResponse(BaseModel):
        status: str;
        documents_processed: int;
        chunks_created: int


    class AvailableModelsResponse(BaseModel):
        models: List[str] = ["gemini-1.5-flash"]


    class RAGPipelineError(Exception):
        pass


    class TennisAPIClient:
        async def get_live_events(self): return []

        async def analyze_head_to_head(self, p1, p2): return {"analysis": "mock"}

        async def get_comprehensive_player_analysis(self, player): return {"analysis": "mock"}

        async def get_player_card(self, player_name: str): return {"error": "Not implemented"}

        async def get_events_by_date(self, d, m, y): return {}

        async def get_calendar_events(self, m, y): return {}

        async def get_player_previous_events(self, pid): return {}

        async def get_atp_rankings(self): return {}

        async def get_wta_rankings(self): return {}

        async def get_tournament_seasons(self, tid): return {}

        async def get_tournament_rounds(self, tid, sid): return {}

        async def close(self): pass

logger = logging.getLogger(__name__)

# Create main router for all endpoints
router = APIRouter()


# ===== CORE RAG ENDPOINTS =====

@router.post("/upload-kb-zip", summary="Upload knowledge base ZIP file")
async def upload_knowledge_base_zip(file: UploadFile = File(...)):
    """Upload and extract a ZIP file containing knowledge base documents"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    try:
        logger.info(f"📦 Uploading KB ZIP: {file.filename}")

        # Ensure the knowledge base directory exists
        kb_dir = Path(settings.KNOWLEDGE_BASE_DIR)
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            # Read and save the uploaded content
            content = await file.read()
            temp_file.write(content)
            temp_zip_path = temp_file.name

        try:
            # Extract the ZIP file to the knowledge base directory
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # Clear existing files first (optional - comment out if you want to append)
                if kb_dir.exists():
                    for existing_file in kb_dir.iterdir():
                        if existing_file.is_file():
                            existing_file.unlink()
                        elif existing_file.is_dir():
                            shutil.rmtree(existing_file)

                # Extract all files
                zip_ref.extractall(kb_dir)
                extracted_files = zip_ref.namelist()

            logger.info(f"✅ ZIP extracted to {kb_dir}")
            logger.info(f"📄 Extracted {len(extracted_files)} files")

            return {
                "status": "success",
                "message": f"Successfully uploaded and extracted {file.filename}",
                "extracted_files": len(extracted_files),
                "kb_directory": str(kb_dir),
                "files_extracted": extracted_files[:10]  # Show first 10 for preview
            }

        finally:
            # Clean up the temporary file
            Path(temp_zip_path).unlink(missing_ok=True)

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Failed to process ZIP upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process ZIP file: {str(e)}")


@router.post("/ingest", response_model=IngestDirectoryResponse, summary="Ingest documents into RAG system")
async def ingest_documents(payload: IngestDirectoryRequest, request: Request):
    """Ingest documents from a directory into the RAG vector store"""
    try:
        logger.info(f"🔄 Starting ingestion: {payload.directory_path}")

        # Validate the directory exists
        directory_path = Path(payload.directory_path)
        if not directory_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {payload.directory_path}")

        if not directory_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {payload.directory_path}")

        # Get the RAG pipeline from the app state
        rag_pipeline = request.app.state.rag_pipeline

        # Perform the ingestion
        documents_processed, chunks_created = await rag_pipeline.ingest_documents_from_directory(
            directory_path_str=str(directory_path)
        )

        logger.info(f"✅ Ingestion complete: {documents_processed} docs, {chunks_created} chunks")

        return IngestDirectoryResponse(
            status="success",
            documents_processed=documents_processed,
            chunks_created=chunks_created
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.get("/models", response_model=AvailableModelsResponse, summary="Get available LLM models")
async def get_available_models(request: Request):
    """Get list of available models from the LLM client"""
    try:
        llm_client = request.app.state.llm_client
        if hasattr(llm_client, 'list_available_models'):
            models = await llm_client.list_available_models()
        else:
            # Fallback for clients that don't have this method
            models = [settings.GEMINI_MODEL] if hasattr(settings, 'GEMINI_MODEL') else ["gemini-1.5-flash"]

        return AvailableModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Failed to get available models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


@router.post("/chat", response_model=QueryResponse, summary="RAG-enhanced chat with tennis intelligence")
async def rag_chat(payload: QueryRequest, request: Request):
    """Process a query using RAG with tennis intelligence and web search fallback"""
    try:
        logger.info(f"💬 RAG CHAT: Processing '{payload.query_text[:50]}...'")
        rag_pipeline = request.app.state.rag_pipeline

        # Use the enhanced tennis intelligence query method
        answer, sources = await rag_pipeline.query_with_tennis_intelligence(
            query_text=payload.query_text,
            top_k_chunks=payload.top_k_chunks,
            model_name_override=payload.model_name
        )

        # Determine if web search was used
        used_web_search = any(s.get("source_type") == "web_search" for s in sources)

        return QueryResponse(
            answer=answer,
            retrieved_chunks_details=sources,
            used_web_search=used_web_search
        )
    except Exception as e:
        logger.error(f"RAG chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


# ===== TENNIS INTELLIGENCE ENDPOINTS =====

class LiveEventsResponse(BaseModel):
    status: str = Field(description="Response status")
    events_count: int = Field(description="Number of live events found")
    events: List[Dict[str, Any]] = Field(description="Live event data")
    data_sources: List[str] = Field(description="Data sources used")
    cache_status: str = Field(description="Cache utilization status")
    response_time_ms: int = Field(description="Response time in milliseconds")


class MatchupAnalysisRequest(BaseModel):
    player1: str = Field(description="First player name")
    player2: str = Field(description="Second player name")


class MatchupAnalysisResponse(BaseModel):
    status: str = Field(description="Response status")
    matchup: str = Field(description="Matchup description")
    analysis: Dict[str, Any] = Field(description="Analysis data")
    confidence_level: str = Field(description="Analysis confidence level")


class PlayerAnalysisRequest(BaseModel):
    player_name: str = Field(description="Player name to analyze")


@router.get("/tennis/player/{player_name}/profile", summary="Get comprehensive player profile")
async def get_player_profile(player_name: str):
    client = None
    try:
        logger.info(f"✨ PLAYER CARD: Building profile for {player_name}")
        client = TennisAPIClient()
        profile_data = await client.get_player_card(player_name)
        if "error" in profile_data:
            raise HTTPException(status_code=404, detail=profile_data["error"])
        return profile_data
    except Exception as e:
        logger.error(f"Failed to get player profile for {player_name}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve player profile.")
    finally:
        if client: await client.close()


@router.get("/tennis/live-events", response_model=LiveEventsResponse, summary="Get live tennis events")
async def get_live_events(min_tier: Optional[str] = None):
    start_time = datetime.now()
    client = None
    try:
        logger.info("🔴 LIVE: Processing live events request")
        client = TennisAPIClient()
        events = await client.get_live_events()
        if min_tier:
            events = [e for e in events if _get_event_tier(e) >= _tier_to_number(min_tier)]
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        return LiveEventsResponse(
            status="success",
            events_count=len(events),
            events=events,
            data_sources=["RapidAPI"],
            cache_status="enabled" if getattr(tennis_config.database, 'enable_caching', False) else "disabled",
            response_time_ms=response_time
        )
    except Exception as e:
        logger.error(f"❌ Live events failed - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Live events service temporarily unavailable"})
    finally:
        if client: await client.close()


@router.post("/tennis/analyze-matchup", response_model=MatchupAnalysisResponse, summary="Head-to-head matchup analysis")
async def analyze_matchup(request: MatchupAnalysisRequest):
    client = None
    try:
        logger.info(f"🎾 ANALYSIS: Analyzing {request.player1} vs {request.player2}")
        client = TennisAPIClient()
        analysis = await client.analyze_head_to_head(request.player1, request.player2)
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        return MatchupAnalysisResponse(
            status="success",
            matchup=analysis.get("matchup", ""),
            analysis=analysis,
            confidence_level=analysis.get("confidence_level", "low")
        )
    except Exception as e:
        logger.error(f"❌ Matchup analysis failed - {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail={"error": "Matchup analysis service temporarily unavailable"})
    finally:
        if client: await client.close()


@router.post("/tennis/analyze-player", summary="Single player analysis")
async def analyze_player(request: PlayerAnalysisRequest):
    client = None
    try:
        logger.info(f"👤 PLAYER: Analyzing player {request.player_name}")
        client = TennisAPIClient()
        player_analysis = await client.get_comprehensive_player_analysis(request.player_name)
        if "error" in player_analysis:
            raise HTTPException(status_code=404, detail=player_analysis["error"])
        return {"status": "success", "player": request.player_name, "analysis": player_analysis}
    except Exception as e:
        logger.error(f"❌ Player analysis failed - {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail={"error": "Player analysis service temporarily unavailable"})
    finally:
        if client: await client.close()


# ===== RAW DATA ENDPOINTS =====

@router.get("/tennis/events/by-date/{year}/{month}/{day}", summary="Get events scheduled for a specific date")
async def get_events_by_date(year: int, month: int, day: int):
    client = None
    try:
        logger.info(f"🗓️ EVENTS BY DATE: Fetching for {year}-{month}-{day}")
        client = TennisAPIClient()
        data = await client.get_events_by_date(day, month, year)
        if not data:
            raise HTTPException(status_code=404, detail="No events found for this date.")
        return data
    except Exception as e:
        logger.error(f"Failed to get events for {year}-{month}-{day}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve events by date.")
    finally:
        if client: await client.close()


@router.get("/tennis/events/calendar/{year}/{month}", summary="Get the event calendar for a month")
async def get_event_calendar(year: int, month: int):
    client = None
    try:
        logger.info(f"📅 CALENDAR: Fetching for {year}-{month}")
        client = TennisAPIClient()
        data = await client.get_calendar_events(month, year)
        if not data:
            raise HTTPException(status_code=404, detail="No calendar data found for this month.")
        return data
    except Exception as e:
        logger.error(f"Failed to get calendar for {year}-{month}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve calendar data.")
    finally:
        if client: await client.close()


@router.get("/tennis/player/{player_id}/previous-events", summary="Get a player's previous events")
async def get_player_previous_events_handler(player_id: int):
    client = None
    try:
        logger.info(f"📜 PLAYER PREVIOUS EVENTS: Fetching for player ID {player_id}")
        client = TennisAPIClient()
        data = await client.get_player_previous_events(player_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"No previous events found for player ID {player_id}.")
        return data
    except Exception as e:
        logger.error(f"Failed to get previous events for player {player_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve player's previous events.")
    finally:
        if client: await client.close()


@router.get("/tennis/rankings/atp", summary="Get live ATP rankings")
async def get_atp_rankings_handler():
    client = None
    try:
        logger.info(f"🏆 RANKINGS: Fetching live ATP rankings")
        client = TennisAPIClient()
        data = await client.get_atp_rankings()
        if not data:
            raise HTTPException(status_code=404, detail="Could not retrieve ATP rankings.")
        return data
    except Exception as e:
        logger.error(f"Failed to get ATP rankings: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve ATP rankings.")
    finally:
        if client: await client.close()


@router.get("/tennis/rankings/wta", summary="Get live WTA rankings")
async def get_wta_rankings_handler():
    client = None
    try:
        logger.info(f"🏆 RANKINGS: Fetching live WTA rankings")
        client = TennisAPIClient()
        data = await client.get_wta_rankings()
        if not data:
            raise HTTPException(status_code=404, detail="Could not retrieve WTA rankings.")
        return data
    except Exception as e:
        logger.error(f"Failed to get WTA rankings: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve WTA rankings.")
    finally:
        if client: await client.close()


@router.get("/tennis/tournament/{tournament_id}/seasons", summary="Get available seasons for a tournament")
async def get_tournament_seasons_handler(tournament_id: int):
    client = None
    try:
        logger.info(f"📅 TOURNAMENT SEASONS: Fetching for tournament ID {tournament_id}")
        client = TennisAPIClient()
        data = await client.get_tournament_seasons(tournament_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"No seasons found for tournament ID {tournament_id}.")
        return data
    except Exception as e:
        logger.error(f"Failed to get seasons for tournament {tournament_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve tournament seasons.")
    finally:
        if client: await client.close()


@router.get("/tennis/tournament/{tournament_id}/season/{season_id}/rounds",
            summary="Get rounds for a tournament season")
async def get_tournament_rounds_handler(tournament_id: int, season_id: int):
    client = None
    try:
        logger.info(f"🔄 TOURNAMENT ROUNDS: Fetching for T:{tournament_id}, S:{season_id}")
        client = TennisAPIClient()
        data = await client.get_tournament_rounds(tournament_id, season_id)
        if not data:
            raise HTTPException(status_code=404,
                                detail=f"No rounds found for tournament {tournament_id}, season {season_id}.")
        return data
    except Exception as e:
        logger.error(f"Failed to get rounds for tournament {tournament_id}, season {season_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail="Failed to retrieve tournament rounds.")
    finally:
        if client: await client.close()


# ===== HELPER FUNCTIONS =====

def _get_event_tier(event: Dict[str, Any]) -> int:
    """Correctly handles the tournament name which is a direct string."""
    tournament_name = event.get("tournament", "").lower()
    if "grand slam" in tournament_name: return 5
    if "masters" in tournament_name or "wta 1000" in tournament_name: return 4
    if "500" in tournament_name: return 3
    if "250" in tournament_name: return 2
    return 1


def _tier_to_number(tier: str) -> int:
    return {"regional": 1, "standard": 2, "professional": 3, "premier": 4, "elite": 5}.get(tier.lower(), 1)