# backend/api_handlers.py
# FINAL FIX: The admin key verification now reads directly from the environment to ensure correctness.

import logging
import zipfile
import tempfile
import shutil
import os  # <<< IMPORT OS TO READ ENVIRONMENT VARIABLES DIRECTLY
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, status, BackgroundTasks, UploadFile, File, Header, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from config import settings, tennis_config

try:
    from database.db_manager import DatabaseManager
    from llm_interface.tennis_api_client import ProfessionalTennisAPIClient as TennisAPIClient
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse
    )
    from utils import RAGPipelineError
except ImportError as e:
    # This block is for fallback purposes and can be ignored if your environment is set up.
    print(f"Import error in API handlers: {e}")


    class MockSettings:
        pass


    settings = MockSettings()


    class MockTennisConfig:
        database = type('obj', (), {'enable_caching': False})()


    tennis_config = MockTennisConfig()


    class QueryRequest(BaseModel):
        query_text: str; top_k_chunks: int = 3; model_name: Optional[str] = None


    class QueryResponse(BaseModel):
        answer: str; retrieved_chunks_details: List = []; used_web_search: bool = False


    class IngestDirectoryRequest(BaseModel):
        directory_path: str


    class IngestDirectoryResponse(BaseModel):
        status: str; documents_processed: int; chunks_created: int


    class AvailableModelsResponse(BaseModel):
        models: List[str] = []


    class RAGPipelineError(Exception):
        pass


    class TennisAPIClient:
        pass


    class DatabaseManager:
        pass

logger = logging.getLogger(__name__)


# --- Admin Security Dependency ---
async def verify_admin_key(x_admin_key: str = Header(None)):
    """Verifies the admin API key provided in the request header."""
    if not x_admin_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin API Key header missing")

    # <<< THE FINAL FIX IS HERE >>>
    # Read the correct key directly from the server's environment at the time of the request.
    # This bypasses any potential config loading issues.
    server_admin_key = os.getenv("ADMIN_API_KEY")

    if not server_admin_key or x_admin_key != server_admin_key:
        logger.warning(f"Failed admin access attempt with key: {x_admin_key}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Admin API Key")
    return x_admin_key


# Create main router for all endpoints
router = APIRouter()


# --- ADMIN ENDPOINT ---
@router.post(
    "/admin/save-match",
    summary="Save a single enriched match data object",
    dependencies=[Depends(verify_admin_key)]
)
async def save_single_match(match_data: Dict[str, Any]):
    """
    Receives a single enriched match data object from a trusted client and saves it to the database.
    """
    logger.info(f"Received request to save match ID: {match_data.get('id', 'Unknown')}")
    db_manager = None
    try:
        db_manager = DatabaseManager()
        db_manager.process_match_data(match_data)
        logger.info(f"Successfully saved match ID: {match_data.get('id')}")
        return {"status": "success", "message": "Match data processed and saved successfully.",
                "match_id": match_data.get('id')}
    except Exception as e:
        logger.error(f"Error saving match data for ID {match_data.get('id', 'Unknown')}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to save match data: {str(e)}")
    finally:
        if db_manager:
            db_manager.close()


# ===== CORE RAG ENDPOINTS =====

@router.post("/upload-kb-zip", summary="Upload knowledge base ZIP file")
async def upload_knowledge_base_zip(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    try:
        kb_dir = Path(settings.KNOWLEDGE_BASE_DIR)
        kb_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_zip_path = temp_file.name
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                if kb_dir.exists():
                    for item in kb_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                zip_ref.extractall(kb_dir)
                extracted_files = zip_ref.namelist()
            return {"status": "success", "message": f"Successfully uploaded and extracted {file.filename}",
                    "extracted_files": len(extracted_files), "kb_directory": str(kb_dir)}
        finally:
            Path(temp_zip_path).unlink(missing_ok=True)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process ZIP file: {str(e)}")


@router.post("/ingest", response_model=IngestDirectoryResponse, summary="Ingest documents into RAG system")
async def ingest_documents(payload: IngestDirectoryRequest, request: Request):
    try:
        directory_path = Path(payload.directory_path)
        if not directory_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {payload.directory_path}")
        rag_pipeline = request.app.state.rag_pipeline
        docs_processed, chunks_created = await rag_pipeline.ingest_documents_from_directory(str(directory_path))
        return IngestDirectoryResponse(status="success", documents_processed=docs_processed,
                                       chunks_created=chunks_created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/models", response_model=AvailableModelsResponse, summary="Get available LLM models")
async def get_available_models(request: Request):
    try:
        llm_client = request.app.state.llm_client
        models = await llm_client.list_available_models() if hasattr(llm_client, 'list_available_models') else [
            settings.GEMINI_MODEL]
        return AvailableModelsResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


@router.post("/chat", response_model=QueryResponse, summary="RAG-enhanced chat with tennis intelligence")
async def rag_chat(payload: QueryRequest, request: Request):
    try:
        rag_pipeline = request.app.state.rag_pipeline
        answer, sources = await rag_pipeline.query_with_tennis_intelligence(payload.query_text, payload.top_k_chunks,
                                                                            payload.model_name)
        used_web_search = any(s.get("source_type") == "web_search" for s in sources)
        return QueryResponse(answer=answer, retrieved_chunks_details=sources, used_web_search=used_web_search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


# ===== TENNIS INTELLIGENCE ENDPOINTS =====

class LiveEventsResponse(BaseModel):
    status: str;
    events_count: int;
    events: List[Dict[str, Any]];
    data_sources: List[str];
    cache_status: str;
    response_time_ms: int


class MatchupAnalysisRequest(BaseModel):
    player1: str;
    player2: str


class MatchupAnalysisResponse(BaseModel):
    status: str;
    matchup: str;
    analysis: Dict[str, Any];
    confidence_level: str


class PlayerAnalysisRequest(BaseModel):
    player_name: str


@router.get("/tennis/player/{player_name}/profile", summary="Get comprehensive player profile")
async def get_player_profile(player_name: str):
    client = None
    try:
        client = TennisAPIClient()
        profile_data = await client.get_player_card(player_name)
        if "error" in profile_data: raise HTTPException(status_code=404, detail=profile_data["error"])
        return profile_data
    finally:
        if client: await client.close()


@router.get("/tennis/live-events", response_model=LiveEventsResponse, summary="Get live tennis events")
async def get_live_events(min_tier: Optional[str] = None):
    start_time = datetime.now()
    client = None
    try:
        client = TennisAPIClient()
        events = await client.get_live_events()
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        return LiveEventsResponse(status="success", events_count=len(events), events=events,
                                  data_sources=["EdgeAI", "RapidAPI"], cache_status="disabled",
                                  response_time_ms=response_time)
    finally:
        if client: await client.close()


@router.post("/tennis/analyze-matchup", response_model=MatchupAnalysisResponse, summary="Head-to-head matchup analysis")
async def analyze_matchup(request: MatchupAnalysisRequest):
    client = None
    try:
        client = TennisAPIClient()
        analysis = await client.analyze_head_to_head(request.player1, request.player2)
        if "error" in analysis: raise HTTPException(status_code=404, detail=analysis["error"])
        return MatchupAnalysisResponse(status="success", matchup=analysis.get("matchup", ""), analysis=analysis,
                                       confidence_level=analysis.get("confidence_level", "low"))
    finally:
        if client: await client.close()


@router.post("/tennis/analyze-player", summary="Single player analysis")
async def analyze_player(request: PlayerAnalysisRequest):
    client = None
    try:
        client = TennisAPIClient()
        player_analysis = await client.get_comprehensive_player_analysis(request.player_name)
        if "error" in player_analysis: raise HTTPException(status_code=404, detail=player_analysis["error"])
        return {"status": "success", "player": request.player_name, "analysis": player_analysis}
    finally:
        if client: await client.close()


# ===== RAW DATA ENDPOINTS =====

@router.get("/tennis/events/by-date/{year}/{month}/{day}", summary="Get events scheduled for a specific date")
async def get_events_by_date(year: int, month: int, day: int):
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_events_by_date(day, month, year)
        if not data: raise HTTPException(status_code=404, detail="No events found for this date.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/events/calendar/{year}/{month}", summary="Get the event calendar for a month")
async def get_event_calendar(year: int, month: int):
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_calendar_events(month, year)
        if not data: raise HTTPException(status_code=404, detail="No calendar data found for this month.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/player/{player_id}/previous-events", summary="Get a player's previous events")
async def get_player_previous_events_handler(player_id: int):
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_player_previous_events(player_id)
        if not data: raise HTTPException(status_code=404, detail=f"No previous events found for player ID {player_id}.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/rankings/atp", summary="Get live ATP rankings")
async def get_atp_rankings_handler():
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_atp_rankings()
        if not data: raise HTTPException(status_code=404, detail="Could not retrieve ATP rankings.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/rankings/wta", summary="Get live WTA rankings")
async def get_wta_rankings_handler():
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_wta_rankings()
        if not data: raise HTTPException(status_code=404, detail="Could not retrieve WTA rankings.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/tournament/{tournament_id}/seasons", summary="Get available seasons for a tournament")
async def get_tournament_seasons_handler(tournament_id: int):
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_tournament_seasons(tournament_id)
        if not data: raise HTTPException(status_code=404, detail=f"No seasons found for tournament ID {tournament_id}.")
        return data
    finally:
        if client: await client.close()


@router.get("/tennis/tournament/{tournament_id}/season/{season_id}/rounds",
            summary="Get rounds for a tournament season")
async def get_tournament_rounds_handler(tournament_id: int, season_id: int):
    client = None
    try:
        client = TennisAPIClient()
        data = await client.get_tournament_rounds(tournament_id, season_id)
        if not data: raise HTTPException(status_code=404,
                                         detail=f"No rounds found for tournament {tournament_id}, season {season_id}.")
        return data
    finally:
        if client: await client.close()