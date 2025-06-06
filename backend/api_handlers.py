# backend/api_handlers.py - FINAL, POLISHED VERSION
import logging
from fastapi import APIRouter, HTTPException, Request, status, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

try:
    from config import settings, tennis_config, validate_tennis_config
    from llm_interface.tennis_api_client import ProfessionalTennisAPIClient as TennisAPIClient
    from models import QueryRequest, QueryResponse
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
        query_text: str; top_k_chunks: int = 3; model_name: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str; retrieved_chunks_details: List[Dict[str, Any]] = []; used_web_search: bool = False

    class RAGPipelineError(Exception): pass

    class TennisAPIClient:
        async def get_live_events(self): return []
        async def analyze_head_to_head(self, p1, p2): return {"analysis": "mock"}
        async def get_comprehensive_player_analysis(self, player): return {"analysis": "mock"}
        async def get_player_card(self, player_name: str): return {"error": "Not implemented"}
        async def close(self): pass

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tennis", tags=["Tennis Intelligence"])

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


@router.get("/player/{player_name}/profile", summary="Get comprehensive player profile")
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


@router.get("/live-events", response_model=LiveEventsResponse, summary="Get live tennis events")
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
        return LiveEventsResponse(status="success", events_count=len(events), events=events, data_sources=["RapidAPI"], cache_status="enabled" if getattr(tennis_config.database, 'enable_caching', False) else "disabled", response_time_ms=response_time)
    except Exception as e:
        logger.error(f"❌ Live events failed - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Live events service temporarily unavailable"})
    finally:
        if client: await client.close()

@router.post("/analyze-matchup", response_model=MatchupAnalysisResponse, summary="Head-to-head matchup analysis")
async def analyze_matchup(request: MatchupAnalysisRequest):
    client = None
    try:
        logger.info(f"🎾 ANALYSIS: Analyzing {request.player1} vs {request.player2}")
        client = TennisAPIClient()
        analysis = await client.analyze_head_to_head(request.player1, request.player2)
        if "error" in analysis: raise HTTPException(status_code=404, detail=analysis["error"])
        return MatchupAnalysisResponse(status="success", matchup=analysis.get("matchup", ""), analysis=analysis, confidence_level=analysis.get("confidence_level", "low"))
    except Exception as e:
        logger.error(f"❌ Matchup analysis failed - {e}", exc_info=True)
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail={"error": "Matchup analysis service temporarily unavailable"})
    finally:
        if client: await client.close()

@router.post("/analyze-player", summary="Single player analysis")
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

@router.post("/chat", response_model=QueryResponse, summary="Tennis intelligence chat")
async def tennis_chat(payload: QueryRequest, request: Request):
    try:
        logger.info(f"💬 CHAT: Processing '{payload.query_text[:50]}...'")
        rag_pipeline = request.app.state.rag_pipeline
        answer, sources = await rag_pipeline.query_with_tennis_intelligence(query_text=payload.query_text, top_k_chunks=payload.top_k_chunks, model_name_override=payload.model_name)
        return QueryResponse(answer=answer, retrieved_chunks_details=sources)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Chat service temporarily unavailable"})

# Helper functions...
def _get_event_tier(event: Dict[str, Any]) -> int:
    """FIXED: Correctly handles the tournament name which is a direct string."""
    tournament_name = event.get("tournament", "").lower()
    if "grand slam" in tournament_name: return 5
    if "masters" in tournament_name or "wta 1000" in tournament_name: return 4
    if "500" in tournament_name: return 3
    if "250" in tournament_name: return 2
    return 1

def _tier_to_number(tier: str) -> int:
    return {"regional": 1, "standard": 2, "professional": 3, "premier": 4, "elite": 5}.get(tier.lower(), 1)