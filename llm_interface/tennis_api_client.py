# llm_interface/tennis_api_client.py - FINAL, ROBUST VERSION
import httpx
import logging
import sqlite3
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import asyncio
from contextlib import asynccontextmanager

try:
    from config import tennis_config, TennisAPIConfig
except ImportError:
    # Fallback if config not available
    class MockTennisAPIConfig:
        def __init__(self):
            self.endpoints = type('obj', (), {'primary_live_api': 'https://api.edgeai.pro/api/tennis',
                                              'rapidapi_base': 'https://tennisapi1.p.rapidapi.com/api/tennis',
                                              'backup_endpoints': []})()
            self.credentials = type('obj', (), {'rapidapi_key': None, 'rapidapi_host': 'tennisapi1.p.rapidapi.com',
                                                'edgeai_key': None})()
            self.intelligence = type('obj', (),
                                     {'enable_betting_intelligence': True, 'default_analysis_depth': 'comprehensive'})()
            self.database = type('obj', (), {'database_url': 'sqlite:///tennis_intelligence.db', 'enable_caching': True,
                                             'cache_duration_minutes': 30})()
            self.request_timeout_seconds = 30
            self.max_retries = 3
            self.rate_limit_calls_per_minute = 60
            self.enable_fallback_data = True


    tennis_config = MockTennisAPIConfig()
    TennisAPIConfig = MockTennisAPIConfig

logger = logging.getLogger(__name__)


@dataclass
class PlayerData:
    id: Optional[int]
    name: str
    ranking: Optional[int]
    points: Optional[int]
    country: Optional[str]
    last_updated: datetime


@dataclass
class MatchData:
    id: int
    player1: PlayerData
    player2: PlayerData
    tournament: str
    surface: str
    status: str
    odds: Optional[Dict[str, float]]
    live_score: Optional[Dict[str, Any]]
    betting_analysis: Optional[Dict[str, Any]]


class TennisDataManager:
    def __init__(self, config: TennisAPIConfig):
        self.config = config
        db_url = getattr(config.database, 'database_url', 'sqlite:///tennis_intelligence.db')
        self.db_path = self._extract_db_path(db_url)

    def _extract_db_path(self, database_url: str) -> str:
        return database_url.replace("sqlite:///", "") if database_url.startswith(
            "sqlite:///") else "tennis_intelligence.db"

    @asynccontextmanager
    async def get_db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn: conn.close()


class ProfessionalTennisAPIClient:
    def __init__(self, config: Optional[TennisAPIConfig] = None):
        self.config = config or tennis_config
        self._client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        logger.info("Professional Tennis API Client initialized.")

    async def _fetch_from_primary_api(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        MODIFIED: Calls the primary API endpoint without any authentication headers,
        assuming it is a public endpoint as per client's feedback.
        """
        url = f"{self.config.endpoints.primary_live_api.rstrip('/')}{endpoint}"
        logger.info(f"Calling Primary API (as public): GET {url}")

        try:
            # Making the call with NO headers
            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Primary API returned HTTP {e.response.status_code} for {url}: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling Primary API at {url}: {e}", exc_info=True)
        return None

    async def _fetch_from_rapidapi(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Dict[str, Any]]:
        if not self.config.credentials.rapidapi_key: return None
        headers = {'X-RapidAPI-Key': self.config.credentials.rapidapi_key,
                   'X-RapidAPI-Host': self.config.credentials.rapidapi_host}
        url = f"{self.config.endpoints.rapidapi_base}{endpoint}"
        try:
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error calling RapidAPI at {url}: {e}", exc_info=True)
        return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[PlayerData]:
        if not data or not data.get('results'): return None
        for result in data['results']:
            if result.get('type') == 'player' and player_name.lower() in result.get('entity', {}).get('name',
                                                                                                      '').lower():
                entity = result['entity']
                logger.info(f"Found player match for '{player_name}': {entity.get('name')}")
                return PlayerData(id=entity.get('id'), name=entity.get('name'), ranking=entity.get('ranking'),
                                  country=entity.get('country', {}).get('name'), points=entity.get('rankingPoints'),
                                  last_updated=datetime.now())
        return None

    async def _get_professional_player_data(self, player_name: str) -> Optional[PlayerData]:
        search_data = await self._fetch_from_rapidapi(f"/search/{player_name}")
        return self._parse_player_search(search_data, player_name) if search_data else None

    def _convert_to_match_data(self, event_data: Dict[str, Any]) -> Optional[MatchData]:
        try:
            p1 = event_data['homePlayer']
            p2 = event_data['awayPlayer']
            t = event_data['tournament']
            player1 = PlayerData(id=p1.get('id'), name=p1.get('name'), ranking=p1.get('ranking'), points=None,
                                 country=p1.get('country', {}).get('name'), last_updated=datetime.now())
            player2 = PlayerData(id=p2.get('id'), name=p2.get('name'), ranking=p2.get('ranking'), points=None,
                                 country=p2.get('country', {}).get('name'), last_updated=datetime.now())
            return MatchData(id=event_data['id'], player1=player1, player2=player2, tournament=t.get('name', 'N/A'),
                             surface=t.get('groundType', 'N/A'),
                             status=event_data.get('status', {}).get('description', 'N/A'), odds=event_data.get('odds'),
                             live_score=event_data.get('liveScore'), betting_analysis=None)
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse event data due to schema mismatch: {e}.")
            return None

    async def get_live_events(self) -> List[Dict[str, Any]]:
        logger.info("🔴 LIVE: Fetching live events from primary API (assuming public access)...")
        live_data = await self._fetch_from_primary_api("/events/live")
        if live_data and 'events' in live_data:
            matches = [asdict(match) for event in live_data['events'] if (match := self._convert_to_match_data(event))]
            logger.info(f"✅ LIVE: Retrieved and parsed {len(matches)} live events.")
            return matches
        logger.warning("Could not retrieve live events. The API might have no live matches, or there was an error.")
        return []

    async def analyze_head_to_head(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        logger.info(f"🎾 H2H Analysis for: {player1_name} vs {player2_name}")
        try:
            p1_data, p2_data = await asyncio.gather(self._get_professional_player_data(player1_name),
                                                    self._get_professional_player_data(player2_name))
            if not p1_data or not p2_data: raise ValueError("Could not retrieve API data for one or both players.")

            event_id = None
            live_events_data = await self._fetch_from_primary_api("/events/live")
            if live_events_data and 'events' in live_events_data:
                for event in live_events_data['events']:
                    home_name, away_name = event.get('homePlayer', {}).get('name', '').lower(), event.get('awayPlayer',
                                                                                                          {}).get(
                        'name', '').lower()
                    if (p1_data.name.lower() in home_name and p2_data.name.lower() in away_name) or (
                            p2_data.name.lower() in home_name and p1_data.name.lower() in away_name):
                        event_id = event.get('id')
                        logger.info(f"Found live event ID {event_id} for H2H duel.")
                        break

            h2h_stats = {"note": "No live event found, so no specific H2H duel data could be retrieved."}
            if event_id:
                duel_data = await self._fetch_from_rapidapi(f"/event/{event_id}/duel")
                h2h_stats = duel_data if duel_data else {"error": "Failed to fetch H2H duel data from API."}

            return {"matchup": f"{p1_data.name} vs {p2_data.name}", "player1_profile": asdict(p1_data),
                    "player2_profile": asdict(p2_data), "historical_h2h": h2h_stats,
                    "confidence_level": "high" if event_id and 'error' not in h2h_stats else "low",
                    "analysis_timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"H2H analysis failed: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during H2H analysis: {str(e)}"}

    async def get_comprehensive_player_analysis(self, player_name: str) -> Dict[str, Any]:
        player_data = await self._get_professional_player_data(player_name)
        return asdict(player_data) if player_data else {"error": "Player not found."}

    async def get_current_tournaments(self) -> Dict[str, Any]:
        return {"note": "Tournament data endpoint not yet implemented."}

    async def close(self):
        await self._client.aclose()
        logger.info("Professional Tennis API Client session closed.")