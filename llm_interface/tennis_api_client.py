# llm_interface/tennis_api_client.py - Quick Win #1: Real H2H Data
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
            self.endpoints = type('obj', (), {
                'primary_live_api': 'https://api.edgeai.pro/api/tennis/events/live',
                'rapidapi_base': 'https://tennisapi1.p.rapidapi.com/api/tennis',
                'backup_endpoints': []
            })()
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
    """Professional player data structure"""
    id: Optional[int]
    name: str
    ranking: Optional[int]
    points: Optional[int]
    country: Optional[str]
    last_updated: datetime


@dataclass
class MatchData:
    """Professional match data structure"""
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
    """Professional tennis data management with database integration."""

    def __init__(self, config: TennisAPIConfig):
        self.config = config
        db_url = getattr(config.database, 'database_url', 'sqlite:///tennis_intelligence.db')
        self.db_path = self._extract_db_path(db_url)
        self._init_database()

    def _extract_db_path(self, database_url: str) -> str:
        if database_url.startswith("sqlite:///"):
            return database_url.replace("sqlite:///", "")
        return "tennis_intelligence.db"

    def _init_database(self):
        db_path = Path(self.db_path)
        if not db_path.exists():
            logger.warning(f"Database not found at {db_path}. It will be created on first connection.")

    @asynccontextmanager
    async def get_db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error to {self.db_path}: {e}")
            raise
        finally:
            if conn:
                conn.close()

    async def get_cached_player_data(self, player_name: str) -> Optional[PlayerData]:
        if not self.config.database.enable_caching: return None
        try:
            async with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cache_cutoff = datetime.now() - timedelta(minutes=self.config.database.cache_duration_minutes)
                cursor.execute(
                    "SELECT * FROM players WHERE LOWER(name) = LOWER(?) AND updated_at > ? ORDER BY updated_at DESC LIMIT 1",
                    (player_name, cache_cutoff.isoformat()))
                row = cursor.fetchone()
                if row:
                    return PlayerData(id=row['api_player_id'], name=row['name'], ranking=None, points=None,
                                      country=row['country_code'],
                                      last_updated=datetime.fromisoformat(row['updated_at']))
        except Exception as e:
            logger.warning(f"Cache lookup failed for {player_name}: {e}")
        return None


class ProfessionalTennisAPIClient:
    """Professional Tennis API Client - Implementing Quick Win #1: Real H2H Data"""

    def __init__(self, config: Optional[TennisAPIConfig] = None):
        self.config = config or tennis_config
        self.data_manager = TennisDataManager(self.config)
        self._client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        self._api_call_count = 0
        self._last_reset = datetime.now()
        logger.info(
            f"Professional Tennis API Client initialized. Primary API: {self.config.endpoints.primary_live_api}")

    async def _fetch_from_rapidapi(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Dict[str, Any]]:
        """Generic and robust fetcher for all RapidAPI endpoints."""
        if not self.config.credentials.rapidapi_key:
            logger.warning("RapidAPI key not configured. Cannot make the call.")
            return None

        headers = {
            'X-RapidAPI-Key': self.config.credentials.rapidapi_key,
            'X-RapidAPI-Host': self.config.credentials.rapidapi_host
        }
        url = f"{self.config.endpoints.rapidapi_base}{endpoint}"

        try:
            logger.debug(f"Calling RapidAPI: GET {url} with params: {params}")
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"RapidAPI returned error {e.response.status_code} for {url}: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error calling RapidAPI at {url}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {url}: {e}")
        return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[PlayerData]:
        """Parses the search result from RapidAPI to find the best player match."""
        if not data.get('results'):
            logger.warning(f"No results found in API search for '{player_name}'")
            return None

        for result in data['results']:
            if result.get('type') == 'player':
                entity = result.get('entity', {})
                if player_name.lower() in entity.get('name', '').lower():
                    logger.info(
                        f"Found player match for '{player_name}': {entity.get('name')} (ID: {entity.get('id')})")
                    return PlayerData(
                        id=entity.get('id'),
                        name=entity.get('name'),
                        ranking=entity.get('ranking'),
                        country=entity.get('country', {}).get('name'),
                        points=entity.get('rankingPoints'),
                        last_updated=datetime.now()
                    )
        logger.warning(f"No direct 'player' type match found in API search for '{player_name}'")
        return None

    async def _get_professional_player_data(self, player_name: str) -> Optional[PlayerData]:
        """Gets player data by using the /search/{playerName} endpoint from the API docs."""
        search_data = await self._fetch_from_rapidapi(f"/search/{player_name}")
        if search_data:
            return self._parse_player_search(search_data, player_name)

        logger.warning(f"Could not fetch data for player: {player_name}")
        return None

    async def analyze_head_to_head(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """
        Quick Win #1: Implements true H2H analysis.
        Finds players, finds a live event between them, and calls the /duel endpoint.
        """
        logger.info(f"🎾 Implementing REAL H2H Analysis for: {player1_name} vs {player2_name}")
        try:
            p1_data = await self._get_professional_player_data(player1_name)
            p2_data = await self._get_professional_player_data(player2_name)

            if not p1_data or not p2_data:
                raise ValueError("Could not retrieve API data for one or both players.")

            event_id = None
            primary_api_url = self.config.endpoints.primary_live_api
            logger.info(f"Searching for live event in primary feed: {primary_api_url}")
            live_events_response = await self._client.get(primary_api_url)

            if live_events_response.status_code == 200:
                live_events_data = live_events_response.json()
                if live_events_data and 'events' in live_events_data:
                    for event in live_events_data['events']:
                        home_player_name = event.get('homePlayer', {}).get('name', '').lower()
                        away_player_name = event.get('awayPlayer', {}).get('name', '').lower()

                        if (p1_data.name.lower() in home_player_name and p2_data.name.lower() in away_player_name) or \
                                (p2_data.name.lower() in home_player_name and p1_data.name.lower() in away_player_name):
                            event_id = event.get('id')
                            logger.info(f"Found live event ID {event_id} for matchup.")
                            break
            else:
                logger.warning(f"Primary live events API call failed with status: {live_events_response.status_code}")

            h2h_stats = {}
            if event_id:
                logger.info(f"Fetching H2H duel data for event ID: {event_id}")
                duel_data = await self._fetch_from_rapidapi(f"/event/{event_id}/duel")
                h2h_stats = duel_data if duel_data else {"error": "Failed to fetch H2H duel data from API."}
            else:
                logger.warning(
                    f"Could not find a live event ID for {player1_name} vs {player2_name}. H2H data will be limited.")
                h2h_stats = {"note": "No live event found, so no specific H2H duel data could be retrieved."}

            analysis = {
                "matchup": f"{p1_data.name} vs {p2_data.name}",
                "player1_profile": asdict(p1_data),
                "player2_profile": asdict(p2_data),
                "historical_h2h": h2h_stats,
                "confidence_level": "high" if event_id and 'error' not in h2h_stats else "low",
                "analysis_timestamp": datetime.now().isoformat()
            }
            return analysis

        except Exception as e:
            logger.error(f"H2H analysis failed for {player1_name} vs {player2_name}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during H2H analysis: {str(e)}"}

    async def get_live_events(self) -> List[Dict[str, Any]]:
        """Gets live events from the primary API and returns them."""
        logger.info("🔴 LIVE: Fetching professional live tennis events...")
        primary_api_url = self.config.endpoints.primary_live_api
        try:
            response = await self._client.get(primary_api_url)
            response.raise_for_status()
            data = response.json()
            return data.get('events', [])
        except Exception as e:
            logger.error(f"Live events fetch failed: {e}", exc_info=True)
            return []

    async def get_comprehensive_player_analysis(self, player_name: str) -> Dict[str, Any]:
        """Fetches and returns data for a single player."""
        logger.info(f"👤Fetching comprehensive analysis for player: {player_name}")
        player_data = await self._get_professional_player_data(player_name)
        if not player_data:
            return {"error": "Player not found or data unavailable."}
        return asdict(player_data)

    async def get_current_tournaments(self) -> Dict[str, Any]:
        """Placeholder for fetching tournament data. To be implemented as a future quick win."""
        logger.info("🏆 Fetching current tournament data (placeholder)...")
        return {"note": "Tournament data endpoint not yet fully implemented."}

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
        logger.info("Professional Tennis API Client session closed.")