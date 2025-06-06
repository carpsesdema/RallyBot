# llm_interface/tennis_api_client.py - ADDED DEBUG LOGGING
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
            self.endpoints = type('obj', (), {'rapidapi_base': 'https://tennisapi1.p.rapidapi.com/api/tennis'})()
            self.credentials = type('obj', (), {'rapidapi_key': None, 'rapidapi_host': 'tennisapi1.p.rapidapi.com'})()
            self.request_timeout_seconds = 30


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


class ProfessionalTennisAPIClient:
    def __init__(self, config: Optional[TennisAPIConfig] = None):
        self.config = config or tennis_config
        self._client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        logger.info("Professional Tennis API Client initialized to use RapidAPI as the primary source.")

    async def _fetch_from_rapidapi(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Dict[str, Any]]:
        if not self.config.credentials.rapidapi_key:
            logger.error("RapidAPI key is not configured. Cannot make API calls.")
            return None
        headers = {'X-RapidAPI-Key': self.config.credentials.rapidapi_key,
                   'X-RapidAPI-Host': self.config.credentials.rapidapi_host}
        url = f"{self.config.endpoints.rapidapi_base}{endpoint}"
        try:
            logger.info(f"Calling RapidAPI: GET {url}")
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"RapidAPI returned HTTP {e.response.status_code} for {url}: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling RapidAPI at {url}: {e}", exc_info=True)
        return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[PlayerData]:
        # --- ADDED DEBUG LOGGING HERE ---
        logger.info(f"Raw API response for '{player_name}' search: {json.dumps(data, indent=2)}")

        if not data or not data.get('results'):
            logger.warning(f"No 'results' key in API response for '{player_name}'.")
            return None

        for result in data['results']:
            if result.get('type') == 'player':
                entity = result.get('entity', {})
                # Making the search less strict to catch names like "Rafael Nadal" from "nadal"
                if player_name.lower() in entity.get('name', '').lower():
                    logger.info(
                        f"Found player match for '{player_name}': {entity.get('name')} (ID: {entity.get('id')})")
                    return PlayerData(id=entity.get('id'), name=entity.get('name'), ranking=entity.get('ranking'),
                                      country=entity.get('country', {}).get('name'), points=entity.get('rankingPoints'),
                                      last_updated=datetime.now())

        logger.warning(f"No player match found in API results for '{player_name}'.")
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
        """Gets live events from the RapidAPI endpoint as requested by the client."""
        logger.info("🔴 LIVE: Fetching live events from RapidAPI...")
        live_data = await self._fetch_from_rapidapi("/events/live")
        if live_data and 'events' in live_data:
            matches = [asdict(match) for event in live_data['events'] if (match := self._convert_to_match_data(event))]
            logger.info(f"✅ LIVE: Retrieved and parsed {len(matches)} live events from RapidAPI.")
            return matches
        logger.warning(
            "Could not retrieve live events from RapidAPI. The API might have no live matches, or there was an error.")
        return []

    async def analyze_head_to_head(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        logger.info(f"🎾 H2H Analysis for: {player1_name} vs {player2_name}")
        try:
            p1_data, p2_data = await asyncio.gather(self._get_professional_player_data(player1_name),
                                                    self._get_professional_player_data(player2_name))
            if not p1_data or not p2_data: raise ValueError("Could not retrieve API data for one or both players.")

            event_id = None
            live_events_data = await self._fetch_from_rapidapi("/events/live")
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

    async def get_player_card(self, player_name: str) -> Dict[str, Any]:
        player_base_data = await self._get_professional_player_data(player_name)
        if not player_base_data or not player_base_data.id:
            return {"error": f"Player '{player_name}' not found."}

        player_id = player_base_data.id
        logger.info(f"Fetching recent events and rankings for player ID {player_id} concurrently.")
        results = await asyncio.gather(
            self._fetch_from_rapidapi(f"/player/{player_id}/events/previous/0"),
            self._fetch_from_rapidapi(f"/player/{player_id}/rankings"),
            return_exceptions=True
        )
        recent_events_data, rankings_data = results

        recent_form = []
        if isinstance(recent_events_data, dict) and 'events' in recent_events_data:
            for event in recent_events_data['events']:
                try:
                    winner_code = event.get('winnerCode')
                    is_home_player = event.get('homePlayer', {}).get('id') == player_id
                    result = 'W' if (is_home_player and winner_code == 1) or (
                            not is_home_player and winner_code == 2) else 'L'
                    recent_form.append(result)
                except (TypeError, KeyError):
                    pass

        ranking_history = rankings_data.get('rankings', []) if isinstance(rankings_data, dict) else []

        return {"profile": asdict(player_base_data), "recent_form_string": "-".join(recent_form[:5]),
                "ranking_history": ranking_history, "data_retrieved_at": datetime.now().isoformat()}

    async def get_comprehensive_player_analysis(self, player_name: str) -> Dict[str, Any]:
        player_data = await self._get_professional_player_data(player_name)
        return asdict(player_data) if player_data else {"error": "Player not found."}

    async def get_current_tournaments(self) -> Dict[str, Any]:
        return {"note": "Tournament data endpoint not yet implemented."}

    async def close(self):
        await self._client.aclose()
        logger.info("Professional Tennis API Client session closed.")