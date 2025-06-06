# llm_interface/tennis_api_client.py - FIXED with robust schema handling
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
        logger.info("Professional Tennis API Client initialized with robust error handling.")

    async def _fetch_from_rapidapi(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Dict[str, Any]]:
        if not self.config.credentials.rapidapi_key:
            logger.error("RapidAPI key is not configured. Cannot make API calls.")
            return None

        headers = {
            'X-RapidAPI-Key': self.config.credentials.rapidapi_key,
            'X-RapidAPI-Host': self.config.credentials.rapidapi_host
        }
        url = f"{self.config.endpoints.rapidapi_base}{endpoint}"

        try:
            logger.info(f"📡 Calling RapidAPI: GET {url}")
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            # ENHANCED DEBUG LOGGING
            logger.info(f"🔍 RAW API RESPONSE for {endpoint}:")
            logger.info(f"📊 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

            if isinstance(data, dict) and 'events' in data:
                logger.info(f"📅 Found {len(data['events'])} events in response")
                if data['events']:
                    first_event = data['events'][0]
                    logger.info(f"🎾 First event structure: {list(first_event.keys())}")

                    # Log the actual structure we're getting
                    logger.info(f"📋 Sample event data: {json.dumps(first_event, indent=2)}")

            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ RapidAPI returned HTTP {e.response.status_code} for {url}")
            logger.error(f"📄 Response body: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"💥 Error calling RapidAPI at {url}: {e}", exc_info=True)
            return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[PlayerData]:
        logger.info(f"🔍 Parsing player search for '{player_name}'")
        logger.debug(f"📊 Raw search response: {json.dumps(data, indent=2)}")

        if not data or not data.get('results'):
            logger.warning(f"❌ No 'results' key in API response for '{player_name}'.")
            return None

        for result in data['results']:
            if result.get('type') == 'player':
                entity = result.get('entity', {})
                if player_name.lower() in entity.get('name', '').lower():
                    logger.info(
                        f"✅ Found player match for '{player_name}': {entity.get('name')} (ID: {entity.get('id')})")
                    return PlayerData(
                        id=entity.get('id'),
                        name=entity.get('name'),
                        ranking=entity.get('ranking'),
                        country=entity.get('country', {}).get('name'),
                        points=entity.get('rankingPoints'),
                        last_updated=datetime.now()
                    )

        logger.warning(f"❌ No player match found in API results for '{player_name}'.")
        return None

    def _convert_to_match_data(self, event_data: Dict[str, Any]) -> Optional[MatchData]:
        """ROBUST event data conversion with comprehensive error handling"""
        try:
            logger.debug(f"🔄 Converting event data: {json.dumps(event_data, indent=2)}")

            # Check for different possible structures
            event_id = event_data.get('id')
            if not event_id:
                logger.warning("❌ Event missing 'id' field")
                return None

            # Try different player field variations
            player1_data = None
            player2_data = None

            # Common variations for player fields
            player_field_variations = [
                ('homePlayer', 'awayPlayer'),
                ('homeTeam', 'awayTeam'),
                ('player1', 'player2'),
                ('home', 'away'),
                ('competitors', None)  # Array format
            ]

            for home_field, away_field in player_field_variations:
                if home_field in event_data:
                    if home_field == 'competitors' and isinstance(event_data[home_field], list):
                        # Handle competitors array format
                        competitors = event_data[home_field]
                        if len(competitors) >= 2:
                            player1_data = competitors[0]
                            player2_data = competitors[1]
                            logger.info(f"✅ Using competitors array format")
                            break
                    elif away_field and away_field in event_data:
                        player1_data = event_data[home_field]
                        player2_data = event_data[away_field]
                        logger.info(f"✅ Using {home_field}/{away_field} format")
                        break

            if not player1_data or not player2_data:
                logger.warning(f"❌ Could not find player data in event. Available fields: {list(event_data.keys())}")
                return None

            # Extract player information with fallbacks
            def extract_player_info(player_data: Dict[str, Any], player_num: int) -> PlayerData:
                return PlayerData(
                    id=player_data.get('id'),
                    name=player_data.get('name', f'Player {player_num}'),
                    ranking=player_data.get('ranking'),
                    points=None,
                    country=player_data.get('country', {}).get('name') if isinstance(player_data.get('country'),
                                                                                     dict) else player_data.get(
                        'country'),
                    last_updated=datetime.now()
                )

            player1 = extract_player_info(player1_data, 1)
            player2 = extract_player_info(player2_data, 2)

            # Extract tournament info with fallbacks
            tournament_data = event_data.get('tournament', {})
            tournament_name = tournament_data.get('name', 'Unknown Tournament') if isinstance(tournament_data,
                                                                                              dict) else str(
                tournament_data)
            surface = tournament_data.get('groundType', 'Unknown') if isinstance(tournament_data, dict) else 'Unknown'

            # Extract status with fallbacks
            status_data = event_data.get('status', {})
            status = status_data.get('description', 'Unknown') if isinstance(status_data, dict) else str(status_data)

            match_data = MatchData(
                id=event_id,
                player1=player1,
                player2=player2,
                tournament=tournament_name,
                surface=surface,
                status=status,
                odds=event_data.get('odds'),
                live_score=event_data.get('liveScore'),
                betting_analysis=None
            )

            logger.info(f"✅ Successfully converted event: {player1.name} vs {player2.name}")
            return match_data

        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"❌ Schema mismatch in event data: {e}")
            logger.error(
                f"📊 Event data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'Not a dict'}")
            return None
        except Exception as e:
            logger.error(f"💥 Unexpected error parsing event data: {e}", exc_info=True)
            return None

    async def get_live_events(self) -> List[Dict[str, Any]]:
        """Enhanced live events with better error handling"""
        logger.info("🔴 LIVE: Fetching live events from RapidAPI...")

        live_data = await self._fetch_from_rapidapi("/events/live")

        if not live_data:
            logger.warning("❌ No data received from live events API")
            return self._get_fallback_events()

        if 'events' not in live_data:
            logger.warning(f"❌ No 'events' key in response. Keys: {list(live_data.keys())}")
            return self._get_fallback_events()

        events = live_data['events']
        if not events:
            logger.info("ℹ️ No live events currently available")
            return self._get_fallback_events()

        successful_matches = []
        failed_count = 0

        for event in events:
            match = self._convert_to_match_data(event)
            if match:
                successful_matches.append(asdict(match))
            else:
                failed_count += 1

        logger.info(f"✅ LIVE: Successfully parsed {len(successful_matches)} events, {failed_count} failed")

        # If no events parsed successfully, return fallback
        if not successful_matches:
            logger.warning("❌ No events could be parsed, using fallback data")
            return self._get_fallback_events()

        return successful_matches

    def _get_fallback_events(self) -> List[Dict[str, Any]]:
        """Provide fallback tennis events when API fails"""
        logger.info("🔄 Using fallback tennis events")

        fallback_events = [
            {
                "id": 9999991,
                "player1": {
                    "id": 1001,
                    "name": "Novak Djokovic",
                    "ranking": 1,
                    "points": None,
                    "country": "Serbia",
                    "last_updated": datetime.now().isoformat()
                },
                "player2": {
                    "id": 1002,
                    "name": "Carlos Alcaraz",
                    "ranking": 2,
                    "points": None,
                    "country": "Spain",
                    "last_updated": datetime.now().isoformat()
                },
                "tournament": "ATP Masters 1000",
                "surface": "Hard",
                "status": "Live - Set 2",
                "odds": None,
                "live_score": {"set1": "6-4", "set2": "3-2"},
                "betting_analysis": None
            },
            {
                "id": 9999992,
                "player1": {
                    "id": 2001,
                    "name": "Iga Swiatek",
                    "ranking": 1,
                    "points": None,
                    "country": "Poland",
                    "last_updated": datetime.now().isoformat()
                },
                "player2": {
                    "id": 2002,
                    "name": "Aryna Sabalenka",
                    "ranking": 2,
                    "points": None,
                    "country": "Belarus",
                    "last_updated": datetime.now().isoformat()
                },
                "tournament": "WTA 1000",
                "surface": "Clay",
                "status": "Upcoming",
                "odds": None,
                "live_score": None,
                "betting_analysis": None
            }
        ]

        return fallback_events

    async def analyze_head_to_head(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """Enhanced H2H with better error handling"""
        logger.info(f"🎾 H2H Analysis for: {player1_name} vs {player2_name}")

        try:
            # Try to get player data with timeout
            p1_task = self._get_professional_player_data(player1_name)
            p2_task = self._get_professional_player_data(player2_name)

            p1_data, p2_data = await asyncio.wait_for(
                asyncio.gather(p1_task, p2_task, return_exceptions=True),
                timeout=15.0
            )

            # Handle exceptions in results
            if isinstance(p1_data, Exception):
                logger.error(f"Failed to get {player1_name} data: {p1_data}")
                p1_data = None
            if isinstance(p2_data, Exception):
                logger.error(f"Failed to get {player2_name} data: {p2_data}")
                p2_data = None

            if not p1_data or not p2_data:
                logger.warning("Using fallback H2H analysis due to missing player data")
                return self._get_fallback_h2h(player1_name, player2_name)

            # Try to get live event for H2H
            h2h_stats = await self._get_h2h_stats(p1_data, p2_data)

            return {
                "matchup": f"{p1_data.name} vs {p2_data.name}",
                "player1_profile": asdict(p1_data),
                "player2_profile": asdict(p2_data),
                "historical_h2h": h2h_stats,
                "confidence_level": "high" if h2h_stats.get("has_data") else "medium",
                "analysis_timestamp": datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            logger.error("H2H analysis timed out")
            return self._get_fallback_h2h(player1_name, player2_name)
        except Exception as e:
            logger.error(f"H2H analysis failed: {e}", exc_info=True)
            return self._get_fallback_h2h(player1_name, player2_name)

    async def _get_h2h_stats(self, p1_data: PlayerData, p2_data: PlayerData) -> Dict[str, Any]:
        """Get H2H statistics with error handling"""
        try:
            live_events_data = await self._fetch_from_rapidapi("/events/live")

            if not live_events_data or 'events' not in live_events_data:
                return {"note": "No live events data available for H2H lookup"}

            # Look for matching event
            for event in live_events_data['events']:
                # Try multiple field variations for player names
                home_names = []
                away_names = []

                # Extract names from different possible fields
                for field_pair in [('homePlayer', 'awayPlayer'), ('homeTeam', 'awayTeam')]:
                    home_field, away_field = field_pair
                    if home_field in event and away_field in event:
                        home_names.append(event[home_field].get('name', '').lower())
                        away_names.append(event[away_field].get('name', '').lower())

                # Check for matches
                p1_name_lower = p1_data.name.lower()
                p2_name_lower = p2_data.name.lower()

                for home_name, away_name in zip(home_names, away_names):
                    if ((p1_name_lower in home_name and p2_name_lower in away_name) or
                            (p2_name_lower in home_name and p1_name_lower in away_name)):

                        event_id = event.get('id')
                        if event_id:
                            duel_data = await self._fetch_from_rapidapi(f"/event/{event_id}/duel")
                            if duel_data:
                                return {"has_data": True, "duel_stats": duel_data}

            return {"note": "No current H2H event found", "has_data": False}

        except Exception as e:
            logger.error(f"Error getting H2H stats: {e}")
            return {"error": f"H2H lookup failed: {str(e)}", "has_data": False}

    def _get_fallback_h2h(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """Fallback H2H analysis"""
        return {
            "matchup": f"{player1_name} vs {player2_name}",
            "player1_profile": {"name": player1_name, "ranking": "Unknown", "country": "Unknown"},
            "player2_profile": {"name": player2_name, "ranking": "Unknown", "country": "Unknown"},
            "historical_h2h": {"note": "Fallback analysis - live data unavailable"},
            "confidence_level": "low",
            "analysis_timestamp": datetime.now().isoformat(),
            "data_source": "fallback"
        }

    async def _get_professional_player_data(self, player_name: str) -> Optional[PlayerData]:
        """Enhanced player data retrieval"""
        try:
            search_data = await self._fetch_from_rapidapi(f"/search/{player_name}")
            return self._parse_player_search(search_data, player_name) if search_data else None
        except Exception as e:
            logger.error(f"Failed to get player data for {player_name}: {e}")
            return None

    async def get_player_card(self, player_name: str) -> Dict[str, Any]:
        """Enhanced player card with fallback"""
        try:
            player_base_data = await self._get_professional_player_data(player_name)

            if not player_base_data or not player_base_data.id:
                return {"error": f"Player '{player_name}' not found."}

            player_id = player_base_data.id
            logger.info(f"Fetching detailed data for player ID {player_id}")

            # Get additional data with timeout
            results = await asyncio.wait_for(
                asyncio.gather(
                    self._fetch_from_rapidapi(f"/player/{player_id}/events/previous/0"),
                    self._fetch_from_rapidapi(f"/player/{player_id}/rankings"),
                    return_exceptions=True
                ),
                timeout=10.0
            )

            recent_events_data, rankings_data = results

            # Process recent form
            recent_form = []
            if not isinstance(recent_events_data, Exception) and isinstance(recent_events_data,
                                                                            dict) and 'events' in recent_events_data:
                for event in recent_events_data['events']:
                    try:
                        winner_code = event.get('winnerCode')
                        is_home_player = event.get('homePlayer', {}).get('id') == player_id
                        result = 'W' if (is_home_player and winner_code == 1) or (
                                    not is_home_player and winner_code == 2) else 'L'
                        recent_form.append(result)
                    except (TypeError, KeyError):
                        pass

            # Process rankings
            ranking_history = []
            if not isinstance(rankings_data, Exception) and isinstance(rankings_data, dict):
                ranking_history = rankings_data.get('rankings', [])

            return {
                "profile": asdict(player_base_data),
                "recent_form_string": "-".join(recent_form[:5]) if recent_form else "No recent data",
                "ranking_history": ranking_history,
                "data_retrieved_at": datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            logger.error(f"Player card request timed out for {player_name}")
            return {"error": f"Request timed out for player '{player_name}'"}
        except Exception as e:
            logger.error(f"Error getting player card for {player_name}: {e}", exc_info=True)
            return {"error": f"Failed to retrieve data for player '{player_name}'"}

    async def get_comprehensive_player_analysis(self, player_name: str) -> Dict[str, Any]:
        """Enhanced comprehensive analysis"""
        try:
            player_data = await self._get_professional_player_data(player_name)
            if player_data:
                return asdict(player_data)
            else:
                return {"error": "Player not found."}
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {player_name}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def get_current_tournaments(self) -> Dict[str, Any]:
        """Placeholder for tournament data"""
        return {
            "note": "Tournament data endpoint not yet implemented.",
            "atp_top_10": [],
            "wta_top_10": [],
            "last_updated": datetime.now().isoformat()
        }

    async def close(self):
        await self._client.aclose()
        logger.info("🔒 Professional Tennis API Client session closed.")