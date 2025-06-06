# llm_interface/tennis_api_client.py - FIXED VERSION WITH NO ERRORS
import httpx
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import asyncio

try:
    from config import tennis_config, TennisAPIConfig
except ImportError:
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
        self.enable_detailed_data = True
        logger.info("🎾 Tennis API Client - FIXED VERSION")

    async def _fetch_from_rapidapi(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Dict[str, Any]]:
        if not self.config.credentials.rapidapi_key:
            logger.error("❌ RapidAPI key not configured")
            return None

        headers = {
            'X-RapidAPI-Key': self.config.credentials.rapidapi_key,
            'X-RapidAPI-Host': self.config.credentials.rapidapi_host
        }
        url = f"{self.config.endpoints.rapidapi_base}{endpoint}"

        try:
            logger.debug(f"📡 API Call: {endpoint}")
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"✅ Success: {endpoint}")
            return data
        except Exception as e:
            logger.error(f"❌ API failed [{endpoint}]: {e}")
            return None

    # --- Event Endpoints ---
    async def get_live_events(self, enhance_with_details: bool = False) -> List[Dict[str, Any]]:
        """Get live events"""
        logger.info("🔴 FETCHING LIVE EVENTS")

        live_data = await self._fetch_from_rapidapi("/events/live")

        if not live_data or 'events' not in live_data:
            return self._get_fallback_events()

        events = live_data['events']
        if not events:
            return self._get_fallback_events()

        successful_matches = []
        for event in events:
            match = self._convert_to_match_data(event)
            if match:
                match_dict = asdict(match)
                match_dict["enhancement_status"] = "basic"
                successful_matches.append(match_dict)

        return successful_matches if successful_matches else self._get_fallback_events()

    async def get_events_by_date(self, day: int, month: int, year: int) -> Optional[Dict[str, Any]]:
        """Get scheduled events for a specific date."""
        logger.info(f"🗓️ Events by Date: {day}/{month}/{year}")
        return await self._fetch_from_rapidapi(f"/events/{day}/{month}/{year}")

    async def get_odds_by_date(self, day: int, month: int, year: int) -> Optional[Dict[str, Any]]:
        """Get scheduled odds for a specific date."""
        logger.info(f"💰 Odds by Date: {day}/{month}/{year}")
        return await self._fetch_from_rapidapi(f"/events/odds/{day}/{month}/{year}")

    async def get_event_details(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get event details"""
        logger.info(f"📊 Event Details: {event_id}")
        return await self._fetch_from_rapidapi(f"/event/{event_id}")

    async def get_event_statistics(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get event statistics"""
        logger.info(f"📈 Event Statistics: {event_id}")
        return await self._fetch_from_rapidapi(f"/event/{event_id}/statistics")

    async def get_event_point_by_point(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get point-by-point data"""
        logger.info(f"🎯 Point-by-Point: {event_id}")
        return await self._fetch_from_rapidapi(f"/event/{event_id}/point-by-point")

    async def get_event_odds(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get event odds"""
        logger.info(f"💰 Event Odds: {event_id}")
        return await self._fetch_from_rapidapi(f"/event/{event_id}/odds")

    async def get_event_h2h(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get event H2H"""
        logger.info(f"🥊 Event H2H: {event_id}")
        return await self._fetch_from_rapidapi(f"/event/{event_id}/duel")

    # --- General & Search Endpoints ---
    async def search_tennis_entity(self, search_term: str) -> Optional[Dict[str, Any]]:
        """Search tennis entities"""
        logger.info(f"🔍 Search: {search_term}")
        return await self._fetch_from_rapidapi(f"/search/{search_term}")

    async def get_calendar_events(self, month: int, year: int) -> Optional[Dict[str, Any]]:
        """Get calendar events for a specific month and year."""
        logger.info(f"📅 Calendar Events: {month}/{year}")
        return await self._fetch_from_rapidapi(f"/calendar/{month}/{year}")

    # --- Player Endpoints ---
    async def get_player_details(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get player details"""
        logger.info(f"👤 Player Details: {player_id}")
        return await self._fetch_from_rapidapi(f"/player/{player_id}")

    async def get_player_previous_events(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get player previous events"""
        logger.info(f"📜 Previous Events: {player_id}")
        return await self._fetch_from_rapidapi(f"/player/{player_id}/events/previous/0")

    async def get_player_future_events(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get player future events"""
        logger.info(f"🔮 Future Events: {player_id}")
        return await self._fetch_from_rapidapi(f"/player/{player_id}/events/next/0")

    async def get_player_rankings_history(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get player rankings"""
        logger.info(f"📈 Player Rankings: {player_id}")
        return await self._fetch_from_rapidapi(f"/player/{player_id}/rankings")

    # --- Rankings Endpoints ---
    async def get_atp_rankings(self) -> Optional[Dict[str, Any]]:
        """Get ATP rankings"""
        logger.info("🏆 ATP Rankings")
        return await self._fetch_from_rapidapi("/rankings/atp/live")

    async def get_wta_rankings(self) -> Optional[Dict[str, Any]]:
        """Get WTA rankings"""
        logger.info("🏆 WTA Rankings")
        return await self._fetch_from_rapidapi("/rankings/wta/live")

    # --- Tournament Endpoints ---
    async def get_tournament_details(self, tournament_id: int) -> Optional[Dict[str, Any]]:
        """Get tournament details"""
        logger.info(f"🏟️ Tournament: {tournament_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}")

    async def get_tournament_info(self, tournament_id: int) -> Optional[Dict[str, Any]]:
        """Get general tournament info."""
        logger.info(f"ℹ️ Tournament Info: {tournament_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/info")

    async def get_tournament_seasons(self, tournament_id: int) -> Optional[Dict[str, Any]]:
        """Get seasons for a tournament."""
        logger.info(f"📅 Tournament Seasons: {tournament_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/seasons")

    async def get_tournament_rounds(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get rounds for a tournament season."""
        logger.info(f"🔄 Tournament Rounds: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/rounds")

    async def get_tournament_last_events(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get last events for a tournament season."""
        logger.info(f"⏪ Tournament Last Events: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/events/last/0")

    async def get_tournament_next_events(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get next events for a tournament season."""
        logger.info(f"⏩ Tournament Next Events: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/events/next/0")

    async def get_tournament_standings(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get total standings for a tournament season."""
        logger.info(f"📊 Tournament Standings: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/standings/total")

    async def get_tournament_cup_trees(self, tournament_id: int, season_id: int, old: bool = False) -> Optional[Dict[str, Any]]:
        """Get cup trees for a tournament season (current or old)."""
        tree_type = "/old" if old else ""
        logger.info(f"🌳 Tournament Cup Trees ({'Old' if old else 'Current'}): T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/cup-trees{tree_type}")

    async def get_tournament_season_info(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get info for a specific tournament season."""
        logger.info(f"ℹ️ Tournament Season Info: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/info")

    async def get_tournament_events_by_round(self, tournament_id: int, season_id: int, round_id: int, slug: str) -> Optional[Dict[str, Any]]:
        """Get tournament events by a specific round."""
        logger.info(f"🔍 Tournament Events by Round: T={tournament_id}, S={season_id}, R={round_id}, Slug={slug}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/events/round/{round_id}/slug/{slug}")

    # --- High-Level Analysis Methods ---
    async def enhance_event_with_live_data(self, event_id: int) -> Dict[str, Any]:
        """Enhanced event data"""
        logger.info(f"🔥 ENHANCING: {event_id}")

        try:
            tasks = [
                self.get_event_details(event_id),
                self.get_event_statistics(event_id),
                self.get_event_odds(event_id),
                self.get_event_h2h(event_id)
            ]

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0
            )

            event_details, statistics, odds_data, h2h_data = results

            enhanced_data = {
                "live_score": self._extract_live_score(event_details) if not isinstance(event_details,
                                                                                        Exception) else None,
                "detailed_stats": statistics if not isinstance(statistics, Exception) else None,
                "live_odds": odds_data if not isinstance(odds_data, Exception) else None,
                "h2h_data": h2h_data if not isinstance(h2h_data, Exception) else None,
                "enhancement_timestamp": datetime.now().isoformat()
            }

            success_count = sum(1 for result in results if not isinstance(result, Exception))
            logger.info(f"✅ Enhanced {event_id}: {success_count}/4 successful")

            return enhanced_data

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Enhancement timeout: {event_id}")
            return {"error": "Enhancement timeout", "event_id": event_id}
        except Exception as e:
            logger.error(f"💥 Enhancement failed: {e}")
            return {"error": str(e), "event_id": event_id}

    async def get_comprehensive_player_analysis(self, player_name: str) -> Dict[str, Any]:
        """Comprehensive player analysis"""
        try:
            search_data = await self.search_tennis_entity(player_name)
            player_data = self._parse_player_search(search_data, player_name) if search_data else None

            if not player_data or not player_data.id:
                return {"error": f"Player '{player_name}' not found"}

            player_id = player_data.id
            logger.info(f"📊 ANALYZING: {player_name} (ID: {player_id})")

            tasks = [
                self.get_player_details(player_id),
                self.get_player_previous_events(player_id),
                self.get_player_future_events(player_id),
                self.get_player_rankings_history(player_id)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            player_details, previous_events, future_events, rankings = results

            recent_form = []
            if not isinstance(previous_events, Exception) and previous_events and 'events' in previous_events:
                for event in previous_events['events'][:10]:
                    try:
                        winner_code = event.get('winnerCode')
                        is_home_player = event.get('homePlayer', {}).get('id') == player_id
                        result = 'W' if (is_home_player and winner_code == 1) or (
                                    not is_home_player and winner_code == 2) else 'L'
                        recent_form.append(result)
                    except (TypeError, KeyError):
                        pass

            return {
                "profile": asdict(player_data),
                "player_details": player_details if not isinstance(player_details, Exception) else None,
                "recent_events": previous_events if not isinstance(previous_events, Exception) else None,
                "future_events": future_events if not isinstance(future_events, Exception) else None,
                "rankings_history": rankings if not isinstance(rankings, Exception) else None,
                "recent_form_string": "-".join(recent_form) if recent_form else "No data",
                "recent_form_array": recent_form,
                "betting_profile": self._generate_betting_profile(recent_form),
                "data_retrieved_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Analysis failed for {player_name}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def analyze_head_to_head(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """H2H analysis"""
        logger.info(f"🎾 H2H: {player1_name} vs {player2_name}")

        try:
            p1_search = await self.search_tennis_entity(player1_name)
            p2_search = await self.search_tennis_entity(player2_name)

            p1_data = self._parse_player_search(p1_search, player1_name) if p1_search else None
            p2_data = self._parse_player_search(p2_search, player2_name) if p2_search else None

            if not p1_data or not p2_data:
                return self._get_fallback_h2h(player1_name, player2_name)

            return {
                "matchup": f"{p1_data.name} vs {p2_data.name}",
                "player1_profile": asdict(p1_data),
                "player2_profile": asdict(p2_data),
                "historical_h2h": {"note": "H2H data available via event analysis"},
                "confidence_level": "medium",
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"H2H failed: {e}")
            return self._get_fallback_h2h(player1_name, player2_name)

    def _convert_to_match_data(self, event_data: Dict[str, Any]) -> Optional[MatchData]:
        """Convert event data - FIXED for player1/player2 schema"""
        try:
            event_id = event_data.get('id')
            if not event_id:
                return None

            player1_data = event_data.get('player1')
            player2_data = event_data.get('player2')

            if not player1_data or not player2_data:
                logger.warning(f"Missing player data in event {event_id}")
                return None

            player1 = PlayerData(
                id=player1_data.get('id'),
                name=player1_data.get('name', 'Player 1'),
                ranking=player1_data.get('ranking'),
                points=None,
                country=player1_data.get('country'),
                last_updated=datetime.now()
            )

            player2 = PlayerData(
                id=player2_data.get('id'),
                name=player2_data.get('name', 'Player 2'),
                ranking=player2_data.get('ranking'),
                points=None,
                country=player2_data.get('country'),
                last_updated=datetime.now()
            )

            return MatchData(
                id=event_id,
                player1=player1,
                player2=player2,
                tournament=event_data.get('tournament', 'Unknown'),
                surface=event_data.get('surface', 'Unknown'),
                status=event_data.get('status', 'Unknown'),
                odds=event_data.get('odds'),
                live_score=event_data.get('live_score'),
                betting_analysis=None
            )

        except Exception as e:
            logger.error(f"Event conversion failed: {e}")
            return None

    def _extract_live_score(self, event_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract live score"""
        if not event_details:
            return None

        for field in ["score", "liveScore", "currentScore"]:
            if field in event_details:
                return event_details[field]

        return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[PlayerData]:
        """Parse player search"""
        if not data or not data.get('results'):
            return None

        for result in data['results']:
            if result.get('type') == 'player':
                entity = result.get('entity', {})
                if player_name.lower() in entity.get('name', '').lower():
                    return PlayerData(
                        id=entity.get('id'),
                        name=entity.get('name'),
                        ranking=entity.get('ranking'),
                        country=entity.get('country', {}).get('name') if isinstance(entity.get('country'),
                                                                                    dict) else entity.get('country'),
                        points=entity.get('rankingPoints'),
                        last_updated=datetime.now()
                    )
        return None

    def _generate_betting_profile(self, recent_form: List[str]) -> Dict[str, Any]:
        """Generate betting profile"""
        if not recent_form:
            return {"tier": "unknown", "form_percentage": 0}

        wins = recent_form.count('W')
        total = len(recent_form)
        form_percentage = (wins / total) * 100

        return {
            "betting_tier": "premium" if form_percentage > 70 else "standard" if form_percentage > 50 else "value",
            "form_percentage": round(form_percentage, 1),
            "wins": wins,
            "losses": total - wins
        }

    def _get_fallback_events(self) -> List[Dict[str, Any]]:
        """Fallback events"""
        return [
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
                "betting_analysis": None,
                "enhancement_status": "fallback"
            }
        ]

    def _get_fallback_h2h(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """Fallback H2H"""
        return {
            "matchup": f"{player1_name} vs {player2_name}",
            "player1_profile": {"name": player1_name},
            "player2_profile": {"name": player2_name},
            "historical_h2h": {"note": "Fallback H2H"},
            "confidence_level": "low",
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def get_player_card(self, player_name: str) -> Dict[str, Any]:
        """Get player card"""
        return await self.get_comprehensive_player_analysis(player_name)

    async def get_current_tournaments(self) -> Dict[str, Any]:
        """Get current tournaments"""
        try:
            atp_data, wta_data = await asyncio.gather(
                self.get_atp_rankings(),
                self.get_wta_rankings(),
                return_exceptions=True
            )

            return {
                "atp_top_10": atp_data.get("rankings", [])[:10] if not isinstance(atp_data,
                                                                                  Exception) and atp_data else [],
                "wta_top_10": wta_data.get("rankings", [])[:10] if not isinstance(wta_data,
                                                                                  Exception) and wta_data else [],
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "atp_top_10": [],
                "wta_top_10": [],
                "error": str(e)
            }

    async def close(self):
        """Close client"""
        await self._client.aclose()
        logger.info("🔒 Tennis API Client closed")