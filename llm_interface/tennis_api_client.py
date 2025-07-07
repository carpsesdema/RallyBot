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
        """
        Get live events from the direct EdgeAI endpoint. This now passes the raw
        event data through to ensure no fields are lost, aligning with the
        behavior of the backfill client.
        """
        logger.info("🔴 FETCHING LIVE EVENTS from EdgeAI endpoint")
        url = "https://api.edgeai.pro/api/tennis/events/live"
        live_data: Optional[Dict[str, Any]] = None

        try:
            logger.debug(f"📡 Direct API Call: {url}")
            response = await self._client.get(url)
            response.raise_for_status()
            live_data = response.json()
            logger.debug(f"✅ Success: {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ API HTTP Error [{url}]: {e.response.status_code} - {e.response.text}")
            return self._get_fallback_events()
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"❌ API failed [{url}]: {e}")
            return self._get_fallback_events()
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while fetching live events: {e}", exc_info=True)
            return self._get_fallback_events()

        if not live_data or 'events' not in live_data:
            logger.warning("Live events data is empty or malformed. Using fallback.")
            return self._get_fallback_events()

        events = live_data.get('events', [])
        if not events:
            logger.info("No live events currently available. Using fallback.")
            return self._get_fallback_events()

        # <<< THE FIX IS HERE! >>>
        # We now return the raw event list directly from the API.
        # This prevents data loss (e.g., winnerCode) that was happening during
        # the old dataclass conversion. The db_manager is now robust enough
        # to handle the different player key schemas (homePlayer vs player1).
        return events

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

    async def get_tournament_cup_trees(self, tournament_id: int, season_id: int, old: bool = False) -> Optional[
        Dict[str, Any]]:
        """Get cup trees for a tournament season (current or old)."""
        tree_type = "/old" if old else ""
        logger.info(f"🌳 Tournament Cup Trees ({'Old' if old else 'Current'}): T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/cup-trees{tree_type}")

    async def get_tournament_season_info(self, tournament_id: int, season_id: int) -> Optional[Dict[str, Any]]:
        """Get info for a specific tournament season."""
        logger.info(f"ℹ️ Tournament Season Info: T={tournament_id}, S={season_id}")
        return await self._fetch_from_rapidapi(f"/tournament/{tournament_id}/season/{season_id}/info")

    async def get_tournament_events_by_round(self, tournament_id: int, season_id: int, round_id: int, slug: str) -> \
    Optional[Dict[str, Any]]:
        """Get tournament events by a specific round."""
        logger.info(f"🔍 Tournament Events by Round: T={tournament_id}, S={season_id}, R={round_id}, Slug={slug}")
        return await self._fetch_from_rapidapi(
            f"/tournament/{tournament_id}/season/{season_id}/events/round/{round_id}/slug/{slug}")

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
            player_data_search_result = self._parse_player_search(search_data, player_name) if search_data else None

            if not player_data_search_result or not player_data_search_result.get('id'):
                logger.error(f"Could not find player '{player_name}' after intelligent search.")
                return {"error": f"Player '{player_name}' not found"}

            player_id = player_data_search_result['id']
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
                        home_player_info = event.get('homeTeam') or event.get('homePlayer')
                        is_home_player = home_player_info.get('id') == player_id if home_player_info else False

                        result = 'W' if (is_home_player and winner_code == 1) or (
                                not is_home_player and winner_code == 2) else 'L'
                        recent_form.append(result)
                    except (TypeError, KeyError):
                        pass

            return {
                "profile": player_data_search_result,
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
            logger.error(f"Analysis failed for {player_name}: {e}", exc_info=True)
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
                "matchup": f"{p1_data.get('name')} vs {p2_data.get('name')}",
                "player1_profile": p1_data,
                "player2_profile": p2_data,
                "historical_h2h": {"note": "H2H data available via event analysis"},
                "confidence_level": "medium",
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"H2H failed: {e}", exc_info=True)
            return self._get_fallback_h2h(player1_name, player2_name)

    def _extract_live_score(self, event_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract live score"""
        if not event_details:
            return None

        for field in ["score", "liveScore", "currentScore"]:
            if field in event_details:
                return event_details[field]

        return None

    def _parse_player_search(self, data: Dict[str, Any], player_name: str) -> Optional[Dict[str, Any]]:
        """
        Intelligently parse player search results.
        Finds all matching players and returns the one with the highest ranking.
        """
        if not data or not data.get('results'):
            logger.debug(f"No search results for '{player_name}'")
            return None

        found_players = []
        for result in data['results']:
            if result.get('type') == 'player':
                entity = result.get('entity', {})
                entity_name = entity.get('name', '').lower()

                if player_name.lower() in entity_name:
                    player_dict = {
                        'id': entity.get('id'),
                        'name': entity.get('name'),
                        'ranking': entity.get('ranking'),
                        'country': entity.get('country', {}).get('name') if isinstance(entity.get('country'),
                                                                                    dict) else entity.get('country'),
                        'points': entity.get('rankingPoints'),
                        'last_updated': datetime.now().isoformat()
                    }
                    found_players.append(player_dict)

        if not found_players:
            logger.warning(f"No player entities found in search results for '{player_name}'.")
            return None

        found_players.sort(key=lambda p: p.get('ranking') if p.get('ranking') is not None else 9999)

        best_match = found_players[0]
        logger.info(
            f"Found {len(found_players)} potential matches for '{player_name}'. Selected best match: {best_match.get('name')} (Ranking: {best_match.get('ranking')})")

        return best_match

    def _generate_betting_profile(self, recent_form: List[str]) -> Dict[str, Any]:
        """Generate betting profile"""
        if not recent_form:
            return {"tier": "unknown", "form_percentage": 0}

        wins = recent_form.count('W')
        total = len(recent_form)
        form_percentage = (wins / total) * 100 if total > 0 else 0

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
                "winnerCode": 1,
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