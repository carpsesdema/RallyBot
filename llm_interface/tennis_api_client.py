# llm_interface/tennis_api_client.py
import httpx
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class TennisAPIClient:
    def __init__(self):
        self.base_url = "https://tennisapi1.p.rapidapi.com/api/tennis"
        self.headers = {
            'x-rapidapi-host': 'tennisapi1.p.rapidapi.com',
            'x-rapidapi-key': '989746336dmshb828c110eabf0acp1e5cf5jsn00b9a7dc14e1'
        }
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("TennisAPIClient initialized")

    async def get_live_events(self) -> List[Dict[str, Any]]:
        """Get all live tennis events"""
        try:
            # Note: Using the EdgeAI endpoint from the PDF
            response = await self._client.get(
                "https://api.edgeai.pro/api/tennis/events/live",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Live events API returned {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching live events: {e}")
            return []

    async def search_player(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Search for a player by name"""
        try:
            url = f"{self.base_url}/search/{player_name}"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error searching player {player_name}: {e}")
            return None

    async def get_event_h2h(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get head-to-head data for an event"""
        try:
            url = f"{self.base_url}/event/{event_id}/duel"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching H2H for event {event_id}: {e}")
            return None

    async def get_event_odds(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get betting odds for an event"""
        try:
            url = f"{self.base_url}/event/{event_id}/odds"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching odds for event {event_id}: {e}")
            return None

    async def get_event_statistics(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed match statistics"""
        try:
            url = f"{self.base_url}/event/{event_id}/statistics"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching stats for event {event_id}: {e}")
            return None

    async def get_player_details(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed player information"""
        try:
            url = f"{self.base_url}/player/{player_id}"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching player details for {player_id}: {e}")
            return None

    async def get_player_recent_matches(self, player_id: str, page: int = 0) -> Optional[Dict[str, Any]]:
        """Get player's recent match history"""
        try:
            url = f"{self.base_url}/player/{player_id}/events/previous/{page}"
            response = await self._client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching recent matches for player {player_id}: {e}")
            return None

    async def analyze_matchup(self, player1_name: str, player2_name: str) -> Dict[str, Any]:
        """
        Comprehensive matchup analysis - THIS IS THE MONEY SHOT
        """
        analysis = {
            "players": {"player1": player1_name, "player2": player2_name},
            "h2h_record": None,
            "recent_form": {},
            "odds": None,
            "surface_stats": {},
            "betting_insights": []
        }

        try:
            # Search for both players
            player1_data = await self.search_player(player1_name)
            player2_data = await self.search_player(player2_name)

            if not player1_data or not player2_data:
                logger.warning(f"Could not find data for {player1_name} or {player2_name}")
                return analysis

            # Try to find a current/recent match between them
            live_events = await self.get_live_events()

            # Look for current match
            current_match = None
            for event in live_events:
                # This logic depends on the actual API response structure
                # We'll need to adjust based on the real data format
                if (player1_name.lower() in str(event).lower() and
                        player2_name.lower() in str(event).lower()):
                    current_match = event
                    break

            if current_match and current_match.get('id'):
                event_id = current_match['id']

                # Get H2H data
                h2h_data = await self.get_event_h2h(event_id)
                if h2h_data:
                    analysis["h2h_record"] = h2h_data

                # Get odds
                odds_data = await self.get_event_odds(event_id)
                if odds_data:
                    analysis["odds"] = odds_data

                # Get match statistics
                stats_data = await self.get_event_statistics(event_id)
                if stats_data:
                    analysis["surface_stats"] = stats_data

            # Generate betting insights based on available data
            analysis["betting_insights"] = self._generate_betting_insights(analysis)

            logger.info(f"Completed matchup analysis for {player1_name} vs {player2_name}")
            return analysis

        except Exception as e:
            logger.error(f"Error in matchup analysis: {e}")
            return analysis

    def _generate_betting_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate betting recommendations from analysis data"""
        insights = []

        # This is where we'd add intelligent analysis
        # For now, basic logic based on available data

        if analysis.get("h2h_record"):
            insights.append("H2H analysis available - check historical matchups")

        if analysis.get("odds"):
            insights.append("Live odds detected - monitor line movement")

        if analysis.get("surface_stats"):
            insights.append("Surface-specific performance data available")

        return insights

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()
        logger.info("TennisAPIClient session closed")


# Quick test function
async def test_tennis_api():
    """Test the tennis API client"""
    client = TennisAPIClient()

    try:
        # Test live events
        live_events = await client.get_live_events()
        print(f"Found {len(live_events)} live events")

        # Test player search
        federer_data = await client.search_player("federer")
        print(f"Federer search result: {federer_data is not None}")

        # Test matchup analysis
        analysis = await client.analyze_matchup("djokovic", "alcaraz")
        print(f"Matchup analysis completed: {len(analysis['betting_insights'])} insights")

    except Exception as e:
        print(f"Test error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_tennis_api())