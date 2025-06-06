# backend/background_tasks.py
import asyncio
import logging
from llm_interface.tennis_api_client import ProfessionalTennisAPIClient
from database.db_manager import DatabaseManager  # We will create this next

logger = logging.getLogger(__name__)


async def update_top_player_rankings():
    """
    A background task that runs periodically to update rankings for top players.
    """
    while True:
        logger.info("BACKGROUND_TASK: Starting top player ranking update...")
        client = None
        db_manager = None
        try:
            client = ProfessionalTennisAPIClient()
            db_manager = DatabaseManager()

            # Fetch live rankings for ATP and WTA using the public methods
            atp_rankings = await client.get_atp_rankings()
            wta_rankings = await client.get_wta_rankings()

            top_players = []
            if atp_rankings and "rankings" in atp_rankings:
                top_players.extend(atp_rankings["rankings"][:20])  # Top 20 ATP
            if wta_rankings and "rankings" in wta_rankings:
                top_players.extend(wta_rankings["rankings"][:20])  # Top 20 WTA

            if not top_players:
                logger.warning("BACKGROUND_TASK: Could not fetch top player rankings from API.")
                # We should still wait before retrying, so the sleep is in the finally block
            else:
                logger.info(f"BACKGROUND_TASK: Fetched {len(top_players)} top players to update.")
                # Update database for each player
                for player_ranking_data in top_players:
                    await db_manager.update_player_from_ranking(player_ranking_data)
                logger.info("BACKGROUND_TASK: Top player ranking update complete.")

        except Exception as e:
            logger.error(f"BACKGROUND_TASK: Error during player ranking update: {e}", exc_info=True)
        finally:
            if client:
                await client.close()
            if db_manager:
                db_manager.close()

            # Wait for 1 hour before running again, even if there was an error
            logger.info("BACKGROUND_TASK: Sleeping for 1 hour before next ranking update.")
            await asyncio.sleep(3600)