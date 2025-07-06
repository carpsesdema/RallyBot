# backend/background_tasks.py
# MODIFIED: Replaced ranking updater with a more powerful live match monitor.

import asyncio
import logging
from llm_interface.tennis_api_client import ProfessionalTennisAPIClient
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


async def monitor_live_matches(rag_pipeline=None, tennis_config=None):  # Added args to match server call
    """
    A background task that runs periodically to fetch live and recent matches
    and update the database with their results.
    """
    logger.info("✅ Starting the background Tennis Match Monitor.")
    while True:
        logger.info("BACKGROUND_TASK: Waking up to check for new match data...")
        client = None
        db_manager = None

        try:
            client = ProfessionalTennisAPIClient()
            db_manager = DatabaseManager()

            live_events = await client.get_live_events()

            if not live_events:
                logger.info("BACKGROUND_TASK: No live events found at this time.")
            else:
                logger.info(f"BACKGROUND_TASK: Found {len(live_events)} events to process.")

                for event_data in live_events:
                    db_manager.process_match_data(event_data)

                logger.info(f"BACKGROUND_TASK: Finished processing batch of {len(live_events)} events.")

        except Exception as e:
            logger.error(f"BACKGROUND_TASK: An unexpected error occurred: {e}", exc_info=True)

        finally:
            if client:
                await client.close()
            if db_manager:
                db_manager.close()

            logger.info("BACKGROUND_TASK: Sleeping for 10 minutes...")
            await asyncio.sleep(600)