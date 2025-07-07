# backend/background_tasks.py
# MODIFIED: Now fetches all of today's events instead of just 'live' ones.

import asyncio
import logging
from datetime import datetime
from llm_interface.tennis_api_client import ProfessionalTennisAPIClient
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


async def monitor_live_matches(rag_pipeline=None, tennis_config=None):
    """
    A background task that runs periodically to fetch RECENT and LIVE matches
    and update the database with their results.

    This is now smarter: it fetches all of today's events, which includes
    scheduled, live, and recently finished matches. This increases the likelihood
    of finding completed matches with a winnerCode to save to the database.
    """
    logger.info("✅ Starting the background Tennis Match Monitor.")
    while True:
        logger.info("BACKGROUND_TASK: Waking up to check for new match data...")
        client = None
        db_manager = None

        try:
            client = ProfessionalTennisAPIClient()
            db_manager = DatabaseManager()

            # <<< THE FIX IS HERE! >>>
            # Instead of just 'live' events, we get all events for today.
            # This is more robust for capturing completed matches.
            today = datetime.now()
            logger.info(f"BACKGROUND_TASK: Fetching all events for date: {today.strftime('%Y-%m-%d')}")

            events_response = await client.get_events_by_date(today.day, today.month, today.year)

            if not events_response or not events_response.get('events'):
                logger.info("BACKGROUND_TASK: No events found for today.")
            else:
                all_todays_events = events_response['events']
                logger.info(f"BACKGROUND_TASK: Found {len(all_todays_events)} total events for today. Processing...")

                for event_data in all_todays_events:
                    # The db_manager will now process each event. It will naturally
                    # skip events that are not yet complete (i.e., no winnerCode),
                    # and successfully save the ones that are.
                    db_manager.process_match_data(event_data)

                logger.info(f"BACKGROUND_TASK: Finished processing batch of {len(all_todays_events)} events for today.")

        except Exception as e:
            logger.error(f"BACKGROUND_TASK: An unexpected error occurred: {e}", exc_info=True)

        finally:
            if client:
                await client.close()
            if db_manager:
                db_manager.close()

            # Sleep for 15 minutes, as we are now checking a whole day's worth of events.
            sleep_duration_seconds = 900
            logger.info(f"BACKGROUND_TASK: Sleeping for {sleep_duration_seconds / 60:.0f} minutes...")
            await asyncio.sleep(sleep_duration_seconds)