# backend/background_tasks.py
# FINAL REDESIGN - This task no longer touches the database directly.

import asyncio
import logging
from datetime import datetime
from llm_interface.tennis_api_client import ProfessionalTennisAPIClient
# DO NOT import DatabaseManager here. This module will no longer interact with the DB.

logger = logging.getLogger(__name__)


async def monitor_live_matches(app_state):
    """
    A background task that runs periodically to fetch RECENT and LIVE matches
    and puts them into the central database queue for the writer process.
    IT NO LONGER WRITES TO THE DATABASE ITSELF.
    """
    logger.info("✅ Starting the background Tennis Match Monitor (Queue Mode).")
    # Get the queue from the application state passed from the lifespan manager.
    db_write_queue = app_state.db_write_queue

    while True:
        logger.info("BACKGROUND_TASK: Waking up to check for new match data...")
        client = None

        try:
            client = ProfessionalTennisAPIClient()

            today = datetime.now()
            logger.info(f"BACKGROUND_TASK: Fetching all events for date: {today.strftime('%Y-%m-%d')}")
            events_response = await client.get_events_by_date(today.day, today.month, today.year)

            if not events_response or not events_response.get('events'):
                logger.info("BACKGROUND_TASK: No events found for today.")
            else:
                all_todays_events = events_response['events']
                logger.info(f"BACKGROUND_TASK: Found {len(all_todays_events)} total events for today. Processing...")

                for event_data in all_todays_events:
                    # The ONLY thing this task does now is put the data in the queue.
                    # This is a non-blocking operation.
                    # It relies on the same filtering logic that the backfill client uses:
                    # it sends all events, and the db_manager will filter out the incomplete ones.
                    await db_write_queue.put(event_data)
                    logger.info(f"BACKGROUND_TASK: Queued event ID {event_data.get('id')} for processing.")

        except Exception as e:
            logger.error(f"BACKGROUND_TASK: An unexpected error occurred: {e}", exc_info=True)

        finally:
            if client:
                await client.close()

            # Sleep for 15 minutes.
            sleep_duration_seconds = 900
            logger.info(f"BACKGROUND_TASK: Sleeping for {sleep_duration_seconds / 60:.0f} minutes...")
            await asyncio.sleep(sleep_duration_seconds)