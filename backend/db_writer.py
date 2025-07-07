# backend/db_writer.py
import logging
from typing import Dict, Any
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

def write_match_to_db(match_data: Dict[str, Any]):
    """
    This function is designed to be run in the background.
    It opens its own database connection, writes the data, and closes it.
    """
    db_manager = None
    match_id = match_data.get('id', 'Unknown')
    logger.info(f"BACKGROUND_SAVE: Starting to save match ID: {match_id}")
    try:
        db_manager = DatabaseManager()
        db_manager.process_match_data(match_data)
        # The success log is now inside process_match_data
    except Exception as e:
        # This log will now appear in the server logs if the background task fails.
        logger.error(f"BACKGROUND_SAVE: ❌ Failed to save match ID {match_id}. Error: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()
        logger.info(f"BACKGROUND_SAVE: Finished task for match ID: {match_id}")