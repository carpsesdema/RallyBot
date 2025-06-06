# database/db_manager.py
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path="tennis_intelligence.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    async def update_player_from_ranking(self, ranking_data: dict):
        """
        Updates or inserts a player's details and their latest ranking.
        """
        cursor = self.conn.cursor()
        try:
            player_info = ranking_data.get("player", {})
            player_id = player_info.get("id")
            player_name = player_info.get("name")

            if not player_id or not player_name:
                return

            # Upsert into players table
            cursor.execute("""
                INSERT INTO players (api_player_id, name, country_code, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(api_player_id) DO UPDATE SET
                    name = excluded.name,
                    country_code = excluded.country_code,
                    updated_at = excluded.updated_at;
            """, (player_id, player_name, player_info.get("country", {}).get("alpha2"), datetime.now().isoformat()))

            # Upsert into player_rankings table
            ranking = ranking_data.get("ranking")
            points = ranking_data.get("points")
            ranking_date = datetime.now().date().isoformat()

            cursor.execute("""
                INSERT INTO player_rankings (player_id, ranking_date, atp_ranking, wta_ranking, ranking_points)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id, ranking_date) DO UPDATE SET
                    atp_ranking = excluded.atp_ranking,
                    wta_ranking = excluded.wta_ranking,
                    ranking_points = excluded.ranking_points;
            """, (player_id, ranking_date, ranking, None, points))  # Assuming ATP for now

            self.conn.commit()
            logger.debug(f"Successfully updated player ID {player_id} ({player_name}) in database.")

        except Exception as e:
            logger.error(f"Failed to update player {ranking_data.get('player', {}).get('name')} in DB: {e}")
            self.conn.rollback()

    def close(self):
        self.conn.close()