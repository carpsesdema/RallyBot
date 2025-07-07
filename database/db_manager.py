import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Handles database connections and data processing.
    Now includes logic to initialize the schema if the database is new.
    """

    def __init__(self, db_path: str = "/data/tennis_intelligence.db"):
        """
        Initializes the DatabaseManager, connecting to the DB and ensuring schema exists.
        """
        self.db_path = db_path
        self.conn = None
        try:
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")

            self._ensure_schema()  # Call the schema check on initialization

            logger.info(f"Database connection established to {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise

    def _ensure_schema(self):
        """Checks if the 'players' table exists, and if not, runs the entire schema setup."""
        if not self.conn:
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players';")
        if cursor.fetchone():
            logger.info("Database schema already exists. Skipping creation.")
            return

        logger.warning("Database tables not found. Initializing schema now...")
        try:
            # Schema file is located relative to this file's project structure
            schema_path = Path(__file__).parent / "tennis_schema.sql"
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            cursor.executescript(schema_sql)
            self.conn.commit()
            logger.info("✅ Successfully created database schema.")
        except Exception as e:
            logger.critical(f"❌ CRITICAL: Failed to create database schema: {e}", exc_info=True)
            self.conn.rollback()
            raise

    def _upsert_player(self, cursor: sqlite3.Cursor, player_data: Dict[str, Any]) -> Optional[int]:
        """Inserts or updates a player's details."""
        api_player_id = player_data.get("id")
        player_name = player_data.get("name")
        country_code = player_data.get("country", {}).get("alpha2")

        if not api_player_id or not player_name:
            return None

        cursor.execute("""
            INSERT INTO players (api_player_id, name, country_code, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(api_player_id) DO UPDATE SET
                name=excluded.name,
                country_code=excluded.country_code,
                updated_at=excluded.updated_at;
        """, (api_player_id, player_name, country_code, datetime.now().isoformat()))

        cursor.execute("SELECT id FROM players WHERE api_player_id = ?", (api_player_id,))
        player_row = cursor.fetchone()
        return player_row['id'] if player_row else None

    def _insert_match(self, cursor: sqlite3.Cursor, match_data: Dict[str, Any], player1_id: int, player2_id: int, winner_id: int):
        """Inserts or updates a match record."""
        api_match_id = match_data.get('id')
        tournament = match_data.get('tournament', {})
        round_info = match_data.get('roundInfo', {})
        home_score = match_data.get('homeScore', {})
        away_score = match_data.get('awayScore', {})
        match_timestamp = match_data.get('startTimestamp')
        match_date = datetime.fromtimestamp(match_timestamp).isoformat() if match_timestamp else datetime.now().isoformat()
        score = f"P1: {home_score.get('display')} - P2: {away_score.get('display')}"

        cursor.execute("""
            INSERT INTO matches (api_match_id, player1_id, player2_id, winner_id, tournament_name, round_name, match_date, score_summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(api_match_id) DO UPDATE SET
                player1_id=excluded.player1_id,
                player2_id=excluded.player2_id,
                winner_id=excluded.winner_id,
                tournament_name=excluded.tournament_name,
                round_name=excluded.round_name,
                match_date=excluded.match_date,
                score_summary=excluded.score_summary;
        """, (api_match_id, player1_id, player2_id, winner_id, tournament.get('name'), round_info.get('round'), match_date, score, datetime.now().isoformat()))

    def _update_head_to_head(self, cursor: sqlite3.Cursor, winner_id: int, loser_id: int):
        """Updates head-to-head statistics."""
        player1_id, player2_id = min(winner_id, loser_id), max(winner_id, loser_id)
        player1_wins_increment = 1 if winner_id == player1_id else 0

        cursor.execute("""
            INSERT INTO head_to_head (player1_id, player2_id, total_matches, player1_wins, last_updated, last_match_winner_id, last_match_date)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(player1_id, player2_id) DO UPDATE SET
                total_matches = total_matches + 1,
                player1_wins = player1_wins + excluded.player1_wins,
                last_updated = excluded.last_updated,
                last_match_winner_id = excluded.last_match_winner_id,
                last_match_date = excluded.last_match_date;
        """, (player1_id, player2_id, player1_wins_increment, datetime.now().isoformat(), winner_id, datetime.now().date().isoformat()))

    def process_match_data(self, match_data: Dict[str, Any]):
        """Processes comprehensive match data, updating players, matches, and H2H stats."""
        if not self.conn:
            logger.error("Cannot process match data, no database connection.")
            return

        home_player_data = match_data.get("homePlayer") or match_data.get("player1")
        away_player_data = match_data.get("awayPlayer") or match_data.get("player2")
        winner_code = match_data.get("winnerCode")

        if not all([home_player_data, away_player_data, winner_code is not None]):
            logger.warning(f"Skipping match ID {match_data.get('id')} due to incomplete player or winner data.")
            return

        cursor = self.conn.cursor()
        try:
            home_player_id = self._upsert_player(cursor, home_player_data)
            away_player_id = self._upsert_player(cursor, away_player_data)

            if not home_player_id or not away_player_id:
                raise sqlite3.Error("Failed to upsert one or both players.")

            winner_id = home_player_id if winner_code == 1 else away_player_id
            loser_id = away_player_id if winner_code == 1 else home_player_id

            self._insert_match(cursor, match_data, home_player_id, away_player_id, winner_id)
            self._update_head_to_head(cursor, winner_id, loser_id)

            self.conn.commit()
            logger.info(f"✅ DB SUCCESS: Processed and saved match data for ID {match_data.get('id')}.")

        except sqlite3.Error as e:
            logger.error(f"DB ERROR processing match ID {match_data.get('id')}: {e}", exc_info=True)
            self.conn.rollback()

    def close(self):
        """Closes the database connection if it exists."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")