# database/db_manager.py
"""
Manages all database operations for the Tennis Intelligence application.

This module provides the DatabaseManager class, which handles connections to the
SQLite database and provides methods for inserting and updating player, match,
and head-to-head data. It is designed to process comprehensive match details
from the API.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles database connections and data processing."""

    def __init__(self, db_path: str = "database/tennis_intelligence.db"):
        """Initializes the DatabaseManager, connecting to the DB.

        Args:
            db_path (str): The file path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Enable foreign key support
            self.conn.execute("PRAGMA foreign_keys = ON;")
            logger.info(f"Database connection established to {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise

    def _upsert_player(self, cursor: sqlite3.Cursor, player_data: Dict[str, Any]) -> Optional[int]:
        """
        Inserts or updates a player's details and returns their internal ID.
        (This helper method was already correct)
        """
        api_player_id = player_data.get("id")
        player_name = player_data.get("name")
        country_code = player_data.get("country", {}).get("alpha2")

        if not api_player_id or not player_name:
            logger.warning("Skipping player upsert due to missing player id or name.")
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

        if not player_row:
            logger.error(f"Failed to retrieve internal id for api_player_id {api_player_id}.")
            return None

        return player_row['id']

    # <<< FIX: Method signature updated to accept player IDs correctly.
    def _insert_match(self, cursor: sqlite3.Cursor, match_data: Dict[str, Any], player1_id: int, player2_id: int,
                      winner_id: int) -> None:
        """
        Inserts or updates a match record in the database.
        """
        api_match_id = match_data.get('id')
        if not api_match_id:
            logger.warning("Skipping match insert due to missing match id.")
            return

        tournament = match_data.get('tournament', {})
        round_info = match_data.get('roundInfo', {})
        home_score = match_data.get('homeScore', {})
        away_score = match_data.get('awayScore', {})

        match_timestamp = match_data.get('startTimestamp')
        match_date = datetime.fromtimestamp(
            match_timestamp).isoformat() if match_timestamp else datetime.now().isoformat()

        score = f"P1: {home_score.get('display')} - P2: {away_score.get('display')}"

        # <<< FIX: SQL statement now uses correct column names from schema (player1_id, player2_id, round_name)
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
        """, (
            api_match_id, player1_id, player2_id, winner_id,
            tournament.get('name'), round_info.get('round'), match_date, score,
            datetime.now().isoformat()
        ))

    def _update_head_to_head(self, cursor: sqlite3.Cursor, winner_id: int, loser_id: int) -> None:
        """
        Updates the head-to-head statistics for the two players involved in a match.
        """
        player1_id = min(winner_id, loser_id)
        player2_id = max(winner_id, loser_id)

        wins1_increment = 1 if winner_id == player1_id else 0
        wins2_increment = 1 if winner_id == player2_id else 0

        # <<< FIX: SQL statement now uses correct column names (player1_wins, player2_wins)
        cursor.execute("""
            INSERT INTO head_to_head (player1_id, player2_id, player1_wins, player2_wins, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(player1_id, player2_id) DO UPDATE SET
                player1_wins = player1_wins + excluded.player1_wins,
                player2_wins = player2_wins + excluded.player2_wins,
                last_updated = excluded.last_updated;
        """, (
            player1_id, player2_id, wins1_increment, wins2_increment,
            datetime.now().isoformat()
        ))

    def process_match_data(self, match_data: Dict[str, Any]) -> None:
        """
        Processes comprehensive match data, updating players, matches, and H2H stats.
        """
        if not self.conn:
            logger.error("Cannot process match data, no database connection.")
            return

        home_player_data = match_data.get("homePlayer")
        away_player_data = match_data.get("awayPlayer")
        winner_code = match_data.get("winnerCode")

        if not all([home_player_data, away_player_data, winner_code]):
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

            # <<< FIX: Call _insert_match with the correct arguments
            self._insert_match(cursor, match_data, home_player_id, away_player_id, winner_id)

            self._update_head_to_head(cursor, winner_id, loser_id)

            self.conn.commit()
            logger.info(f"Successfully processed match data for match ID {match_data.get('id')}.")

        except sqlite3.Error as e:
            logger.error(f"Database error processing match ID {match_data.get('id')}: {e}", exc_info=True)
            self.conn.rollback()

    def close(self) -> None:
        """Closes the database connection if it exists."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")