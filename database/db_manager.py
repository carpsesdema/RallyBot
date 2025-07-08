# database/db_manager.py
# FINAL HARDENED VERSION - NO COMPLEX QUERIES

import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from config import tennis_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Handles database connections and data processing.
    This version uses simple, robust SELECT/INSERT/UPDATE logic to avoid silent crashes
    from complex ON CONFLICT clauses. It is fully compatible with the complete tennis_schema.sql.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or tennis_config.database.database_path
        self.conn = None
        try:
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA journal_mode=WAL;") # Improve concurrency
            logger.info(f"Database connection established to {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise

    def _upsert_player(self, cursor: sqlite3.Cursor, player_data: Dict[str, Any]) -> Optional[int]:
        """Robustly inserts or updates a player."""
        api_player_id = player_data.get("id")
        if not api_player_id: return None

        cursor.execute("SELECT id FROM players WHERE api_player_id = ?", (api_player_id,))
        result = cursor.fetchone()

        if result:
            return result['id']
        else:
            player_info = {
                "api_player_id": api_player_id,
                "name": player_data.get("name"),
                "country_code": player_data.get("country", {}).get("alpha2"),
                "updated_at": datetime.now().isoformat()
            }
            cursor.execute("""
                INSERT INTO players (api_player_id, name, country_code, updated_at)
                VALUES (:api_player_id, :name, :country_code, :updated_at)
            """, player_info)
            return cursor.lastrowid

    def _upsert_match(self, cursor: sqlite3.Cursor, match_data: Dict[str, Any], p1_id: int, p2_id: int, win_id: int):
        """Robustly inserts or updates a match."""
        api_match_id = match_data.get('id')
        if not api_match_id: return

        match_info = {
            "api_match_id": api_match_id,
            "player1_id": p1_id,
            "player2_id": p2_id,
            "winner_id": win_id,
            "tournament_name": match_data.get('tournament', {}).get('name'),
            "round_name": match_data.get('roundInfo', {}).get('round'),
            "match_date": datetime.fromtimestamp(match_data["startTimestamp"]).date().isoformat() if match_data.get("startTimestamp") else None,
            "score_summary": f"{match_data.get('homeScore', {}).get('display', '')}-{match_data.get('awayScore', {}).get('display', '')}",
            "surface": match_data.get('tournament', {}).get('groundType'),
            "created_at": datetime.now().isoformat()
        }

        cursor.execute("SELECT id FROM matches WHERE api_match_id = ?", (api_match_id,))
        result = cursor.fetchone()

        if result:
            # Match exists, maybe update it
            pass # For now, we do nothing if the match already exists
        else:
            cursor.execute("""
                INSERT INTO matches (api_match_id, player1_id, player2_id, winner_id, tournament_name, round_name, match_date, score_summary, surface, created_at)
                VALUES (:api_match_id, :player1_id, :player2_id, :winner_id, :tournament_name, :round_name, :match_date, :score_summary, :surface, :created_at)
            """, match_info)

    def _update_head_to_head(self, cursor: sqlite3.Cursor, winner_id: int, loser_id: int):
        """Robustly updates head-to-head stats."""
        p1_id, p2_id = min(winner_id, loser_id), max(winner_id, loser_id)

        cursor.execute("SELECT * FROM head_to_head WHERE player1_id = ? AND player2_id = ?", (p1_id, p2_id))
        h2h_record = cursor.fetchone()

        p1_wins_inc = 1 if winner_id == p1_id else 0
        p2_wins_inc = 1 if winner_id == p2_id else 0

        if h2h_record:
            # Record exists, update it
            new_total = h2h_record['total_matches'] + 1
            new_p1_wins = h2h_record['player1_wins'] + p1_wins_inc
            new_p2_wins = h2h_record['player2_wins'] + p2_wins_inc

            cursor.execute("""
                UPDATE head_to_head SET
                    total_matches = ?,
                    player1_wins = ?,
                    player2_wins = ?,
                    last_match_date = ?,
                    last_match_winner_id = ?,
                    last_updated = ?
                WHERE id = ?
            """, (new_total, new_p1_wins, new_p2_wins, datetime.now().date().isoformat(), winner_id, datetime.now().isoformat(), h2h_record['id']))
        else:
            # No record, insert a new one
            cursor.execute("""
                INSERT INTO head_to_head (player1_id, player2_id, total_matches, player1_wins, player2_wins, last_match_date, last_match_winner_id, last_updated)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            """, (p1_id, p2_id, p1_wins_inc, p2_wins_inc, datetime.now().date().isoformat(), winner_id, datetime.now().isoformat()))

    def process_match_data(self, match_data: Dict[str, Any]):
        if not self.conn:
            logger.error("Cannot process match data, no database connection.")
            return

        home_player_data = match_data.get("homePlayer") or match_data.get("homeTeam")
        away_player_data = match_data.get("awayPlayer") or match_data.get("awayTeam")
        winner_code = match_data.get("winnerCode")

        if not all([home_player_data, away_player_data, winner_code is not None]):
            logger.warning(f"Skipping match ID {match_data.get('id', 'N/A')} due to incomplete player or winner data.")
            return

        cursor = self.conn.cursor()
        try:
            home_player_id = self._upsert_player(cursor, home_player_data)
            away_player_id = self._upsert_player(cursor, away_player_data)

            if not home_player_id or not away_player_id:
                raise ValueError("Failed to upsert one or both players.")

            winner_id = home_player_id if winner_code == 1 else away_player_id
            loser_id = away_player_id if winner_code == 1 else home_player_id

            self._upsert_match(cursor, match_data, home_player_id, away_player_id, winner_id)
            self._update_head_to_head(cursor, winner_id, loser_id)

            self.conn.commit()
            logger.info(f"✅ DB SUCCESS: Processed and saved match data for ID {match_data.get('id')}.")

        except (sqlite3.Error, ValueError) as e:
            logger.error(f"DB ERROR processing match ID {match_data.get('id')}: {e}", exc_info=True)
            self.conn.rollback()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")