import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import the centralized tennis config
from config import tennis_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Handles database connections and data processing.
    This version assumes the database and schema have ALREADY been created by a setup script.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the DatabaseManager, connecting to the DB.
        It NO LONGER handles schema creation.
        """
        self.db_path = db_path or tennis_config.database.database_path
        self.conn = None
        try:
            # The setup script in the Procfile is responsible for creating the directory and file.
            # This code now assumes it exists.
            if not Path(self.db_path).exists():
                 # This is a fallback for local testing, in production the Procfile handles it.
                 logger.warning(f"Database file not found at {self.db_path}. A new empty one will be created, but it will lack a schema.")

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
            logger.info(f"Database connection established to {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise

    def _upsert_player(self, cursor: sqlite3.Cursor, player_data: Dict[str, Any]) -> Optional[int]:
        """Inserts or updates a player's details, compatible with the full schema."""
        if not player_data or not player_data.get("id") or not player_data.get("name"):
            return None

        player_info = {
            "api_player_id": player_data.get("id"),
            "name": player_data.get("name"),
            "short_name": player_data.get("shortName"),
            "gender": player_data.get("gender"),
            "country_code": player_data.get("country", {}).get("alpha2"),
            "country_name": player_data.get("country", {}).get("name"),
            "date_of_birth": datetime.fromtimestamp(player_data["dateOfBirthTimestamp"]).date().isoformat() if player_data.get("dateOfBirthTimestamp") else None,
            "turned_pro": player_data.get("turnedPro"),
            "height_cm": player_data.get("height"),
            "weight_kg": player_data.get("weight"),
            "plays": player_data.get("plays"),
            "backhand": None,
            "updated_at": datetime.now().isoformat()
        }

        cursor.execute("""
            INSERT INTO players (api_player_id, name, short_name, gender, country_code, country_name, date_of_birth, turned_pro, height_cm, weight_kg, plays, backhand, updated_at)
            VALUES (:api_player_id, :name, :short_name, :gender, :country_code, :country_name, :date_of_birth, :turned_pro, :height_cm, :weight_kg, :plays, :backhand, :updated_at)
            ON CONFLICT(api_player_id) DO UPDATE SET
                name=excluded.name,
                country_code=excluded.country_code,
                updated_at=excluded.updated_at;
        """, player_info)

        cursor.execute("SELECT id FROM players WHERE api_player_id = ?", (player_info["api_player_id"],))
        player_row = cursor.fetchone()
        return player_row['id'] if player_row else None

    def _insert_match(self, cursor: sqlite3.Cursor, match_data: Dict[str, Any], player1_id: int, player2_id: int, winner_id: int):
        """Inserts or updates a match record, compatible with the full schema."""
        tournament = match_data.get('tournament', {})
        round_info = match_data.get('roundInfo', {})
        home_score = match_data.get('homeScore', {})
        away_score = match_data.get('awayScore', {})
        match_timestamp = match_data.get('startTimestamp')

        match_info = {
            "api_match_id": match_data.get('id'),
            "match_date": datetime.fromtimestamp(match_timestamp).date().isoformat() if match_timestamp else None,
            "tournament_name": tournament.get('name'),
            "tournament_level": tournament.get('category', {}).get('name'),
            "round_name": round_info.get('round'),
            "surface": tournament.get('groundType'),
            "player1_id": player1_id,
            "player2_id": player2_id,
            "winner_id": winner_id,
            "score_summary": f"{home_score.get('display', '')}-{away_score.get('display', '')}",
            "created_at": datetime.now().isoformat()
        }

        cursor.execute("""
            INSERT INTO matches (api_match_id, player1_id, player2_id, winner_id, tournament_name, round_name, match_date, score_summary, surface, tournament_level, created_at)
            VALUES (:api_match_id, :player1_id, :player2_id, :winner_id, :tournament_name, :round_name, :match_date, :score_summary, :surface, :tournament_level, :created_at)
            ON CONFLICT(api_match_id) DO UPDATE SET
                winner_id=excluded.winner_id,
                score_summary=excluded.score_summary;
        """, match_info)

    def _update_head_to_head(self, cursor: sqlite3.Cursor, winner_id: int, loser_id: int, match_data: Dict[str, Any]):
        """Updates H2H stats, compatible with the full schema and with corrected win logic."""
        player1_id, player2_id = min(winner_id, loser_id), max(winner_id, loser_id)
        p1_wins_inc = 1 if winner_id == player1_id else 0
        p2_wins_inc = 1 if winner_id == player2_id else 0
        surface = match_data.get('tournament', {}).get('groundType', 'unknown').lower()
        clay_inc = 1 if surface == 'clay' else 0
        grass_inc = 1 if surface == 'grass' else 0
        hard_inc = 1 if surface == 'hard' else 0
        p1_clay_wins_inc = 1 if p1_wins_inc and clay_inc else 0
        p1_grass_wins_inc = 1 if p1_wins_inc and grass_inc else 0
        p1_hard_wins_inc = 1 if p1_wins_inc and hard_inc else 0

        cursor.execute(f"""
            INSERT INTO head_to_head (
                player1_id, player2_id, total_matches, player1_wins, player2_wins,
                last_match_date, last_match_winner_id, last_updated,
                clay_matches, clay_player1_wins,
                grass_matches, grass_player1_wins,
                hard_matches, hard_player1_wins
            )
            VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player1_id, player2_id) DO UPDATE SET
                total_matches = total_matches + 1,
                player1_wins = player1_wins + {p1_wins_inc},
                player2_wins = player2_wins + {p2_wins_inc},
                last_match_date = excluded.last_match_date,
                last_match_winner_id = excluded.last_match_winner_id,
                last_updated = excluded.last_updated,
                clay_matches = clay_matches + {clay_inc},
                clay_player1_wins = clay_player1_wins + {p1_clay_wins_inc},
                grass_matches = grass_matches + {grass_inc},
                grass_player1_wins = grass_player1_wins + {p1_grass_wins_inc},
                hard_matches = hard_matches + {hard_inc},
                hard_player1_wins = hard_player1_wins + {p1_hard_wins_inc};
        """, (
            player1_id, player2_id, p1_wins_inc, p2_wins_inc,
            datetime.now().date().isoformat(), winner_id, datetime.now().isoformat(),
            clay_inc, p1_clay_wins_inc,
            grass_inc, p1_grass_wins_inc,
            hard_inc, p1_hard_wins_inc
        ))

    def process_match_data(self, match_data: Dict[str, Any]):
        """Processes comprehensive match data, updating players, matches, and H2H stats."""
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
                raise sqlite3.Error("Failed to upsert one or both players.")
            winner_id = home_player_id if winner_code == 1 else away_player_id
            loser_id = away_player_id if winner_code == 1 else home_player_id
            self._insert_match(cursor, match_data, home_player_id, away_player_id, winner_id)
            self._update_head_to_head(cursor, winner_id, loser_id, match_data)
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