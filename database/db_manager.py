# database/db_manager.py
# FINAL POSTGRES VERSION

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# --- Database Connection Setup ---
# This safely gets the DATABASE_URL provided by Railway.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class DatabaseManager:
    """
    Handles database connections and data processing using SQLAlchemy for PostgreSQL.
    This is the final, correct version for the new architecture.
    """

    def __init__(self):
        if not DATABASE_URL:
            # This will cause a clean failure if the variable is missing.
            raise ValueError(
                "CRITICAL: DATABASE_URL environment variable is not set or not visible to the application.")
        try:
            self.engine = create_engine(DATABASE_URL)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("DatabaseManager initialized with SQLAlchemy engine for PostgreSQL.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy engine: {e}", exc_info=True)
            raise

    def process_match_data(self, match_data: Dict[str, Any]):
        """Processes and saves one match's data to the PostgreSQL database."""
        session = self.Session()
        try:
            home_player_data = match_data.get("homePlayer") or match_data.get("homeTeam")
            away_player_data = match_data.get("awayPlayer") or match_data.get("awayTeam")
            winner_code = match_data.get("winnerCode")

            if not all([home_player_data, away_player_data, winner_code is not None]):
                return

            # This block of code runs inside a single database transaction.
            home_player_id = self._upsert_player(session, home_player_data)
            away_player_id = self._upsert_player(session, away_player_data)

            if not home_player_id or not away_player_id:
                raise ValueError("Failed to upsert one or both players.")

            winner_id = home_player_id if winner_code == 1 else away_player_id

            self._upsert_match(session, match_data, home_player_id, away_player_id, winner_id)
            self._update_head_to_head(session, winner_id, home_player_id, away_player_id)

            session.commit()
            logger.info(f"✅ DB SUCCESS: Processed and saved match data for ID {match_data.get('id')}.")

        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"DB ERROR processing match ID {match_data.get('id')}: {e}", exc_info=True)
            session.rollback()
        finally:
            session.close()

    def _upsert_player(self, session, player_data: Dict[str, Any]) -> Optional[int]:
        api_player_id = str(player_data.get("id"))
        if not api_player_id: return None

        result = session.execute(text("SELECT id FROM players WHERE api_player_id = :api_id"),
                                 {"api_id": api_player_id}).fetchone()
        if result:
            return result[0]

        insert_stmt = text("""
            INSERT INTO players (api_player_id, name, country_code, updated_at)
            VALUES (:api_player_id, :name, :country_code, :updated_at) RETURNING id;
        """)
        params = {
            "api_player_id": api_player_id,
            "name": player_data.get("name"),
            "country_code": player_data.get("country", {}).get("alpha2"),
            "updated_at": datetime.now()
        }
        result = session.execute(insert_stmt, params).fetchone()
        return result[0] if result else None

    def _upsert_match(self, session, match_data: Dict[str, Any], p1_id: int, p2_id: int, win_id: int):
        stmt = text("""
            INSERT INTO matches (api_match_id, player1_id, player2_id, winner_id, tournament_name, round_name, match_date, score_summary, surface, created_at)
            VALUES (:api_match_id, :p1_id, :p2_id, :win_id, :t_name, :r_name, :m_date, :score, :surface, :created_at)
            ON CONFLICT (api_match_id) DO NOTHING;
        """)
        params = {
            "api_match_id": str(match_data.get("id")), "p1_id": p1_id, "p2_id": p2_id, "win_id": win_id,
            "t_name": match_data.get("tournament", {}).get("name"),
            "r_name": match_data.get("roundInfo", {}).get("round"),
            "m_date": datetime.fromtimestamp(match_data["startTimestamp"]).date() if match_data.get(
                "startTimestamp") else None,
            "score": f"{match_data.get('homeScore', {}).get('display', '')}-{match_data.get('awayScore', {}).get('display', '')}",
            "surface": match_data.get('tournament', {}).get('groundType'),
            "created_at": datetime.now()
        }
        session.execute(stmt, params)

    def _update_head_to_head(self, session, winner_id: int, p1_id_any: int, p2_id_any: int):
        p1_id, p2_id = min(p1_id_any, p2_id_any), max(p1_id_any, p2_id_any)

        h2h_id_result = session.execute(
            text("SELECT id FROM head_to_head WHERE player1_id = :p1_id AND player2_id = :p2_id"),
            {"p1_id": p1_id, "p2_id": p2_id}).fetchone()

        p1_wins_inc = 1 if winner_id == p1_id else 0
        p2_wins_inc = 1 if winner_id == p2_id else 0

        if h2h_id_result:
            stmt = text(f"""
                UPDATE head_to_head SET total_matches = total_matches + 1, player1_wins = player1_wins + {p1_wins_inc}, player2_wins = player2_wins + {p2_wins_inc}, last_match_date = :last_date, last_match_winner_id = :winner_id, last_updated = :updated
                WHERE id = :h2h_id;
            """)
            params = {"last_date": datetime.now().date(), "winner_id": winner_id, "updated": datetime.now(),
                      "h2h_id": h2h_id_result[0]}
        else:
            stmt = text("""
                INSERT INTO head_to_head (player1_id, player2_id, total_matches, player1_wins, player2_wins, last_match_date, last_match_winner_id, last_updated)
                VALUES (:p1_id, :p2_id, 1, :p1_wins, :p2_wins, :last_date, :winner_id, :updated);
            """)
            params = {"p1_id": p1_id, "p2_id": p2_id, "p1_wins": p1_wins_inc, "p2_wins": p2_wins_inc,
                      "last_date": datetime.now().date(), "winner_id": winner_id, "updated": datetime.now()}

        session.execute(stmt, params)

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()