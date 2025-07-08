import sqlite3
import sys
from pathlib import Path

# This script is now the single source of truth for creating the database.
# It's run by the Procfile BEFORE the server starts.

# This path MUST match the one used in config.py
# We are hardcoding it here to have zero dependencies.
DB_PATH = Path("/data/tennis_intelligence.db")


def main():
    """
    Creates the database and schema from tennis_schema.sql.
    Exits with code 1 on failure, 0 on success.
    """
    print("--- Running Production Database Setup ---")

    try:
        # Ensure the target directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Directory {DB_PATH.parent} ensured.")

        # Connect to the database (it will be created if it doesn't exist)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if the schema is already in place by looking for a key table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players';")
        if cursor.fetchone():
            print("Database schema already exists. Setup is complete.")
            conn.close()
            sys.exit(0)  # Exit successfully

        print("Schema not found. Creating from tennis_schema.sql...")

        # Find and execute the schema file
        script_dir = Path(__file__).parent
        schema_file = script_dir / "tennis_schema.sql"

        if not schema_file.exists():
            print(f"CRITICAL: Schema file not found at {schema_file}")
            conn.close()
            sys.exit(1)  # Exit with failure

        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Execute the entire schema script
        cursor.executescript(schema_sql)
        conn.commit()
        conn.close()

        print("✅ Database and schema created successfully.")
        sys.exit(0)  # Exit successfully

    except Exception as e:
        print(f"❌ CRITICAL DATABASE SETUP FAILED: {e}")
        # This will cause the Railway deployment to fail, showing this error in the deploy log.
        sys.exit(1)  # Exit with failure


if __name__ == "__main__":
    main()