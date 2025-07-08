import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def main():
    """
    Connects to the PostgreSQL database and creates the schema.
    This is run by the start command BEFORE the server starts.
    """
    print("--- Running Production Database Setup (PostgreSQL) ---")

    # Railway provides the DATABASE_URL environment variable automatically
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("CRITICAL: DATABASE_URL environment variable not found.")
        sys.exit(1)

    # SQLAlchemy uses 'postgresql' as the dialect name
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    try:
        # Create an engine to connect to the database
        engine = create_engine(db_url)
        with engine.connect() as connection:
            print("✅ Database connection successful.")

            # Find and execute the schema file
            script_dir = Path(__file__).parent
            schema_file = script_dir / "tennis_schema.sql"

            if not schema_file.exists():
                print(f"CRITICAL: Schema file not found at {schema_file}")
                sys.exit(1)

            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            # Execute the entire schema script
            connection.execute(text(schema_sql))
            connection.commit()
            print("✅ Schema setup complete or already exists.")

    except OperationalError as e:
        print(f"❌ CRITICAL: Could not connect to the database: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ CRITICAL DATABASE SETUP FAILED: {e}")
        sys.exit(1)

    print("--- Database setup finished successfully. ---")
    sys.exit(0)

if __name__ == "__main__":
    main()