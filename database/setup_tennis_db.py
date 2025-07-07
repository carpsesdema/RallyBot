# database/setup_tennis_db.py - FIXED VERSION
import sqlite3
import os
from pathlib import Path


def create_tennis_database():
    """Create the tennis intelligence database with proper path handling for production."""

    # For production (like Railway), we want the DB on a persistent volume, typically /data
    # For local dev, we can keep it in the database directory.
    is_production = os.getenv("RAILWAY_APP_URL") is not None
    if is_production:
        db_dir = Path("/data")
        db_path = db_dir / "tennis_intelligence.db"
        print("🚀 Production environment detected (Railway). Targeting persistent volume.")
    else:
        db_dir = Path(__file__).parent
        db_path = db_dir / "tennis_intelligence.db"
        print("🏠 Local environment detected. Targeting local 'database' directory.")

    print(f"📍 Database will be created at: {db_path}")

    # Ensure the target directory exists
    db_dir.mkdir(parents=True, exist_ok=True)

    # Schema file is always relative to this script's location
    script_dir = Path(__file__).parent
    schema_file = script_dir / "tennis_schema.sql"
    print(f"📍 Looking for schema at: {schema_file}")

    if not schema_file.exists():
        print(f"❌ Schema file not found at: {schema_file}")
        return None

    print(f"✅ Found schema file: {schema_file}")

    conn = None
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        print(f"✅ Schema file read successfully ({len(schema_sql)} characters)")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print(f"✅ Tennis database created successfully!")
        print(f"📍 Database location: {db_path}")
        print(f"📊 Tables created: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")

        return str(db_path)

    except Exception as e:
        print(f"❌ Error creating or verifying database: {e}")
        return None
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print("🎾 Tennis Intelligence Database Setup")
    print("=" * 50)
    db_path = create_tennis_database()
    if db_path:
        print("\n🎉 Setup complete! Database is ready.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")