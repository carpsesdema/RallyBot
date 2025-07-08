import sqlite3
import os
from pathlib import Path

# Import the centralized tennis config
from config import tennis_config


def create_tennis_database():
    """
    Creates the tennis intelligence database directly on the Railway persistent volume.
    This version removes environment detection to be 100% reliable and uses centralized config.
    """
    # Use the path from the centralized config
    db_path = Path(tennis_config.database.database_path)
    db_dir = db_path.parent

    print("🚀 Targeting Railway persistent volume directly.")
    print(f"📍 Database path set to: {db_path} (from config)")

    # Ensure the target directory exists
    db_dir.mkdir(parents=True, exist_ok=True)

    # Schema file is always relative to this script's location inside the container
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
        print(f"✅ Schema file read successfully.")

        conn = sqlite3.connect(db_path)
        # Check if tables already exist to avoid errors on redeploy
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players';")
        if cursor.fetchone():
            print("✅ Database tables already exist. Setup is already complete.")
            return str(db_path)

        # If tables don't exist, create them
        print("Database tables not found. Creating them now...")
        cursor.executescript(schema_sql)
        conn.commit()

        # Verify tables were created
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