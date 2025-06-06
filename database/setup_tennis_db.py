# database/setup_tennis_db.py - FIXED VERSION
import sqlite3
import os
from pathlib import Path


def create_tennis_database():
    """Create the tennis intelligence database with proper path handling"""

    # Get the script's directory to ensure correct paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level to project root

    print(f"📍 Script location: {script_dir}")
    print(f"📍 Project root: {project_root}")

    # Database directory (should be where this script is)
    db_dir = script_dir
    print(f"📍 Database directory: {db_dir}")

    # Schema file should be in the same directory as this script
    schema_file = script_dir / "tennis_schema.sql"
    print(f"📍 Looking for schema at: {schema_file}")

    # Check if schema file exists
    if not schema_file.exists():
        print(f"❌ Schema file not found at: {schema_file}")
        print(f"📋 Available files in {script_dir}:")
        for file in script_dir.iterdir():
            print(f"   - {file.name}")
        return None

    print(f"✅ Found schema file: {schema_file}")

    try:
        # Read the schema file
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        print(f"✅ Schema file read successfully ({len(schema_sql)} characters)")

        # Create database file
        db_path = db_dir / "tennis_intelligence.db"
        print(f"📍 Creating database at: {db_path}")

        # Connect and create database
        conn = sqlite3.connect(db_path)

        try:
            # Execute the entire schema
            conn.executescript(schema_sql)
            conn.commit()

            # Verify tables were created
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            print(f"✅ Tennis database created successfully!")
            print(f"📍 Database location: {db_path}")
            print(f"📊 Tables created: {len(tables)}")
            for table in tables:
                print(f"   - {table[0]}")

            return str(db_path)

        except Exception as e:
            print(f"❌ Error executing schema: {e}")
            return None

    except FileNotFoundError as e:
        print(f"❌ Schema file not found: {e}")
        return None
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def test_database(db_path):
    """Test the database by inserting some sample data"""
    if not db_path or not Path(db_path).exists():
        print("❌ Database not found for testing")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Test insert a player
        cursor.execute("""
            INSERT INTO players (name, short_name, gender, country_code) 
            VALUES (?, ?, ?, ?)
        """, ("Novak Djokovic", "N. Djokovic", "M", "SRB"))

        # Test query
        cursor.execute("SELECT COUNT(*) FROM players")
        count = cursor.fetchone()[0]

        conn.commit()
        conn.close()

        print(f"✅ Database test successful! {count} player(s) in database")
        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎾 Tennis Intelligence Database Setup")
    print("=" * 50)

    # Create the database
    db_path = create_tennis_database()

    if db_path:
        print("\n🧪 Testing database...")
        test_database(db_path)
        print(f"\n🎉 Setup complete! Database ready at:")
        print(f"   {db_path}")
    else:
        print("\n❌ Setup failed. Please check the errors above.")