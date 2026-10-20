"""Create the SQLite database and load the curated events table. Usage: python scripts/init_db.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import init_db
if __name__ == "__main__":
    print("database ready at", init_db())
