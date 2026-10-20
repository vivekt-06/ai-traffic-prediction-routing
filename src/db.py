"""SQLite persistence: events table (loaded from the curated CSV), weather response cache, query log."""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.config import CONTEXT_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY, event_name TEXT NOT NULL, event_type TEXT, location TEXT,
  lat REAL, lon REAL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, impact_radius_m REAL,
  citywide INTEGER NOT NULL DEFAULT 0, source TEXT, verification_status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS weather_cache (
  cache_key TEXT PRIMARY KEY, fetched_at REAL NOT NULL, source_url TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS query_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, kind TEXT,
  params TEXT, model TEXT, result_summary TEXT);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(reload_events=True):
    with connect() as con:
        con.executescript(SCHEMA)
        if reload_events:
            df = pd.read_csv(CONTEXT_DIR / "events.csv")
            df["citywide"] = df["citywide"].astype(str).str.lower().isin(["true", "1"]).astype(int)
            con.execute("DELETE FROM events")
            df.to_sql("events", con, if_exists="append", index=False)
    return DB_PATH


def log_query(kind, params, model, summary):
    try:
        with connect() as con:
            con.execute("INSERT INTO query_log(kind, params, model, result_summary) VALUES (?,?,?,?)",
                        (kind, str(params), model, str(summary)))
    except Exception:  # logging must never break a request
        pass
