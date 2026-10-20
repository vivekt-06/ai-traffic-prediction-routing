"""Central configuration: paths + JSON config files (dataset mapping, congestion parameters)."""
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONTEXT_DIR = DATA_DIR / "contextual"
MODEL_DIR = ROOT / "models" / "trained"
DB_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "data" / "app.db"))


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def dataset_config() -> dict:
    return _load("dataset.json")


def congestion_config() -> dict:
    return _load("congestion.json")


# Weather location (city centre, used for the Open-Meteo request). Not a secret.
WEATHER_LAT = float(os.getenv("WEATHER_LAT", "28.6139"))
WEATHER_LON = float(os.getenv("WEATHER_LON", "77.2090"))
WEATHER_TTL_SECONDS = int(os.getenv("WEATHER_TTL_SECONDS", "1800"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "8"))
