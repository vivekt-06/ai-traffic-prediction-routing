"""
Download hourly Open-Meteo history for the dataset window -> data/contextual/weather_history.parquet.
Needs internet; no API key. Afterwards: python scripts/train_model.py --with-weather
Usage: python scripts/fetch_weather_history.py
"""
import sys
from datetime import date
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONTEXT_DIR
from src.db import init_db
from src.panel import Panel
from src.weather import fetch_hourly

if __name__ == "__main__":
    init_db(reload_events=False)
    p = Panel()
    r = fetch_hourly(p.timestamps[0].date(), p.timestamps[-1].date(), today=date.today())
    if not r.get("available"):
        raise SystemExit(r["error"])
    df = pd.DataFrame(r["hourly"]).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    df = df.reindex(p.timestamps)
    if df.isna().any().any():
        raise SystemExit(f"Open-Meteo response has gaps: {df.isna().sum().to_dict()}")
    out = CONTEXT_DIR / "weather_history.parquet"
    df.to_parquet(out)
    print("saved", out, df.shape, "source:", r["source"])
