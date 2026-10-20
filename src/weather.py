"""
Open-Meteo client (free, non-commercial, no API key). Responses are cached in SQLite.
Historical dates -> archive API; recent/future -> forecast API. Failures never raise to the
caller: they return {"available": False, "error": ...} (stale cache is returned, flagged, if present).
Weather is CONTEXT only unless a *_weather model has been trained (see README).
"""
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from src.config import HTTP_TIMEOUT, WEATHER_LAT, WEATHER_LON, WEATHER_TTL_SECONDS
from src.db import connect

log = logging.getLogger("weather")
HOURLY = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 7


def _cache_get(key):
    with connect() as con:
        r = con.execute("SELECT fetched_at, payload, source_url FROM weather_cache WHERE cache_key=?", (key,)).fetchone()
    return (r["fetched_at"], json.loads(r["payload"]), r["source_url"]) if r else None


def _cache_put(key, payload, url):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO weather_cache VALUES (?,?,?,?)", (key, time.time(), url, json.dumps(payload)))


def fetch_hourly(start: date, end: date, client=None, today: date = None) -> dict:
    today = today or date.today()
    historical = end < today - timedelta(days=ARCHIVE_LAG_DAYS)
    url = ARCHIVE_URL if historical else FORECAST_URL
    params = {"latitude": WEATHER_LAT, "longitude": WEATHER_LON, "hourly": HOURLY, "timezone": "Asia/Kolkata",
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    key = f"{url}|{start}|{end}|{WEATHER_LAT}|{WEATHER_LON}"
    ttl = 30 * 86400 if historical else WEATHER_TTL_SECONDS
    cached = _cache_get(key)
    if cached and time.time() - cached[0] < ttl:
        return {**cached[1], "cache": "hit"}
    try:
        c = client or httpx
        resp = c.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        h = resp.json()["hourly"]
        payload = {"available": True, "source": "Open-Meteo " + ("Historical Weather API" if historical else "Forecast API"),
                   "source_url": url, "units": resp.json().get("hourly_units", {}), "latitude": WEATHER_LAT,
                   "longitude": WEATHER_LON, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "hourly": [{"time": t, **{k: h[k][i] for k in HOURLY.split(",")}} for i, t in enumerate(h["time"])]}
        _cache_put(key, payload, url)
        return {**payload, "cache": "miss"}
    except Exception as e:                         # network, HTTP status, malformed JSON
        log.warning("Open-Meteo request failed: %s", e)
        if cached:
            return {**cached[1], "cache": "stale", "warning": f"live request failed ({type(e).__name__}); showing cached data"}
        return {"available": False, "error": f"Open-Meteo unreachable ({type(e).__name__}). Weather is optional context; routing and predictions are unaffected.",
                "source": "Open-Meteo", "source_url": url}
