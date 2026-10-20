"""
Curated events (data/contextual/events.csv -> SQLite). Events are CONTEXT: this module only reports
spatial + temporal overlap with a route. It does not adjust predicted speeds; no automatic event
detection exists. Impact radii are documented assumptions in the CSV, not learned values.
"""
import numpy as np
import pandas as pd

from src.db import connect
from src.graph import _haversine_m


def list_events() -> list:
    with connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM events ORDER BY start_time")]
    for r in rows:
        r["citywide"] = bool(r["citywide"])
    return rows


def events_overlapping(route_points, t_start, t_end) -> list:
    """route_points: [[lat, lon], ...]; t_start/t_end: timestamps of the trip."""
    t_start, t_end = pd.Timestamp(t_start), pd.Timestamp(t_end)
    pts = np.asarray(route_points, dtype=float).reshape(-1, 2)
    out = []
    for e in list_events():
        s, en = pd.Timestamp(e["start_time"]), pd.Timestamp(e["end_time"])
        if not (s < t_end and en > t_start):
            continue
        item = {**e, "overlap_type": None, "min_distance_m": None}
        if e["citywide"]:
            item["overlap_type"] = "citywide (temporal only)"
        elif len(pts):
            d = float(_haversine_m(e["lon"], e["lat"], pts[:, 1], pts[:, 0]).min())
            item["min_distance_m"] = round(d)
            if d <= (e["impact_radius_m"] or 0):
                item["overlap_type"] = "route passes inside impact radius"
        if item["overlap_type"]:
            out.append(item)
    return out
