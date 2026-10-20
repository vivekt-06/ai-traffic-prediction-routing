"""Route computation: shortest-distance vs congestion-aware TDSP, evaluated under the same forecast."""
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import congestion_config
from src.congestion import LEVELS
from src.forecast import ForecastUnavailable, get_forecaster
from src.graph import get_graph
from src.tdsp import NoRoute, simulate, static_dijkstra, time_dependent_dijkstra

log = logging.getLogger("routing")


class Field:
    """Forecast speeds/congestion as a function of dataset hour for one model."""
    def __init__(self, fc, model):
        self.fc, self.model = fc, model
    def speed_at(self, h):
        if not 0 <= h < self.fc.panel.T:
            raise ForecastUnavailable(f"forecast for {self.fc.panel.timestamps[0] + pd.Timedelta(hours=h)} is outside "
                                      f"dataset coverage ({self.fc.panel.timestamps[0]} .. {self.fc.panel.timestamps[-1]})")
        return self.fc.congestion(h, self.model)["speed_list"]
    def ci_at(self, h):
        self.speed_at(h)
        return self.fc.congestion(h, self.model)


def _ts(fc, seconds):
    return fc.panel.timestamps[0] + pd.Timedelta(seconds=float(seconds))


def summarize(g, fc, ev, label, t0_s):
    cfg = congestion_config()
    rows = ev.edge_rows
    dur_min = (ev.arrive_s - ev.depart_s) / 60.0
    ff_min = sum(r["length_km"] / g.free_flow[r["edge"]] * 60 for r in rows)
    tot = max(ev.distance_km, 1e-9)
    ci_w = sum(r["ci"] * r["length_km"] for r in rows) / tot
    high = [r for r in rows if r["level"] >= 2]
    mins_high = sum((r["exit_s"] - r["enter_s"]) / 60 for r in high)
    level_km = {LEVELS[i]: round(sum(r["length_km"] for r in rows if r["level"] == i), 3) for i in range(4)}
    # important congested segments: rank by congestion-weighted time spent
    ranked = sorted(rows, key=lambda r: r["ci"] * (r["exit_s"] - r["enter_s"]), reverse=True)
    merged, seen = [], set()
    for r in ranked:
        key = g.street[r["edge"]]
        if key in seen or r["level"] < 1:
            continue
        seen.add(key)
        merged.append({"segment_id": int(g.static["segment_id"].iat[r["edge"]]), "street": str(key),
                       "length_m": round(r["length_km"] * 1000), "congestion_index": round(r["ci"], 3),
                       "level": LEVELS[r["level"]], "speed_kmh": round(r["speed_kmh"], 1),
                       "entered_at": str(_ts(fc, r["enter_s"]))})
        if len(merged) == 5:
            break
    geometry = []
    segs = []
    for r in rows:
        pts = g.geometry(r["edge"], r["reverse"])
        segs.append({"points": pts, "level": LEVELS[r["level"]], "street": str(g.street[r["edge"]]),
                     "ci": round(r["ci"], 3)})
        geometry.extend(pts if not geometry else pts[1:])
    return {
        "label": label, "distance_km": round(ev.distance_km, 3), "duration_min": round(dur_min, 2),
        "free_flow_duration_min": round(ff_min, 2), "delay_min": round(dur_min - ff_min, 2),
        "departure": str(_ts(fc, ev.depart_s)), "arrival": str(_ts(fc, ev.arrive_s)),
        "congestion_exposure": {"mean_ci": round(ci_w, 4), "mean_level": _lvl(ci_w),
                                "minutes_in_high_or_severe": round(mins_high, 2),
                                "km_by_level": level_km,
                                "pct_distance_high_or_severe": round(100 * sum(r["length_km"] for r in high) / tot, 1)},
        "top_congested_segments": merged, "n_segments": len(rows),
        "geometry": geometry, "segments": segs,
    }


def _lvl(ci):
    from src.congestion import classify_ci
    return classify_ci(ci)


def compute_routes(origin, destination, depart_ts, model=None):
    """origin/destination: (lat, lon). Returns both routes plus diffs, or raises a clear error."""
    fc, g = get_forecaster(), get_graph()
    model = model or fc.default_model()
    depart_ts = pd.Timestamp(depart_ts)
    t0 = (depart_ts - fc.panel.timestamps[0]).total_seconds()
    if t0 < 0 or depart_ts > fc.panel.timestamps[-1] + pd.Timedelta(minutes=59):
        raise ForecastUnavailable(f"departure {depart_ts} is outside the forecast window "
                                  f"{fc.panel.timestamps[0]} .. {fc.panel.timestamps[-1] + pd.Timedelta(minutes=59)}")
    s, ds = g.snap(*origin)
    d, dd = g.snap(*destination)
    if s == d:
        raise ValueError("origin and destination snap to the same junction")
    f = Field(fc, model)
    arcs_t, _ = time_dependent_dijkstra(g, s, d, t0, f.speed_at)
    arcs_s = static_dijkstra(g, s, d)
    ev_t = simulate(g, arcs_t, t0, f.speed_at, f.ci_at)
    ev_s = simulate(g, arcs_s, t0, f.speed_at, f.ci_at)
    r_t = summarize(g, fc, ev_t, "Congestion-aware (TDSP)", t0)
    r_s = summarize(g, fc, ev_s, "Shortest distance", t0)
    return {
        "model": model, "model_label": next(m["label"] for m in fc.registry() if m["key"] == model),
        "split_of_departure_hour": fc.split_label(fc.panel.index_of(depart_ts)),
        "snapped": {"origin": {"node": s, "lat": g.node_xy(s)[0], "lon": g.node_xy(s)[1], "snap_m": round(ds)},
                    "destination": {"node": d, "lat": g.node_xy(d)[0], "lon": g.node_xy(d)[1], "snap_m": round(dd)}},
        "tdsp": r_t, "shortest": r_s,
        "difference": {"time_saved_min": round(r_s["duration_min"] - r_t["duration_min"], 2),
                       "extra_distance_km": round(r_t["distance_km"] - r_s["distance_km"], 3),
                       "same_path": arcs_t == arcs_s},
        "notes": ["Speeds are a heuristic derived from predicted probe counts (dataset has no measured speed).",
                  "Segments are treated as bidirectional (dataset has no one-way attribute)."],
    }
