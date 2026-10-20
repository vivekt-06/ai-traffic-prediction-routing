"""Departure-time comparison: same O-D, departures Now / +15 / +30 / +45 min, each re-routed with TDSP."""
import pandas as pd

from src.config import congestion_config
from src.forecast import ForecastUnavailable
from src.graph import get_graph
from src.forecast import get_forecaster
from src.routing import Field, compute_routes
from src.tdsp import NoRoute

OFFSETS = (0, 15, 30, 45)


def recommend(rows: list, negligible_min: float):
    """Pure function (unit-tested). rows: dicts with duration_min / arrival / offset_min / available."""
    ok = [r for r in rows if r.get("available")]
    if not ok:
        return {"best_offset_min": None, "explanation": "No departure scenario could be evaluated (forecast unavailable)."}
    now = next((r for r in ok if r["offset_min"] == 0), None)
    best = min(ok, key=lambda r: (r["duration_min"], r["offset_min"]))
    if now is None:
        return {"best_offset_min": best["offset_min"], "explanation":
                f"'Now' could not be evaluated; among available options {best['label']} has the shortest trip ({best['duration_min']:.1f} min)."}
    saving = now["duration_min"] - best["duration_min"]
    if best["offset_min"] == 0 or saving < negligible_min:
        return {"best_offset_min": 0 if best["offset_min"] == 0 else best["offset_min"],
                "negligible": best["offset_min"] != 0,
                "explanation": ("Leaving now gives the shortest predicted trip "
                                f"({now['duration_min']:.1f} min); waiting does not reduce it."
                                if best["offset_min"] == 0 else
                                f"{best['label']} is predicted to be only {saving:.1f} min faster than leaving now "
                                f"(below the {negligible_min:g} min threshold) - no meaningful benefit from waiting.")}
    later = (pd.Timestamp(best["arrival"]) - pd.Timestamp(now["arrival"])).total_seconds() / 60
    return {"best_offset_min": best["offset_min"], "time_saved_vs_now_min": round(saving, 2), "negligible": False,
            "arrival_delta_vs_now_min": round(later, 2),
            "explanation": (f"Leaving {best['label']} gives a predicted trip of {best['duration_min']:.1f} min versus "
                            f"{now['duration_min']:.1f} min now, i.e. {saving:.1f} min less driving. "
                            f"Net arrival is {abs(later):.1f} min {'later' if later > 0 else 'earlier'} than leaving now.")}


def analyse(origin, destination, depart_ts, model=None, offsets=OFFSETS):
    fc = get_forecaster()
    model = model or fc.default_model()
    depart_ts = pd.Timestamp(depart_ts)
    rows = []
    for off in offsets:
        label = "Now" if off == 0 else f"+{off} min"
        ts = depart_ts + pd.Timedelta(minutes=off)
        row = {"offset_min": off, "label": label, "departure": str(ts), "available": False}
        try:
            r = compute_routes(origin, destination, ts, model)
            row.update(available=True, duration_min=r["tdsp"]["duration_min"], distance_km=r["tdsp"]["distance_km"],
                       arrival=r["tdsp"]["arrival"], mean_ci=r["tdsp"]["congestion_exposure"]["mean_ci"],
                       level=r["tdsp"]["congestion_exposure"]["mean_level"],
                       shortest_route_duration_min=r["shortest"]["duration_min"],
                       delay_min=r["tdsp"]["delay_min"])
        except ForecastUnavailable as e:
            row["reason"] = str(e)
        rows.append(row)
    rec = recommend(rows, congestion_config()["negligible_saving_minutes"])
    return {"model": model, "scenarios": rows, "recommendation": rec,
            "note": "Each scenario re-runs TDSP with that departure time; forecasts are hourly buckets, so a +15/+30/+45 min "
                    "departure changes which hourly forecast applies along the route."}
