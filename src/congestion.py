"""
Predicted probe count -> estimated speed -> Congestion Index -> level.

IMPORTANT (honest scope): the dataset has NO measured speeds. The speed here is a
documented, configurable, UNCALIBRATED heuristic (configs/congestion.json):

    intensity = clip(pred_probe / saturation_ref, 0, intensity_cap)
                (saturation_ref = segment's own P90 probe count over the TRAIN period)
    speed     = max(free_flow * (1 - k * intensity), min_speed)
    CI        = 1 - speed / free_flow            (clipped to [0, 1])

free_flow = the segment's posted speed limit. Everything downstream (map colour, charts,
route cost, exposure) uses these same arrays, so values are consistent everywhere.
"""
import numpy as np

from src.config import congestion_config

LEVELS = ["LOW", "MODERATE", "HIGH", "SEVERE"]


def thresholds() -> list:
    t = congestion_config()["thresholds"]
    return [t["LOW"], t["MODERATE"], t["HIGH"]]


def level_codes(ci) -> np.ndarray:
    """0..3 for LOW..SEVERE, -1 for invalid (NaN) values."""
    ci = np.asarray(ci, dtype=np.float64)
    codes = np.searchsorted(np.asarray(thresholds()), ci, side="right")
    return np.where(np.isfinite(ci), codes, -1).astype(np.int8)


def level_name(code: int) -> str:
    return LEVELS[code] if 0 <= code < len(LEVELS) else "UNKNOWN"


def classify_ci(ci: float) -> str:
    return level_name(int(level_codes(np.array([ci]))[0]))


def compute_congestion(pred_probe, saturation_ref, free_flow_kmh, cfg=None):
    """Vectorised. Invalid inputs (NaN / non-positive free flow) give NaN speed/CI (never a silent 0)."""
    cfg = cfg or congestion_config()
    pred = np.asarray(pred_probe, dtype=np.float64)
    sat = np.maximum(np.asarray(saturation_ref, dtype=np.float64), 1.0)
    ff = np.asarray(free_flow_kmh, dtype=np.float64)
    intensity = np.clip(np.maximum(pred, 0) / sat, 0.0, cfg["intensity_cap"])
    speed = np.maximum(ff * (1.0 - cfg["k_speed_loss"] * intensity), cfg["min_speed_kmh"])
    speed = np.minimum(speed, ff)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = np.clip(1.0 - speed / ff, 0.0, 1.0)
    bad = ~np.isfinite(pred) | ~(ff > 0)
    speed = np.where(bad, np.nan, speed)
    ci = np.where(bad, np.nan, ci)
    return {"intensity": np.where(bad, np.nan, intensity), "speed_kmh": speed, "ci": ci,
            "level": level_codes(ci)}
