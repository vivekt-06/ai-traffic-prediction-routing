"""Dataset validation + quality statistics, computed from the actual files (never hard-coded)."""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, dataset_config
from src.panel import Panel, SEGMENTS_PATH

STATUS_PATH = PROCESSED_DIR / "dataset_status.json"


def validate_mapping(columns, mapping: dict) -> list:
    """Return the list of mapped columns missing from a dataframe's column list."""
    return [v for v in mapping.values() if v not in set(columns)]


def compute_status() -> dict:
    cfg = dataset_config()
    cols = cfg["columns"]
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet",
                           columns=[cols["segment_id"], cols["timestamp"], cols["target"]])
    seg = pd.read_parquet(SEGMENTS_PATH, columns=list(cfg["static_columns"].values())[:-1])
    miss_map = validate_mapping(feat.columns, {k: cols[k] for k in ("segment_id", "timestamp", "target")})
    t = feat[cols["timestamp"]]
    y = feat[cols["target"]]
    dup = int(feat.duplicated(subset=[cols["segment_id"], cols["timestamp"]]).sum())
    per_seg = feat.groupby(cols["segment_id"]).size()
    expected_hours = int(t.nunique())
    sl = seg[cfg["static_columns"]["speed_limit"]]
    v = cfg["validation"]
    p = Panel()
    status = {
        "name": cfg["name"], "target_kind": cfg["target_kind"], "resolution_minutes": cfg["resolution_minutes"],
        "period_start": str(t.min()), "period_end": str(t.max()), "hours": expected_hours,
        "days": round(expected_hours / 24, 2),
        "records": int(len(feat)), "segments": int(seg.shape[0]),
        "quality": {
            "missing_target": int(y.isna().sum()), "duplicate_keys": dup,
            "segments_with_incomplete_history": int((per_seg != expected_hours).sum()),
            "negative_target": int((y < 0).sum()), "above_plausible_max": int((y > v["max_probe_count"]).sum()),
            "zero_target_pct": round(float((y == 0).mean() * 100), 2),
            "target_mean": round(float(y.mean()), 2), "target_p50": float(y.median()),
            "target_p99": float(y.quantile(0.99)), "target_max": int(y.max()),
            "speed_limit_out_of_range": int(((sl < v["min_speed_limit"]) | (sl > v["max_speed_limit"])).sum()),
            "zero_length_segments": int((seg[cfg["static_columns"]["distance_km"]] <= 0).sum()),
        },
        "coverage_bbox": {"west": float(seg.start_lon.min()), "east": float(seg.start_lon.max()),
                          "south": float(seg.start_lat.min()), "north": float(seg.start_lat.max())},
        "splits": {k: {"start": str(p.timestamps[a]), "end": str(p.timestamps[b - 1]), "hours": b - a}
                   for k, (a, b) in p.split_bounds().items()},
        "mapping_missing_columns": miss_map,
        "limitations": [
            "Target is hourly probe COUNT (traffic-volume proxy). The dataset contains no measured per-segment speed.",
            "Hourly resolution: 15- and 30-minute forecast horizons are NOT supported by this data.",
            f"Only {round(expected_hours / 24)} days of data (one August window): no seasonality beyond weekly cycle can be learned.",
            "No one-way / turn-restriction attribute: segments are routed as bidirectional.",
        ],
    }
    return status


def write_status():
    s = compute_status()
    STATUS_PATH.write_text(json.dumps(s, indent=2))
    return s


def read_status():
    if not STATUS_PATH.exists():
        return write_status()
    return json.loads(STATUS_PATH.read_text())
