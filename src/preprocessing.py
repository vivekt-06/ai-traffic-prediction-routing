"""
preprocessing.py
-----------------
Cleans the raw (segment, date, hour, probe_count) long table and merges in the
static per-segment attributes (speed_limit, frc, distance, geometry).

What real preprocessing is actually needed for THIS dataset:
  - The raw export has no missing rows (every segment has exactly 24 hourly entries
    every day - verified during inspection) and no duplicate (segment_id, date, hour)
    keys, since each is generated once per official HERE export. We still explicitly
    check for and drop duplicates / nulls defensively, because a prototype should not
    silently assume a clean file.
  - Timestamps are reconstructed from (date, hour) into a proper pandas datetime and
    the table is sorted chronologically (required before any time-based split).
  - day_of_week / is_weekend / peak-hour indicator are derived - real, useful temporal
    features given Delhi's clear weekday commute pattern.
"""

import pandas as pd
from pathlib import Path
from src.data_loader import load_all_probe_counts, load_segments_static

DAILY_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "_daily_cache"

# Delhi weekday morning/evening commute peaks (used only to build a peak-hour
# indicator feature - not used anywhere as a fabricated ground-truth value).
MORNING_PEAK = (8, 11)   # 8:00-10:59
EVENING_PEAK = (17, 21)  # 17:00-20:59


def clean_probe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive cleaning: drop exact duplicates, drop rows with nulls in key
    columns, and sort chronologically per segment."""
    before = len(df)
    df = df.dropna(subset=["segment_id", "date", "hour", "probe_count"])
    df = df.drop_duplicates(subset=["segment_id", "date", "hour"])
    after = len(df)
    if before != after:
        print(f"  dropped {before - after} null/duplicate rows")

    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds real, explainable temporal features derived from the timestamp."""
    df["day_of_week"] = df["timestamp"].dt.dayofweek       # 0=Mon .. 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_morning_peak"] = df["hour"].between(*MORNING_PEAK).astype("int8")
    df["is_evening_peak"] = df["hour"].between(*EVENING_PEAK).astype("int8")
    df["is_peak_hour"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype("int8")
    return df


def merge_static_attributes(probe_df: pd.DataFrame, seg_static: pd.DataFrame) -> pd.DataFrame:
    """Join only the NUMERIC static per-segment attributes (speed_limit, frc,
    distance_km) onto the hourly probe table. street_name/geometry are deliberately
    left out here - they don't help the model and would bloat a 12M-row table with
    a long duplicated string per row. They stay in segments_static.parquet, keyed
    by segment_id, for the map/graph-building code to join back in when needed."""
    small_cols = seg_static[["segment_id", "speed_limit", "frc", "distance_km"]].copy()
    small_cols["speed_limit"] = small_cols["speed_limit"].astype("int16")
    small_cols["frc"] = small_cols["frc"].astype("int8")
    small_cols["distance_km"] = small_cols["distance_km"].astype("float32")

    merged = probe_df.merge(small_cols, on="segment_id", how="left")
    missing = merged["speed_limit"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows could not be matched to a segment's static attributes")
    return merged


def run_preprocessing() -> pd.DataFrame:
    """Full preprocessing pipeline: load -> clean -> temporal features -> merge."""
    print("Loading static segment attributes...")
    seg_static = load_segments_static()

    print("Loading raw hourly probe counts (all 20 days)...")
    probes = load_all_probe_counts(cache_dir=DAILY_CACHE_DIR)

    print("Cleaning...")
    probes = clean_probe_table(probes)

    print("Adding temporal features...")
    probes = add_temporal_features(probes)

    print("Merging static segment attributes...")
    full = merge_static_attributes(probes, seg_static)

    return full, seg_static


if __name__ == "__main__":
    df, seg_static = run_preprocessing()
    print(df.head())
    print(df.dtypes)
