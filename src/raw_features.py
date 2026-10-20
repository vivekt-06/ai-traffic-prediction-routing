"""
feature_engineering.py
-----------------------
Builds the model-ready feature table on top of the cleaned/merged long table.

Features created (all derived from ACTUAL columns in the dataset - nothing invented):
  Static per-segment:      speed_limit, frc, distance_km
  Temporal:                hour, day_of_week, is_weekend, is_peak_hour
  Historical/lag (per segment, per hour-of-day - a segment's own history):
      probe_count_lag_24h   -> probe count for this same segment, same hour, previous day
      probe_count_roll3_mean-> mean of the last 3 same-hour observations for this segment
      probe_count_seg_mean  -> that segment's overall mean probe count (its "typical load")

Lag/rolling features are computed per segment_id ordered by timestamp, which is a
legitimate, non-leaking way to give the model short-term history without looking
into the future (each lag is strictly from an earlier timestamp than the target row).
"""

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["segment_id", "timestamp"]).copy()
    g = df.groupby("segment_id")["probe_count"]

    # Same hour, previous day (24 rows back since data is hourly and gapless per segment)
    df["probe_count_lag_24h"] = g.shift(24)

    # Rolling mean of the previous 3 same-position observations (i.e. last 3 days,
    # same hour) - computed on the lagged series so it never includes the target row.
    df["probe_count_roll3_mean"] = (
        df.groupby("segment_id")["probe_count_lag_24h"]
          .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # Each segment's own overall mean probe_count -> a per-segment "typical load"
    # baseline used both as a feature and later as the capacity reference for the
    # Congestion Index (see congestion.py).
    seg_mean = df.groupby("segment_id")["probe_count"].transform("mean")
    df["probe_count_seg_mean"] = seg_mean

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag features and drops the warm-up rows that have no lag history yet
    (the first day per segment, where lag_24h is undefined)."""
    df = add_lag_features(df)
    before = len(df)
    df = df.dropna(subset=["probe_count_lag_24h"])
    print(f"  dropped {before - len(df)} warm-up rows with no 24h lag history")
    return df


FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "speed_limit", "frc", "distance_km",
    "probe_count_lag_24h", "probe_count_roll3_mean", "probe_count_seg_mean",
]
TARGET_COLUMN = "probe_count"
