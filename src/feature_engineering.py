"""
Feature construction shared by TRAINING and INFERENCE (single code path => no train/serve skew).

Features for target (hour t, segment s) - every input is >= 24 h older than t, so no
target leakage and no real-time feed is needed:
  hour, day_of_week, is_weekend, is_peak_hour    calendar (peak = hours 8-11 and 17-21)
  speed_limit, frc, distance_km                  static road attributes
  lag24    probe_count(s, t-24h)
  roll3    mean probe_count(s, t-24h, t-25h, t-26h)        (precomputed in the panel)
  seg_mean mean probe_count of s over the TRAIN period only  (fixed: the original version
           used the whole period incl. test days -> leakage)
Optional weather columns (only for the *_weather model variants).
"""
import numpy as np
import pandas as pd

BASE_FEATURES = ["hour", "day_of_week", "is_weekend", "is_peak_hour", "speed_limit", "frc",
                 "distance_km", "probe_count_lag_24h", "probe_count_roll3_mean", "probe_count_seg_mean"]
WEATHER_FEATURES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
PEAK_HOURS = set(range(8, 12)) | set(range(17, 22))


class FeatureBuilder:
    def __init__(self, panel, static_df, seg_mean, weather=None):
        """static_df rows must follow panel.seg_ids order. weather: DataFrame indexed by hourly
        timestamp with WEATHER_FEATURES columns (or None)."""
        if not np.array_equal(static_df["segment_id"].to_numpy(), panel.seg_ids):
            raise ValueError("static table order does not match panel segment order")
        self.p = panel
        self.speed_limit = static_df["speed_limit"].to_numpy(np.float32)
        self.frc = static_df["frc"].to_numpy(np.float32)
        self.dist = static_df["distance_km"].to_numpy(np.float32)
        self.seg_mean = np.asarray(seg_mean, dtype=np.float32)
        self.weather = None
        if weather is not None:
            w = weather.reindex(panel.timestamps)
            if w[WEATHER_FEATURES].isna().any().any():
                raise ValueError("weather history does not cover every dataset hour")
            self.weather = w[WEATHER_FEATURES].to_numpy(np.float32)

    def rows(self, t_idx, s_idx, with_weather=False) -> pd.DataFrame:
        t_idx, s_idx = np.asarray(t_idx), np.asarray(s_idx)
        ts = self.p.timestamps[t_idx]
        hour = ts.hour.to_numpy()
        dow = ts.dayofweek.to_numpy()
        d = {
            "hour": hour.astype(np.float32), "day_of_week": dow.astype(np.float32),
            "is_weekend": (dow >= 5).astype(np.float32),
            "is_peak_hour": np.isin(hour, list(PEAK_HOURS)).astype(np.float32),
            "speed_limit": self.speed_limit[s_idx], "frc": self.frc[s_idx], "distance_km": self.dist[s_idx],
            "probe_count_lag_24h": self.p.lag24[t_idx, s_idx],
            "probe_count_roll3_mean": self.p.roll3[t_idx, s_idx],
            "probe_count_seg_mean": self.seg_mean[s_idx],
        }
        if with_weather:
            if self.weather is None:
                raise ValueError("weather features requested but no weather history loaded")
            for j, c in enumerate(WEATHER_FEATURES):
                d[c] = self.weather[t_idx, j]
        return pd.DataFrame(d)

    def hour_snapshot(self, t: int, with_weather=False) -> pd.DataFrame:
        n = self.p.N
        return self.rows(np.full(n, t), np.arange(n), with_weather)

    def block(self, t_lo, t_hi, with_weather=False):
        """All segments for hours [t_lo, t_hi). Returns (X, y)."""
        n = self.p.N
        t = np.repeat(np.arange(t_lo, t_hi), n)
        s = np.tile(np.arange(n), t_hi - t_lo)
        return self.rows(t, s, with_weather), self.p.target[t_lo:t_hi].reshape(-1)


def features_for(with_weather: bool):
    return BASE_FEATURES + (WEATHER_FEATURES if with_weather else [])
