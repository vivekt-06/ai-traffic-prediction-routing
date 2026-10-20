"""
Panel store: the observed data as dense [hours x segments] arrays, built ONCE from
features.parquet (scripts/build_panel.py) so the API never loads 11M-row tables.

Hour 0 of the series = first hour in the dataset. The raw hour before that is not
stored; probe_count at (t-24) for the first day is recovered from lag24 of day 1,
giving a 24-hour pre-history (`prehistory`, shape [24, N]).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, dataset_config

PANEL_PATH = PROCESSED_DIR / "panel.npz"
SEGMENTS_PATH = PROCESSED_DIR / "segments_static.parquet"


def build_panel(features_path=None, segments_path=None, out_path=PANEL_PATH):
    cfg = dataset_config()["columns"]
    seg = pd.read_parquet(segments_path or SEGMENTS_PATH, columns=["segment_id"])
    seg_ids = seg["segment_id"].to_numpy()
    cols = [cfg["segment_id"], cfg["timestamp"], cfg["target"], cfg["lag24"], cfg["roll3"]]
    df = pd.read_parquet(features_path or PROCESSED_DIR / "features.parquet", columns=cols)
    df = df.rename(columns={v: k for k, v in cfg.items() if v in cols})
    ts = np.sort(df["timestamp"].unique())
    t_idx = np.searchsorted(ts, df["timestamp"].to_numpy())
    s_idx = pd.Index(seg_ids).get_indexer(df["segment_id"])
    if (s_idx < 0).any():
        raise ValueError("features contain segment_ids absent from segments_static")
    T, N = len(ts), len(seg_ids)
    out = {}
    for key, dtype in (("target", np.float32), ("lag24", np.float32), ("roll3", np.float32)):
        a = np.full((T, N), np.nan, dtype=dtype)
        a[t_idx, s_idx] = df[key].to_numpy(dtype)
        out[key] = a
    np.savez_compressed(out_path, seg_ids=seg_ids, timestamps=ts.astype("datetime64[ns]").astype(np.int64), **out)
    return T, N


class Panel:
    def __init__(self, path=PANEL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing - run: python scripts/build_panel.py")
        z = np.load(path)
        self.seg_ids = z["seg_ids"]
        self.timestamps = pd.to_datetime(z["timestamps"])
        self.target, self.lag24, self.roll3 = z["target"], z["lag24"], z["roll3"]
        self.T, self.N = self.target.shape
        self.prehistory = self.lag24[:24]              # target at hours -24..-1

    def index_of(self, ts) -> int:
        ts = pd.Timestamp(ts).floor("h")
        i = self.timestamps.searchsorted(ts)
        if i >= self.T or self.timestamps[i] != ts:
            raise KeyError(f"{ts} outside dataset coverage {self.timestamps[0]} .. {self.timestamps[-1]}")
        return int(i)

    def full_series(self) -> np.ndarray:
        """[24 + T, N] observed probe counts including the recovered pre-history day."""
        return np.concatenate([self.prehistory, self.target], axis=0)

    def split_bounds(self) -> dict:
        sp = dataset_config()["splits"]
        a = int(self.timestamps.searchsorted(pd.Timestamp(sp["train_end"])))
        b = int(self.timestamps.searchsorted(pd.Timestamp(sp["val_end"])))
        return {"train": (0, a), "val": (a, b), "test": (b, self.T)}

    def split_of(self, i: int) -> str:
        for name, (lo, hi) in self.split_bounds().items():
            if lo <= i < hi:
                return name
        return "unknown"
