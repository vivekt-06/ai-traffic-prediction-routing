"""
Framework-free (numpy/scipy) helpers for the deep models, so they can be unit-tested without PyTorch.

Window definition (identical for LSTM / GRU / ST-GCN-GRU, consistent with the tree models'
"inputs are >= 24 h old" rule): to predict hour t, the model sees the 24 hourly observations
of hours t-47 .. t-24. In `full_series()` coordinates (24 recovered pre-history hours first) that is
series[t-23 : t+1], so the newest input is exactly 24 h old and the target is never an input.
"""
import numpy as np
from scipy.sparse import csr_matrix, identity, diags

SEQ_LEN = 24
LAG = 24
MIN_TARGET_HOUR = SEQ_LEN - 1          # first dataset hour index with a complete window (= 23)


def window(series: np.ndarray, t: int) -> np.ndarray:
    """[SEQ_LEN, N] inputs for target hour index t."""
    if t < MIN_TARGET_HOUR:
        raise ValueError(f"hour {t} has no complete {SEQ_LEN}h window (needs t >= {MIN_TARGET_HOUR})")
    return series[t - (SEQ_LEN - 1): t + 1]


def window_hours(t: int):
    """Dataset hour indices (may be negative = pre-history) covered by the window for target t."""
    return list(range(t - LAG - SEQ_LEN + 1, t - LAG + 1))


def normalized_adjacency(adj: csr_matrix) -> csr_matrix:
    """GCN propagation matrix  D^-1/2 (A + I) D^-1/2  (Kipf & Welling)."""
    a = adj + identity(adj.shape[0], format="csr")
    d = np.asarray(a.sum(1)).ravel()
    dinv = diags(1.0 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr()


def time_features(timestamps) -> np.ndarray:
    """[T, 4]: hour sin, hour cos, day_of_week/6, is_weekend of the TARGET hour."""
    h, d = timestamps.hour.to_numpy(), timestamps.dayofweek.to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), d / 6.0, (d >= 5)], 1).astype(np.float32)
