"""
Model readiness checks. A model is reported READY only if

  1. its artifacts exist,
  2. they load (or, for deep models, are structurally consistent with the current panel),
  3. a held-out sanity prediction passes (finite, non-negative, right shape, not constant, and better
     than a constant predictor on a real held-out hour / consistent with the stored test metrics).

Statuses:   "ready"      weights present + loaded + validation passed
            "untrained"  no weights / no metrics for this model (nothing was ever trained)
            "error"      artifacts exist (or metrics claim training) but loading / validation failed

Nothing here fabricates numbers: every check either compares against real data in panel.npz or reports why it
could not run.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

READY, UNTRAINED, ERROR = "ready", "untrained", "error"
DEEP_KEYS = ("lstm", "gru", "st_gcn_gru")
MIN_DEEP_HOUR = 23          # first dataset hour with a complete 24 h input window (see src/deep_data.py)


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def result(status: str, detail: str, checks=None) -> dict:
    return {"status": status, "detail": detail, "checks": checks or []}


def panel_fingerprint(seg_ids, n_hours: int) -> str:
    """Identifies the exact panel (segment order + length) a deep model was trained on."""
    h = hashlib.sha1(np.asarray(seg_ids, dtype=np.int64).tobytes())
    h.update(str(int(n_hours)).encode())
    return h.hexdigest()[:16]


def validate_prediction(p, n_segments: int, max_value: float = 1e6):
    """Return an error string if a per-segment prediction vector is unusable, else None."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n_segments,):
        return f"prediction shape {p.shape}, expected ({n_segments},)"
    if not np.isfinite(p).all():
        return f"{int((~np.isfinite(p)).sum())} non-finite predictions"
    if (p < -1e-6).any():
        return "negative predictions (probe counts cannot be negative)"
    if p.max() > max_value:
        return f"implausible prediction max {p.max():.3g}"
    if np.ptp(p) < 1e-9:
        return "prediction is constant across all segments (model output is degenerate)"
    return None


def beats_constant(pred, actual) -> tuple:
    """Held-out sanity: MAE must beat predicting the hour's mean count for every segment."""
    pred, actual = np.asarray(pred, np.float64), np.asarray(actual, np.float64)
    mae = float(np.mean(np.abs(pred - actual)))
    base = float(np.mean(np.abs(actual - actual.mean())))
    return mae < base, mae, base


# --------------------------------------------------------------------------- historical average
def check_historical_average(hist, panel, test_range, stored_test_mae, rel_tol=1e-3) -> dict:
    cks = []
    if hist is None:
        return result(UNTRAINED, "historical_average.npy not found. Run: python scripts/train_model.py")
    hist = np.asarray(hist)
    if hist.shape != (24, panel.N):
        return result(ERROR, f"historical_average.npy has shape {hist.shape}, expected (24, {panel.N}); "
                             "it was built for a different segment set. Re-run scripts/train_model.py")
    if not np.isfinite(hist).all() or (hist < 0).any():
        return result(ERROR, "historical_average.npy contains non-finite or negative values")
    cks.append(check("table loads", True, f"shape {hist.shape}"))
    lo, hi = test_range
    pred = hist[panel.timestamps[lo:hi].hour.to_numpy()]
    mae = float(np.mean(np.abs(pred - panel.target[lo:hi])))
    cks.append(check("held-out prediction finite", True, f"test MAE recomputed = {mae:.3f}"))
    if stored_test_mae is not None:
        same = abs(mae - stored_test_mae) <= rel_tol * max(1.0, stored_test_mae)
        cks.append(check("matches stored test MAE", same, f"recomputed {mae:.3f} vs stored {stored_test_mae:.3f}"))
        if not same:
            return result(ERROR, f"artifact does not reproduce the stored test MAE ({mae:.3f} vs {stored_test_mae:.3f}); "
                                 "metrics.json and weights are out of sync. Re-run scripts/train_model.py", cks)
    return result(READY, "table loaded; held-out predictions reproduce the stored test MAE", cks)


# --------------------------------------------------------------------------- sklearn / xgboost
def check_tabular_model(key, loader, snapshot_fn, panel, hour_idx) -> dict:
    """loader() -> fitted model; snapshot_fn(hour_idx) -> feature frame for that dataset hour (all segments)."""
    cks = []
    try:
        model = loader()
        cks.append(check("weights load", True))
    except Exception as e:
        return result(ERROR, f"weights failed to load: {type(e).__name__}: {e}")
    try:
        p = np.clip(np.asarray(model.predict(snapshot_fn(hour_idx)), dtype=np.float64), 0, None)
    except Exception as e:
        return result(ERROR, f"inference failed (incompatible checkpoint or feature mismatch): {type(e).__name__}: {e}", cks)
    bad = validate_prediction(p, panel.N)
    cks.append(check("prediction valid", bad is None, bad or f"{panel.N} segments, mean {p.mean():.1f}"))
    if bad:
        return result(ERROR, f"validation prediction rejected: {bad}", cks)
    ok, mae, base = beats_constant(p, panel.target[hour_idx])
    cks.append(check("beats constant predictor on a held-out hour", ok,
                     f"{panel.timestamps[hour_idx]}: MAE {mae:.1f} vs constant {base:.1f}"))
    if not ok:
        return result(ERROR, f"held-out sanity check failed (MAE {mae:.1f} is not better than constant {base:.1f})", cks)
    return result(READY, "weights loaded; held-out validation prediction passed", cks)


# --------------------------------------------------------------------------- deep models (served from precomputed predictions)
def check_deep_model(key, model_dir: Path, panel, metrics_entry, rel_tol=1e-3) -> dict:
    """Deep models are served from models/trained/deep_pred_<key>.npy (written by scripts/train_deep.py).
    Ready requires: weights file, config, prediction matrix, matching panel fingerprint and a prediction
    matrix that reproduces the stored test MAE."""
    model_dir = Path(model_dir)
    w, c, pr = model_dir / f"{key}.pt", model_dir / f"deep_config_{key}.json", model_dir / f"deep_pred_{key}.npy"
    present = [p.name for p in (w, c, pr) if p.exists()]
    if not present and not metrics_entry:
        return result(UNTRAINED, "No weights found. Train with: python scripts/train_deep.py --models "
                                 f"{key} --epochs 30  (GPU recommended; see README)")
    if metrics_entry and metrics_entry.get("smoke_run"):
        return result(UNTRAINED, "Only a smoke-test run exists (1 epoch, a few steps). Run the full training command.")
    missing = [p.name for p in (w, c, pr) if not p.exists()]
    if missing:
        return result(ERROR, "incomplete artifacts, missing: " + ", ".join(missing) +
                             ". Re-run scripts/train_deep.py or copy the whole models/trained/ folder back.")
    if not metrics_entry:
        return result(ERROR, "weights exist but metrics.json has no evaluation for this model. Re-run scripts/train_deep.py.")
    cks = []
    # 1. weights readable
    if not zipfile.is_zipfile(w):
        return result(ERROR, f"{w.name} is not a valid PyTorch checkpoint")
    try:
        cfg = json.loads(c.read_text())
    except Exception as e:
        return result(ERROR, f"{c.name} unreadable: {e}")
    cks.append(check("weights file + config present", True, f"{w.stat().st_size/1024:.0f} KB"))
    # optional strict load when torch happens to be installed (serving does not need it)
    try:
        import torch  # noqa: F401
        sd = torch.load(w, map_location="cpu", weights_only=True)
        shapes = {k: list(v.shape) for k, v in sd.items()}
        same = shapes == cfg.get("param_shapes")
        cks.append(check("state_dict matches config param shapes", same, f"{len(shapes)} tensors"))
        if not same:
            return result(ERROR, "checkpoint tensors do not match deep_config param_shapes (incompatible checkpoint)", cks)
    except ImportError:
        cks.append(check("state_dict load", True, "skipped: PyTorch not installed on this machine (not needed to serve)"))
    except Exception as e:
        return result(ERROR, f"checkpoint failed to load: {type(e).__name__}: {e}", cks)
    # 2. fingerprint
    fp = panel_fingerprint(panel.seg_ids, panel.T)
    if cfg.get("panel_fingerprint") != fp:
        return result(ERROR, "model was trained on a different panel (segment set / length changed). Retrain.", cks)
    cks.append(check("panel fingerprint matches", True, fp))
    # 3. prediction matrix
    try:
        pred = np.load(pr, mmap_mode="r")
    except Exception as e:
        return result(ERROR, f"{pr.name} unreadable: {e}", cks)
    if pred.shape != panel.target.shape:
        return result(ERROR, f"{pr.name} shape {pred.shape} != panel {panel.target.shape}", cks)
    body = np.asarray(pred[MIN_DEEP_HOUR:], dtype=np.float64)
    if not np.isfinite(body).all() or (body < -1e-6).any():
        return result(ERROR, "prediction matrix has non-finite or negative values after the first complete window", cks)
    cks.append(check("prediction matrix finite and non-negative", True))
    # 4. reproduces stored test metrics
    t0 = panel.split_bounds()["test"][0]
    lo, hi = max(t0, MIN_DEEP_HOUR), panel.T
    mae = float(np.mean(np.abs(np.asarray(pred[lo:hi], np.float64) - panel.target[lo:hi])))
    stored = metrics_entry.get("test", {}).get("MAE")
    if stored is None:
        return result(ERROR, "metrics.json lacks test MAE for this model", cks)
    same = abs(mae - stored) <= rel_tol * max(1.0, stored)
    cks.append(check("reproduces stored test MAE", same, f"recomputed {mae:.3f} vs stored {stored:.3f}"))
    if not same:
        return result(ERROR, f"predictions do not reproduce the stored test MAE ({mae:.3f} vs {stored:.3f}); artifacts are out of sync", cks)
    return result(READY, "checkpoint + predictions validated; test MAE reproduced from held-out predictions", cks)
