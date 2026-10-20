"""Model readiness checks: a model is READY only after loading + a passing held-out validation prediction."""
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import model_status as ms

N, T = 40, 24 * 6


def make_panel(seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-08-12", periods=T, freq="h")
    base = rng.uniform(50, 500, N)
    daily = 1 + 0.5 * np.sin(2 * np.pi * ts.hour.to_numpy() / 24)[:, None]
    target = (base[None, :] * daily + rng.normal(0, 5, (T, N))).clip(0).astype(np.float32)
    p = SimpleNamespace(N=N, T=T, seg_ids=np.arange(1000, 1000 + N), timestamps=ts, target=target)
    p.split_bounds = lambda: {"train": (0, 72), "val": (72, 108), "test": (108, T)}
    return p


def hist_table(p):
    return np.stack([p.target[:72][p.timestamps[:72].hour == h].mean(0) for h in range(24)]).astype(np.float32)


# ------------------------------------------------------------------ prediction validation
def test_validate_prediction_rejects_bad_vectors():
    ok = np.linspace(1, 100, N)
    assert ms.validate_prediction(ok, N) is None
    assert "shape" in ms.validate_prediction(ok[:-1], N)
    assert "non-finite" in ms.validate_prediction(np.where(np.arange(N) == 3, np.nan, ok), N)
    assert "negative" in ms.validate_prediction(ok - 50, N)
    assert "constant" in ms.validate_prediction(np.full(N, 7.0), N)
    assert "implausible" in ms.validate_prediction(ok * 1e9, N)


# ------------------------------------------------------------------ historical average
def test_historical_average_ready_only_when_it_reproduces_stored_mae():
    p = make_panel(); h = hist_table(p)
    lo, hi = p.split_bounds()["test"]
    true_mae = float(np.mean(np.abs(h[p.timestamps[lo:hi].hour] - p.target[lo:hi])))
    assert ms.check_historical_average(h, p, (lo, hi), true_mae)["status"] == ms.READY
    r = ms.check_historical_average(h, p, (lo, hi), true_mae * 2)
    assert r["status"] == ms.ERROR and "out of sync" in r["detail"]
    assert ms.check_historical_average(None, p, (lo, hi), None)["status"] == ms.UNTRAINED
    assert ms.check_historical_average(h[:, :-1], p, (lo, hi), None)["status"] == ms.ERROR


# ------------------------------------------------------------------ sklearn-style models
def _feat(hours, p):
    """seg_mean x daily profile: a legitimately informative feature, so a fitted LR should beat the constant baseline."""
    prof = 1 + 0.5 * np.sin(2 * np.pi * np.asarray(hours) / 24)
    return (p.target[:72].mean(0)[None, :] * prof[:, None]).ravel()


def fit_lr(p):
    X = _feat(p.timestamps.hour.to_numpy()[:72], p)[:, None]
    return LinearRegression().fit(X, p.target[:72].ravel())


def snapshot(p):
    return lambda h: _feat([p.timestamps[h].hour], p)[:, None]


def test_tabular_model_ready_when_loaded_and_validated():
    p = make_panel(); m = fit_lr(p)
    r = ms.check_tabular_model("lr", lambda: m, snapshot(p), p, 120)
    assert r["status"] == ms.READY and all(c["ok"] for c in r["checks"])


def test_tabular_model_error_paths():
    p = make_panel(); m = fit_lr(p)

    def boom():
        raise OSError("corrupt weights")
    r = ms.check_tabular_model("lr", boom, snapshot(p), p, 120)
    assert r["status"] == ms.ERROR and "failed to load" in r["detail"]
    # wrong feature count = incompatible checkpoint
    r = ms.check_tabular_model("lr", lambda: m, lambda h: np.zeros((N, 7)), p, 120)
    assert r["status"] == ms.ERROR and "inference failed" in r["detail"]
    # model that returns a constant -> degenerate
    const = SimpleNamespace(predict=lambda X: np.full(len(X), 100.0))
    assert ms.check_tabular_model("c", lambda: const, snapshot(p), p, 120)["status"] == ms.ERROR
    # model that is worse than predicting the hour's mean
    bad = SimpleNamespace(predict=lambda X: np.linspace(0, 5000, len(X))[::-1])
    r = ms.check_tabular_model("b", lambda: bad, snapshot(p), p, 120)
    assert r["status"] == ms.ERROR and "sanity" in r["detail"]
    nan = SimpleNamespace(predict=lambda X: np.full(len(X), np.nan))
    assert ms.check_tabular_model("n", lambda: nan, snapshot(p), p, 120)["status"] == ms.ERROR


# ------------------------------------------------------------------ deep models
def write_checkpoint(path):
    try:
        import torch
        torch.save({"w": torch.zeros(2, 3)}, path)
    except ImportError:                                   # structurally a zip, like torch.save output
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("archive/data.pkl", b"x")


def write_deep(d, p, key="gru", mae_scale=1.0, fingerprint=None):
    """Fake but internally consistent deep artifacts: predictions = actuals * 1.1 (so MAE is well defined)."""
    pred = (p.target * 1.1).astype(np.float32)
    pred[:ms.MIN_DEEP_HOUR] = np.nan
    np.save(d / f"deep_pred_{key}.npy", pred)
    write_checkpoint(d / f"{key}.pt")
    cfg = {"param_shapes": {"w": [2, 3]}, "panel_fingerprint": fingerprint or ms.panel_fingerprint(p.seg_ids, p.T)}
    (d / f"deep_config_{key}.json").write_text(json.dumps(cfg))
    lo = p.split_bounds()["test"][0]
    mae = float(np.mean(np.abs(pred[lo:].astype(np.float64) - p.target[lo:]))) * mae_scale
    return {"status": "trained", "servable": True, "test": {"MAE": mae}}


def test_deep_untrained_when_nothing_exists(tmp_path):
    r = ms.check_deep_model("lstm", tmp_path, make_panel(), None)
    assert r["status"] == ms.UNTRAINED and "train_deep.py" in r["detail"]


def test_deep_smoke_run_is_not_ready(tmp_path):
    p = make_panel(); entry = write_deep(tmp_path, p); entry["smoke_run"] = True
    assert ms.check_deep_model("gru", tmp_path, p, entry)["status"] == ms.UNTRAINED


def test_deep_ready_when_everything_is_consistent(tmp_path):
    p = make_panel(); entry = write_deep(tmp_path, p)
    r = ms.check_deep_model("gru", tmp_path, p, entry)
    assert r["status"] == ms.READY, r
    assert any(c["name"] == "reproduces stored test MAE" and c["ok"] for c in r["checks"])


def test_deep_error_cases(tmp_path):
    p = make_panel(); entry = write_deep(tmp_path, p)
    # stored metric does not match the predictions -> artifacts out of sync (never trust metrics.json blindly)
    bad = dict(entry, test={"MAE": entry["test"]["MAE"] * 1.5})
    assert ms.check_deep_model("gru", tmp_path, p, bad)["status"] == ms.ERROR
    # trained on another panel
    d2 = tmp_path / "other"; d2.mkdir()
    e2 = write_deep(d2, p, fingerprint="deadbeef")
    assert "different panel" in ms.check_deep_model("gru", d2, p, e2)["detail"]
    # weights missing but predictions + metrics remain
    (tmp_path / "gru.pt").unlink()
    r = ms.check_deep_model("gru", tmp_path, p, entry)
    assert r["status"] == ms.ERROR and "gru.pt" in r["detail"]


def test_deep_nan_predictions_are_rejected(tmp_path):
    p = make_panel(); entry = write_deep(tmp_path, p)
    pred = np.load(tmp_path / "deep_pred_gru.npy"); pred[50, 3] = np.nan
    np.save(tmp_path / "deep_pred_gru.npy", pred)
    assert ms.check_deep_model("gru", tmp_path, p, entry)["status"] == ms.ERROR
