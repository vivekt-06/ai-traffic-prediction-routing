"""
Training + evaluation for the classical / ensemble models (CPU). Deep models: scripts/train_deep.py.

Chronological split from configs/dataset.json (never shuffled across time):
  train  2024-08-12 .. 08-23   val 08-24 .. 08-26   test 08-27 .. 08-30
Per-segment statistics (seg_mean, saturation_ref, historical average) come from TRAIN only.
All metrics in metrics.json are computed from real held-out predictions made by this script.
"""
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import MODEL_DIR, PROCESSED_DIR, CONTEXT_DIR, congestion_config
from src.congestion import compute_congestion, level_codes
from src.feature_engineering import FeatureBuilder, features_for
from src.metrics import regression_metrics
from src.panel import Panel, SEGMENTS_PATH

MAX_TRAIN_ROWS = 400_000
WEATHER_PATH = CONTEXT_DIR / "weather_history.parquet"

MODEL_CATALOG = {
    "historical_average": ("Historical Average", "baseline", "Per-segment mean probe count for each hour of day (train period only)."),
    "linear_regression": ("Linear Regression", "baseline", "OLS on the shared feature set."),
    "random_forest": ("Random Forest", "primary", "Bagged regression trees (primary baseline in the synopsis)."),
    "xgboost": ("XGBoost", "extension", "Gradient-boosted trees, early-stopped on the validation split."),
    "lstm": ("LSTM", "primary", "Sequence model over the 24 h window ending 24 h before the target. Colab: scripts/train_deep.py"),
    "gru": ("GRU", "extension", "As LSTM with GRU cells. Colab: scripts/train_deep.py"),
    "st_gcn_gru": ("ST-GCN-GRU", "extension", "Graph convolution over road connectivity + GRU over time. Colab: scripts/train_deep.py"),
    "random_forest_weather": ("Random Forest + weather", "extension", "RF with Open-Meteo history features (needs scripts/fetch_weather_history.py, train with --with-weather)."),
    "xgboost_weather": ("XGBoost + weather", "extension", "XGBoost with Open-Meteo history features."),
}


def load_static(panel) -> pd.DataFrame:
    s = pd.read_parquet(SEGMENTS_PATH)
    s = s.set_index("segment_id").loc[panel.seg_ids].reset_index()
    return s


def train_segment_stats(panel, static) -> pd.DataFrame:
    lo, hi = panel.split_bounds()["train"]
    q = congestion_config()["saturation_quantile"]
    tr = panel.target[lo:hi]
    hours = panel.timestamps[lo:hi].hour.to_numpy()
    hist = np.stack([tr[hours == h].mean(0) for h in range(24)]).astype(np.float32)   # [24, N]
    np.save(MODEL_DIR / "historical_average.npy", hist)
    stats = pd.DataFrame({"segment_id": panel.seg_ids, "seg_mean_train": tr.mean(0),
                          "saturation_ref": np.quantile(tr, q, axis=0)})
    stats.to_parquet(MODEL_DIR / "segment_stats.parquet", index=False)
    return stats, hist


def evaluate_block(predict_fn, fb, panel, lo, hi, with_weather, chunk_hours=24):
    ys, ps, fb_rows = [], [], []
    for a in range(lo, hi, chunk_hours):
        b = min(hi, a + chunk_hours)
        X, y = fb.block(a, b, with_weather)
        ys.append(y); ps.append(np.clip(predict_fn(X), 0, None))
    y, p = np.concatenate(ys), np.concatenate(ps)
    m = regression_metrics(y, p)
    # Does the derived congestion LEVEL agree between predicted and observed counts?
    return m, y, p


def level_agreement(y, p, panel, stats, static, lo, hi):
    n_h = hi - lo
    sat = np.tile(stats["saturation_ref"].to_numpy(), n_h)
    ff = np.tile(static["speed_limit"].to_numpy(), n_h)
    la = compute_congestion(y, sat, ff)["level"]
    lp = compute_congestion(p, sat, ff)["level"]
    return float(np.mean(la == lp))


def run_training(with_weather=False, skip=(), seed=42, max_rows=MAX_TRAIN_ROWS):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = Panel()
    static = load_static(panel)
    stats, hist = train_segment_stats(panel, static)
    bounds = panel.split_bounds()
    weather = pd.read_parquet(WEATHER_PATH) if (with_weather and WEATHER_PATH.exists()) else None
    if with_weather and weather is None:
        raise SystemExit(f"--with-weather needs {WEATHER_PATH}; run scripts/fetch_weather_history.py first")
    fb = FeatureBuilder(panel, static, stats["seg_mean_train"].to_numpy(), weather)

    rng = np.random.default_rng(seed)
    (tlo, thi), (vlo, vhi), (slo, shi) = bounds["train"], bounds["val"], bounds["test"]
    flat = rng.choice((thi - tlo) * panel.N, size=min(max_rows, (thi - tlo) * panel.N), replace=False)
    t_s, s_s = tlo + flat // panel.N, flat % panel.N
    y_tr = panel.target[t_s, s_s]
    # Fixed val subsample (for early stopping only)
    fv = rng.choice((vhi - vlo) * panel.N, size=100_000, replace=False)
    Xv_es, yv_es = fb.rows(vlo + fv // panel.N, fv % panel.N, with_weather), panel.target[vlo + fv // panel.N, fv % panel.N]

    results = {}

    def record(name, predict_fn, fit_seconds, wx, extra=None):
        out = {"status": "trained", "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "fit_seconds": round(fit_seconds, 1), "uses_weather": wx, "servable": True}
        for split, (lo, hi) in (("val", (vlo, vhi)), ("test", (slo, shi))):
            m, y, p = evaluate_block(predict_fn, fb, panel, lo, hi, wx)
            out[split] = m
            if split == "test":
                out["test"]["congestion_level_agreement"] = level_agreement(y, p, panel, stats, static, lo, hi)
        out.update(extra or {})
        results[name] = out
        print(f"{name:24s} test MAE={out['test']['MAE']:.2f} RMSE={out['test']['RMSE']:.2f} "
              f"MAPE={out['test']['MAPE']:.1f}% R2={out['test']['R2']:.3f}", flush=True)

    # 1. Historical average (no fitting beyond the table already saved)
    hours_col = lambda X: X["hour"].to_numpy().astype(int)
    seg_col_cache = {}
    def ha_predict(X):
        # X rows are ordered (hour block, all segments): segment index = position mod N
        n = len(X) // 1
        s = np.tile(np.arange(panel.N), n // panel.N)
        return hist[hours_col(X), s]
    if "historical_average" not in skip:
        record("historical_average", ha_predict, 0.0, False, {"n_train_rows": int((thi - tlo) * panel.N)})

    feats = lambda wx: features_for(wx)
    wx_flags = [False] + ([True] if weather is not None else [])
    for wx in wx_flags:
        sfx = "_weather" if wx else ""
        X_tr = fb.rows(t_s, s_s, wx)
        cols = feats(wx)
        if not wx and "linear_regression" not in skip:
            t0 = time.time(); lr = LinearRegression().fit(X_tr[cols], y_tr)
            joblib.dump(lr, MODEL_DIR / "linear_regression.joblib")
            record("linear_regression", lambda X, m=lr: m.predict(X[cols]), time.time() - t0, wx,
                   {"n_train_rows": len(X_tr)})
        if f"random_forest{sfx}" not in skip:
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=seed)
            rf.fit(X_tr[cols], y_tr)
            joblib.dump(rf, MODEL_DIR / f"random_forest{sfx}.joblib", compress=3)
            record(f"random_forest{sfx}", lambda X, m=rf, c=cols: m.predict(X[c]), time.time() - t0, wx,
                   {"n_train_rows": len(X_tr)})
        if f"xgboost{sfx}" not in skip:
            from xgboost import XGBRegressor
            t0 = time.time()
            xg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=seed,
                              early_stopping_rounds=20, tree_method="hist")
            xg.fit(X_tr[cols], y_tr, eval_set=[(Xv_es[cols], yv_es)], verbose=False)
            xg.save_model(str(MODEL_DIR / f"xgboost{sfx}.json"))
            record(f"xgboost{sfx}", lambda X, m=xg, c=cols: m.predict(X[c]), time.time() - t0, wx,
                   {"n_train_rows": len(X_tr), "best_iteration": int(xg.best_iteration)})

    old = {}
    mp = MODEL_DIR / "metrics.json"
    if mp.exists():
        old = json.loads(mp.read_text()).get("models", {})
    merged = {**old, **results}                      # keep deep-model results from other runs
    ok = {k: v for k, v in merged.items() if v.get("status") == "trained" and v.get("servable")}
    default = min(ok, key=lambda k: ok[k]["val"]["MAE"]) if ok else None
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "probe_count (hourly)", "split": {k: [str(panel.timestamps[a]), str(panel.timestamps[b - 1])]
                                                     for k, (a, b) in bounds.items()},
        "split_rows": {k: int((b - a) * panel.N) for k, (a, b) in bounds.items()},
        "train_subsample_rows": int(len(t_s)), "default_model": default,
        "default_model_rule": "lowest validation MAE among servable trained models",
        "models": merged,
    }
    mp.write_text(json.dumps(doc, indent=2))
    print("default model (by validation MAE):", default)
    return doc
