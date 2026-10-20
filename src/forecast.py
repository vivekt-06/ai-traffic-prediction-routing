"""
Forecast service: loads saved model artifacts (never retrains) and answers
"what is the predicted probe count / speed / congestion of every segment at hour t?".

Deep-model forecasts are read from the prediction arrays written by scripts/train_deep.py
(models/trained/deep_pred_<name>.npy) so the web app needs no PyTorch at runtime.
"""
import json
import logging
from functools import lru_cache
from threading import Lock, RLock

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, CONTEXT_DIR, congestion_config
from src.congestion import compute_congestion
from src import model_status as ms
from src.feature_engineering import FeatureBuilder, features_for
from src.panel import Panel
from src.training import MODEL_CATALOG, load_static, WEATHER_PATH

log = logging.getLogger("forecast")


class ForecastUnavailable(Exception):
    pass


class Forecaster:
    def __init__(self):
        self.panel = Panel()
        self.static = load_static(self.panel)
        stats_path = MODEL_DIR / "segment_stats.parquet"
        if not stats_path.exists():
            raise FileNotFoundError("models/trained/segment_stats.parquet missing - run: python scripts/train_model.py")
        self.stats = pd.read_parquet(stats_path).set_index("segment_id").loc[self.panel.seg_ids]
        self.metrics = json.loads((MODEL_DIR / "metrics.json").read_text()) if (MODEL_DIR / "metrics.json").exists() else {"models": {}}
        weather = pd.read_parquet(WEATHER_PATH) if WEATHER_PATH.exists() else None
        self.fb = FeatureBuilder(self.panel, self.static, self.stats["seg_mean_train"].to_numpy(), weather)
        self.hist = np.load(MODEL_DIR / "historical_average.npy") if (MODEL_DIR / "historical_average.npy").exists() else None
        self.sat = self.stats["saturation_ref"].to_numpy()
        self.free_flow = self.static["speed_limit"].to_numpy(np.float64)
        self._models, self._deep, self._lock = {}, {}, Lock()
        self._cache = {}
        self._status, self._status_lock = None, RLock()

    # ---- model registry -----------------------------------------------------------
    def statuses(self, refresh: bool = False) -> dict:
        """{model_key: {status: ready|untrained|error, detail, checks}} - computed once by really loading each
        model and running a held-out validation prediction (src/model_status.py); cached until refresh=True."""
        with self._status_lock:
            if self._status is None or refresh:
                self._status = {}
                with self._lock:
                    self._models.clear(); self._deep.clear()
                self._cache.clear()
                for key in MODEL_CATALOG:
                    try:
                        self._status[key] = self._check_model(key)
                    except Exception as e:                      # a checker bug must never take the API down
                        log.exception("status check crashed for %s", key)
                        self._status[key] = ms.result(ms.ERROR, f"status check crashed: {type(e).__name__}: {e}")
                    st = self._status[key]
                    log.info("model %-22s %-9s %s", key, st["status"], st["detail"])
            return self._status

    def _check_model(self, key) -> dict:
        entry = self.metrics.get("models", {}).get(key)
        b = self.panel.split_bounds()
        if key == "historical_average":
            res = ms.check_historical_average(self.hist, self.panel, b["test"], (entry or {}).get("test", {}).get("MAE"))
            if entry is None and res["status"] == ms.READY:
                return ms.result(ms.ERROR, "table exists but metrics.json has no evaluation for it; re-run scripts/train_model.py", res["checks"])
            return res
        if key in ms.DEEP_KEYS:
            return ms.check_deep_model(key, MODEL_DIR, self.panel, entry)
        wx = key.endswith("_weather")
        files = [MODEL_DIR / f"{key}.{ext}" for ext in ("joblib", "json")]
        have = [f for f in files if f.exists()]
        if not have and not entry:
            return ms.result(ms.UNTRAINED, self._why_missing(key))
        if not have:
            return ms.result(ms.ERROR, f"metrics.json lists '{key}' as trained but its weights file is missing in models/trained/")
        if not entry:
            return ms.result(ms.ERROR, "weights exist but metrics.json has no evaluation for this model; re-run scripts/train_model.py")
        if wx and self.fb.weather is None:
            return ms.result(ms.ERROR, "weather model needs data/contextual/weather_history.parquet (run scripts/fetch_weather_history.py)")
        hour = b["test"][0] + 12                                   # 12:00 of the first held-out day
        return ms.check_tabular_model(key, lambda: self._load(key),
                                      lambda h: self.fb.hour_snapshot(h, wx)[features_for(wx)], self.panel, hour)

    def _mark_error(self, key, msg):
        with self._status_lock:
            if self._status is not None:
                self._status[key] = ms.result(ms.ERROR, msg, self._status.get(key, {}).get("checks"))
        log.error("model %s marked ERROR at runtime: %s", key, msg)

    def registry(self) -> list:
        st, out = self.statuses(), []
        for key, (label, role, desc) in MODEL_CATALOG.items():
            m, s = self.metrics.get("models", {}).get(key), st[key]
            entry = {"key": key, "label": label, "role": role, "description": desc, "status": s["status"],
                     "status_detail": s["detail"], "checks": s["checks"], "servable": s["status"] == ms.READY,
                     "metrics": m if s["status"] == ms.READY else None, "stored_metrics": m}
            if s["status"] != ms.READY:
                entry["reason"] = s["detail"]
            out.append(entry)
        return out

    @staticmethod
    def _why_missing(key):
        if key in ms.DEEP_KEYS:
            return (f"Not trained: no weights in models/trained/. Run on a GPU machine: "
                    f"python scripts/train_deep.py --models {key} --epochs 30 (see README).")
        if key.endswith("_weather"):
            return "Not trained: requires scripts/fetch_weather_history.py (internet), then train_model.py --with-weather."
        return "Not trained: run python scripts/train_model.py."

    def servable_models(self):
        return [r["key"] for r in self.registry() if r["servable"]]

    def default_model(self):
        d = self.metrics.get("default_model")
        if d not in self.servable_models():
            s = self.servable_models()
            if not s:
                raise ForecastUnavailable("No trained model available. Run: python scripts/train_model.py")
            d = s[0]
        return d

    # ---- inference -------------------------------------------------------------------
    def _load(self, key):
        with self._lock:
            if key in self._models:
                return self._models[key]
            if key.startswith("xgboost"):
                from xgboost import XGBRegressor
                m = XGBRegressor(); m.load_model(str(MODEL_DIR / f"{key}.json"))
            else:
                m = joblib.load(MODEL_DIR / f"{key}.joblib")
            self._models[key] = m
            return m

    def predict_probe(self, hour_idx: int, model: str) -> np.ndarray:
        """Predicted probe count for every segment at dataset hour index `hour_idx` (cached)."""
        if not 0 <= hour_idx < self.panel.T:
            raise ForecastUnavailable(f"hour index {hour_idx} outside dataset coverage")
        if model not in self.servable_models():
            raise ForecastUnavailable(f"model '{model}' is not available (trained + artifact present)")
        k = (hour_idx, model)
        if k in self._cache:
            return self._cache[k]
        if model == "historical_average":
            p = self.hist[self.panel.timestamps[hour_idx].hour]
        elif model in ("lstm", "gru", "st_gcn_gru"):
            if model not in self._deep:
                self._deep[model] = np.load(MODEL_DIR / f"deep_pred_{model}.npy", mmap_mode="r")
            p = np.asarray(self._deep[model][hour_idx], dtype=np.float64)
            if not np.isfinite(p).all():
                raise ForecastUnavailable(f"{model} has no forecast for {self.panel.timestamps[hour_idx]} "
                                          "(needs a full 47 h history window before the target hour)")
        else:
            wx = model.endswith("_weather")
            try:
                X = self.fb.hour_snapshot(hour_idx, wx)[features_for(wx)]
                p = self._load(model).predict(X)
            except Exception as e:
                self._mark_error(model, f"inference failed: {type(e).__name__}: {e}")
                raise ForecastUnavailable(f"model '{model}' failed during inference ({type(e).__name__}: {e})") from e
        p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
        if len(self._cache) > 96:
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = p
        return p

    def congestion(self, hour_idx: int, model: str) -> dict:
        k = ("c", hour_idx, model)
        if k not in self._cache:
            pred = self.predict_probe(hour_idx, model)
            c = compute_congestion(pred, self.sat, self.free_flow)
            c["pred"] = pred
            c["speed_list"] = c["speed_kmh"].tolist()
            self._cache[k] = c
        return self._cache[k]

    def split_label(self, hour_idx):
        return self.panel.split_of(hour_idx)


@lru_cache(maxsize=1)
def get_forecaster() -> Forecaster:
    return Forecaster()


def _segment_series(self, seg_idx: int, model: str) -> np.ndarray:
    """Predicted probe count for ONE segment at every dataset hour (cheap: T rows)."""
    if model not in self.servable_models():
        raise ForecastUnavailable(f"model '{model}' is not available")
    T = self.panel.T
    if model == "historical_average":
        return self.hist[self.panel.timestamps.hour.to_numpy(), seg_idx].astype(np.float64)
    if model in ("lstm", "gru", "st_gcn_gru"):
        if model not in self._deep:
            self._deep[model] = np.load(MODEL_DIR / f"deep_pred_{model}.npy", mmap_mode="r")
        return np.clip(np.asarray(self._deep[model][:, seg_idx], dtype=np.float64), 0, None)
    wx = model.endswith("_weather")
    X = self.fb.rows(np.arange(T), np.full(T, seg_idx), wx)[features_for(wx)]
    return np.clip(self._load(model).predict(X), 0, None)


Forecaster.segment_series = _segment_series
