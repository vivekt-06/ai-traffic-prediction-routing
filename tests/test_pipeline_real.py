"""Tests against the real processed data + trained artifacts. Skipped if they are absent."""
import json
import numpy as np, pandas as pd, pytest
from pathlib import Path
from src.config import MODEL_DIR, PROCESSED_DIR

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "panel.npz").exists() or not (MODEL_DIR / "metrics.json").exists(),
                                reason="run scripts/build_panel.py and scripts/train_model.py first")


@pytest.fixture(scope="module")
def panel():
    from src.panel import Panel
    return Panel()


def test_dataset_mapping_validation():
    from src.dataset_quality import validate_mapping
    assert validate_mapping(["a", "b"], {"x": "a", "y": "c"}) == ["c"]
    from src.config import dataset_config
    # mapped columns all exist in the real processed table
    import pyarrow.parquet as pq
    names = pq.ParquetFile(PROCESSED_DIR / "features.parquet").schema.names
    assert validate_mapping(names, dataset_config()["columns"]) == []


def test_panel_complete_and_lag_consistent(panel):
    assert not np.isnan(panel.target).any() and panel.timestamps.is_monotonic_increasing
    assert (np.diff(panel.timestamps.asi8) == 3600 * 10**9).all(), "hourly, gapless"
    assert np.array_equal(panel.lag24[24:], panel.target[:-24])


def test_chronological_splits_are_disjoint_ordered_and_cover(panel):
    b = panel.split_bounds()
    assert b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0] and b["test"][1] == panel.T
    assert panel.timestamps[b["train"][1] - 1] < panel.timestamps[b["val"][0]] < panel.timestamps[b["test"][0]]
    assert panel.split_of(0) == "train" and panel.split_of(panel.T - 1) == "test"


def test_segment_mean_feature_uses_train_period_only(panel):
    stats = pd.read_parquet(MODEL_DIR / "segment_stats.parquet")
    lo, hi = panel.split_bounds()["train"]
    assert np.allclose(stats["seg_mean_train"].to_numpy(), panel.target[lo:hi].mean(0), rtol=1e-5)
    assert not np.allclose(stats["seg_mean_train"].to_numpy(), panel.target.mean(0), rtol=1e-4), "must differ from full-period mean"


def test_training_and_inference_features_identical(panel):
    from src.forecast import get_forecaster
    fc = get_forecaster(); fb = fc.fb
    t, rng = 400, np.random.default_rng(0)
    segs = rng.choice(panel.N, 50, replace=False)
    a = fb.rows(np.full(50, t), segs)
    b = fb.hour_snapshot(t).iloc[segs].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert (a["probe_count_lag_24h"].to_numpy() == panel.target[t - 24, segs]).all()


def test_no_target_leakage_all_features_older_than_target(panel):
    # lag/roll features at hour t only reference hours <= t-24
    t = 200
    ref = np.mean([panel.target[t - 24], panel.target[t - 25], panel.target[t - 26]], axis=0)
    assert np.allclose(panel.roll3[t], ref, atol=1e-3)


def test_saved_metrics_are_real_and_reproducible():
    from src.forecast import get_forecaster
    from src.metrics import regression_metrics
    fc = get_forecaster()
    doc = json.loads((MODEL_DIR / "metrics.json").read_text())
    lo, hi = fc.panel.split_bounds()["test"]
    # Recompute the Historical Average test MAE independently from the saved table.
    hrs = fc.panel.timestamps[lo:hi].hour.to_numpy()
    pred = fc.hist[hrs]
    m = regression_metrics(fc.panel.target[lo:hi], pred)
    assert abs(m["MAE"] - doc["models"]["historical_average"]["test"]["MAE"]) < 1e-3
    for k, v in doc["models"].items():
        assert v["status"] == "trained" and v["test"]["MAE"] is not None


def test_model_inference_loads_saved_artifacts_and_is_deterministic():
    from src.forecast import get_forecaster
    fc = get_forecaster()
    for m in fc.servable_models():
        a = fc.predict_probe(380, m); b = fc.predict_probe(380, m)
        assert a.shape == (fc.panel.N,) and np.isfinite(a).all() and (a >= 0).all() and np.array_equal(a, b)


def test_unavailable_models_are_not_servable():
    from src.forecast import get_forecaster, ForecastUnavailable
    fc = get_forecaster()
    reg = {r["key"]: r for r in fc.registry()}
    for k in ("lstm", "gru", "st_gcn_gru"):
        if not (MODEL_DIR / f"{k}.pt").exists():
            assert reg[k]["status"] == "untrained" and not reg[k]["servable"] and reg[k]["reason"]
            with pytest.raises(ForecastUnavailable):
                fc.predict_probe(380, k)


def test_every_registry_entry_has_a_real_status():
    from src.forecast import get_forecaster
    reg = get_forecaster().registry()
    assert {r["status"] for r in reg} <= {"ready", "untrained", "error"}
    for r in reg:
        assert r["servable"] == (r["status"] == "ready")
        assert r["status_detail"]
        if r["status"] == "ready":
            assert r["metrics"] is not None and r["checks"] and all(c["ok"] for c in r["checks"])
        else:
            assert r["metrics"] is None, "a non-ready model must not present metrics as its own"
