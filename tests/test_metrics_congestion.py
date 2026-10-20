import numpy as np
from src.metrics import regression_metrics
from src.congestion import compute_congestion, level_codes, classify_ci, LEVELS
from src.config import congestion_config


def test_metrics_basic_values():
    m = regression_metrics([10, 20, 30], [12, 18, 33])
    assert abs(m["MAE"] - (2 + 2 + 3) / 3) < 1e-9
    assert abs(m["RMSE"] - np.sqrt((4 + 4 + 9) / 3)) < 1e-9
    assert m["R2"] < 1 and m["mape_coverage"] == 1.0


def test_metrics_zero_targets_excluded_from_mape_but_reported():
    m = regression_metrics([0, 0, 10], [1, 2, 12])
    assert abs(m["MAPE"] - 20.0) < 1e-9            # only the non-zero row
    assert abs(m["mape_coverage"] - 1 / 3) < 1e-9
    assert m["MAE"] == (1 + 2 + 2) / 3


def test_metrics_undefined_cases_are_none_not_zero():
    assert regression_metrics([0, 0], [1, 1])["MAPE"] is None
    assert regression_metrics([5, 5, 5], [4, 5, 6])["R2"] is None      # constant truth
    assert regression_metrics([], [])["MAE"] is None


def test_metrics_nonfinite_dropped_and_counted():
    m = regression_metrics([1, 2, np.nan, 4], [1, 2, 3, np.inf])
    assert m["n"] == 2 and m["n_dropped_nonfinite"] == 2


def test_all_four_levels_are_reachable_with_default_config():
    cfg = congestion_config()
    sat = np.full(5, 100.0); ff = np.full(5, 50.0)
    c = compute_congestion(np.array([0, 30, 55, 100, 150]), sat, ff)
    assert set(c["level"].tolist()) == {0, 1, 2, 3}, "SEVERE must be reachable (old K=0.5 made CI<=0.5)"
    assert (c["ci"] >= 0).all() and (c["ci"] <= 1).all()
    assert (c["speed_kmh"] >= cfg["min_speed_kmh"]).all() and (c["speed_kmh"] <= ff).all()


def test_ci_monotonic_in_load_and_formula():
    cfg = congestion_config()
    pred = np.linspace(0, 200, 50)
    c = compute_congestion(pred, np.full(50, 100.0), np.full(50, 60.0))
    assert (np.diff(c["ci"]) >= -1e-12).all()
    # CI = 1 - speed/free_flow
    assert np.allclose(c["ci"], 1 - c["speed_kmh"] / 60.0)
    assert abs(c["ci"][0]) < 1e-12 and abs(c["ci"][-1] - cfg["k_speed_loss"] * cfg["intensity_cap"]) < 1e-9


def test_threshold_boundaries():
    t = congestion_config()["thresholds"]
    assert classify_ci(t["LOW"] - 1e-6) == "LOW" and classify_ci(t["LOW"]) == "MODERATE"
    assert classify_ci(t["MODERATE"]) == "HIGH" and classify_ci(t["HIGH"]) == "SEVERE"
    assert classify_ci(1.0) == "SEVERE"


def test_invalid_values_become_nan_not_zero():
    c = compute_congestion(np.array([np.nan, 10.0, 10.0]), np.array([100.0, 100.0, 100.0]), np.array([50.0, 0.0, 50.0]))
    assert np.isnan(c["ci"][0]) and np.isnan(c["ci"][1]) and np.isfinite(c["ci"][2])
    assert level_codes(c["ci"]).tolist()[:2] == [-1, -1]


def test_zero_saturation_reference_is_safe():
    c = compute_congestion(np.array([5.0]), np.array([0.0]), np.array([50.0]))
    assert np.isfinite(c["ci"][0])
