"""Evaluation metrics computed from held-out predictions. Handles zeros / NaNs explicitly."""
import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n_total, y_true, y_pred = len(y_true), y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"n": 0, "MAE": None, "RMSE": None, "MAPE": None, "R2": None, "WAPE": None, "mape_coverage": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None   # undefined for constant truth
    nz = y_true > 0                                                       # MAPE undefined at 0
    mape = float(np.mean(np.abs(err[nz]) / y_true[nz]) * 100) if nz.any() else None
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom * 100) if denom > 0 else None
    return {"n": int(len(y_true)), "n_dropped_nonfinite": int(n_total - len(y_true)),
            "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "WAPE": wape,
            "mape_coverage": float(nz.mean())}
