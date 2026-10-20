"""FastAPI backend. Run:  uvicorn api.main:app --port 8000   (see README)."""
import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src import events as events_svc
from src import weather as weather_svc
from src.config import congestion_config, DB_PATH, ROOT
from src.congestion import LEVELS, compute_congestion
from src.dataset_quality import read_status
from src.db import init_db, log_query
from src.departure import analyse
from src.forecast import ForecastUnavailable, get_forecaster
from src.graph import get_graph
from src.routing import compute_routes
from src.tdsp import NoRoute

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

def _json_safe(obj):
    """Replace NaN and Infinity with JSON-compatible null values."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj

@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield


def _startup():
    try:
        init_db()
        fc = get_forecaster(); get_graph()
        st = fc.statuses()                                   # loads every model + runs its validation prediction
        log.info("model status: %s", {k: v["status"] for k, v in st.items()})
        try:
            log.info("backend ready (default model: %s)", fc.default_model())
        except ForecastUnavailable as e:                     # API stays up so /api/models can show WHY nothing is ready
            log.warning("backend up but no model is ready: %s", e)
    except Exception as e:                         # keep the server up so /api/health can explain
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.exception("startup failed")


def _need_ready():
    if STATE["error"]:
        raise HTTPException(503, f"Backend not ready - {STATE['error']}. See README section 'Run order'.")


app = FastAPI(title="Traffic Congestion Prediction & TDSP Routing API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

STATE = {"error": None, "started": time.time()}

# Landmarks are plain public coordinates (no geocoding API). Only those that snap onto the dataset's
# road network (within max_snap_distance_m) are offered; the others are reported as out of coverage.
LANDMARKS = {
    "Connaught Place": (28.6315, 77.2195), "India Gate": (28.6129, 77.2295), "Red Fort": (28.6562, 77.2410),
    "Karol Bagh": (28.6519, 77.1900), "Chandni Chowk": (28.6506, 77.2300), "Lajpat Nagar": (28.5677, 77.2432),
    "Saket": (28.5245, 77.2100), "Dwarka Sector 21": (28.5521, 77.0585), "Rohini": (28.7328, 77.1200),
    "Nehru Place": (28.5487, 77.2502), "ISKCON Temple Delhi": (28.5535, 77.2588), "Rajouri Garden": (28.6492, 77.1200),
    "AIIMS": (28.5672, 77.2100), "Hauz Khas": (28.5494, 77.2001),
}


# ---------------------------------------------------------------- error handling
@app.exception_handler(ForecastUnavailable)
async def _fu(_, e): return JSONResponse(status_code=409, content={"detail": str(e), "kind": "forecast_unavailable"})
@app.exception_handler(NoRoute)
async def _nr(_, e): return JSONResponse(status_code=422, content={"detail": str(e), "kind": "no_route"})
@app.exception_handler(ValueError)
async def _ve(_, e): return JSONResponse(status_code=422, content={"detail": str(e), "kind": "invalid_request"})
@app.exception_handler(KeyError)
async def _ke(_, e): return JSONResponse(status_code=422, content={"detail": str(e.args[0]) if e.args else "not found", "kind": "not_found"})
@app.exception_handler(Exception)
async def _any(request: Request, e):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error. Details were written to the server log.", "kind": "internal"})


# ---------------------------------------------------------------- schemas
class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

class RouteReq(BaseModel):
    origin: LatLon
    destination: LatLon
    departure: datetime
    model: Optional[str] = None

class DepartureReq(RouteReq):
    pass

class PredictReq(BaseModel):
    segment_id: int
    timestamp: datetime
    model: Optional[str] = None


def _model_or_default(fc, m):
    m = m or fc.default_model()
    if m not in fc.servable_models():
        why = next((r["status_detail"] for r in fc.registry() if r["key"] == m), "unknown model key")
        raise HTTPException(422, f"model '{m}' is not available ({why}). Ready models: {fc.servable_models()}")
    return m


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    ready = STATE["error"] is None
    out = {"status": "ok" if ready else "degraded", "error": STATE["error"], "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "uptime_s": round(time.time() - STATE["started"]), "database": str(DB_PATH.name)}
    if ready:
        fc = get_forecaster()
        try:
            dm = fc.default_model()
        except ForecastUnavailable:
            dm = None
        out.update(default_model=dm, servable_models=fc.servable_models(),
                   model_status={k: v["status"] for k, v in fc.statuses().items()},
                   graph={"nodes": get_graph().n_nodes, "segments": get_graph().n_seg,
                          "components": get_graph().n_components})
    return out


@app.get("/api/config")
def config():
    _need_ready()
    g, cfg = get_graph(), congestion_config()
    lm, out_of_cov = [], []
    for name, (lat, lon) in LANDMARKS.items():
        node, d = g.nearest_node(lat, lon)
        (lm if d <= cfg["max_snap_distance_m"] else out_of_cov).append(
            {"name": name, "lat": lat, "lon": lon, "snap_m": round(d)})
    p = get_forecaster().panel
    return {"congestion": cfg, "levels": LEVELS, "landmarks": lm, "landmarks_out_of_coverage": out_of_cov,
            "time_window": {"start": str(p.timestamps[0]), "end": str(p.timestamps[-1] + pd.Timedelta(minutes=59))},
            "splits": {k: [str(p.timestamps[a]), str(p.timestamps[b - 1])] for k, (a, b) in p.split_bounds().items()},
            "map": {"center": [28.5937, 77.2090], "attribution": "© <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors",
                    "tile_policy_url": "https://operations.osmfoundation.org/policies/tiles/"}}


@app.get("/api/dataset/status")
def dataset_status():
    s = read_status()
    if not STATE["error"]:
        g = get_graph()
        s["graph"] = {"junctions": g.n_nodes, "segments": g.n_seg, "connected_components": g.n_components,
                      "segments_not_routable": int((~g.routable).sum())}
    return s


@app.get("/api/models")
def models(refresh: bool = False):
    """Real readiness of every model. refresh=true re-loads all weights and re-runs the validation predictions."""
    _need_ready()
    fc = get_forecaster()
    if refresh:
        fc.statuses(refresh=True)
    try:
        dm = fc.default_model()
    except ForecastUnavailable:
        dm = None
    return {"default_model": dm, "models": [{k: v for k, v in r.items() if k not in ("metrics", "stored_metrics")} | {
        "has_metrics": r["metrics"] is not None} for r in fc.registry()]}


@app.get("/api/model-metrics")
def model_metrics():
    _need_ready()
    fc = get_forecaster()
    reg = fc.registry()
    return {**{k: v for k, v in fc.metrics.items() if k != "models"}, "registry": reg,
            "note": "All numbers are computed from real held-out predictions in scripts/train_model.py / train_deep.py."}


@app.get("/api/segments/search")
def search_segments(q: str = Query(min_length=2, max_length=60), limit: int = Query(15, ge=1, le=50)):
    _need_ready()
    g, fc = get_graph(), get_forecaster()
    m = g.static["street_name"].fillna("").str.contains(q, case=False, regex=False).to_numpy()
    idx = np.flatnonzero(m)
    idx = idx[np.argsort(-fc.sat[idx])][:limit]           # busiest first
    return [{"segment_id": int(g.static["segment_id"].iat[i]), "street": str(g.street[i]),
             "length_m": round(g.length_km[i] * 1000), "frc": int(g.frc[i]), "speed_limit": int(g.free_flow[i]),
             "lat": float(g.static["start_lat"].iat[i]), "lon": float(g.static["start_lon"].iat[i])} for i in idx]


def _seg_index(fc, segment_id):
    i = int(np.searchsorted(np.sort(fc.panel.seg_ids), segment_id))
    pos = np.flatnonzero(fc.panel.seg_ids == segment_id)
    if len(pos) == 0:
        raise KeyError(f"unknown segment_id {segment_id}")
    return int(pos[0])


@app.post("/api/predict")
def predict(req: PredictReq):
    _need_ready()
    fc = get_forecaster()
    model = _model_or_default(fc, req.model)
    s = _seg_index(fc, req.segment_id)
    t0 = fc.panel.index_of(req.timestamp)
    hours = []
    for k in range(0, 4):
        t = t0 + k
        if t >= fc.panel.T:
            hours.append({"available": False, "reason": "beyond dataset coverage"}); continue
        c = fc.congestion(t, model)
        a = float(fc.panel.target[t, s])
        hours.append({"available": True, "hour_start": str(fc.panel.timestamps[t]), "offset_hours": k,
                      "split": fc.split_label(t), "predicted_probe_count": round(float(c["pred"][s]), 2),
                      "actual_probe_count": a, "estimated_speed_kmh": round(float(c["speed_kmh"][s]), 1),
                      "congestion_index": round(float(c["ci"][s]), 3), "congestion_level": LEVELS[int(c["level"][s])],
                      "observed_congestion_level": _observed_level(fc, t, s)})
    g = get_graph()
    return _json_safe({"segment_id": req.segment_id, "street": str(g.street[s]), "model": model,
            "model_label": next(r["label"] for r in fc.registry() if r["key"] == model),
            "free_flow_kmh": float(fc.free_flow[s]), "saturation_ref": round(float(fc.sat[s]), 1), "hourly": hours,
            "resolution": "hourly",
            "note": "15/30-minute horizons are not supported: the dataset is hourly. Speed is a heuristic derived from the predicted probe count."})


def _observed_level(fc, t, s):
    from src.congestion import compute_congestion
    c = compute_congestion(np.array([fc.panel.target[t, s]]), fc.sat[s:s + 1], fc.free_flow[s:s + 1])
    return LEVELS[int(c["level"][0])]


@app.get("/api/segment-series")
def segment_series(segment_id: int, model: Optional[str] = None):
    _need_ready()
    from src.congestion import compute_congestion
    fc = get_forecaster()
    model = _model_or_default(fc, model)
    s = _seg_index(fc, segment_id)
    pred = fc.segment_series(s, model)
    c = compute_congestion(pred, np.full(len(pred), fc.sat[s]), np.full(len(pred), fc.free_flow[s]))
    act = fc.panel.target[:, s].astype(float)
    ca = compute_congestion(act, np.full(len(act), fc.sat[s]), np.full(len(act), fc.free_flow[s]))
    split = [fc.split_label(i) for i in range(fc.panel.T)]
    from src.metrics import regression_metrics
    lo, hi = fc.panel.split_bounds()["test"]
    return _json_safe({"segment_id": segment_id, "street": str(get_graph().street[s]), "model": model,
            "timestamps": [str(t) for t in fc.panel.timestamps], "split": split,
            "actual_probe": act.tolist(), "predicted_probe": np.round(pred, 2).tolist(),
            "predicted_speed_kmh": np.round(c["speed_kmh"], 2).tolist(), "predicted_ci": np.round(c["ci"], 4).tolist(),
            "observed_ci": np.round(ca["ci"], 4).tolist(),
            "test_metrics_this_segment": regression_metrics(act[lo:hi], pred[lo:hi])})


@app.get("/api/map/segments")
def map_segments(west: float, south: float, east: float, north: float, timestamp: datetime,
                 model: Optional[str] = None, source: Literal["predicted", "observed"] = "predicted",
                 max_frc: int = Query(6, ge=1, le=6), limit: int = Query(5000, le=8000)):
    """Road segments (real dataset geometry) in a viewport, coloured by congestion level.
    source=predicted: model forecast for the hour.  source=observed: the SAME heuristic applied to the observed
    historical probe counts of that hour (dataset replay, not live traffic)."""
    _need_ready()
    fc, g = get_forecaster(), get_graph()
    t = fc.panel.index_of(timestamp)
    if source == "observed":
        model = None
        count = fc.panel.target[t].astype(np.float64)
        c = compute_congestion(count, fc.sat, fc.free_flow)
    else:
        model = _model_or_default(fc, model)
        c = fc.congestion(t, model)
        count = c["pred"]
    idx = g.edges_in_bbox(west, south, east, north)
    idx = idx[g.frc[idx] <= max_frc]
    truncated = len(idx) > limit
    if truncated:
        idx = idx[np.argsort(g.frc[idx], kind="stable")][:limit]
    feats = [{"id": int(g.static["segment_id"].iat[i]), "street": str(g.street[i]), "level": LEVELS[int(c["level"][i])],
              "ci": round(float(c["ci"][i]), 3), "speed": round(float(c["speed_kmh"][i]), 1),
              "count": round(float(count[i]), 1), "points": g.geometry(int(i))}
             for i in idx]
    lv = c["level"]
    return {"hour": str(fc.panel.timestamps[t]), "source": source, "model": model, "split": fc.split_label(t), "truncated": truncated,
            "count_kind": "predicted_probe_count" if source == "predicted" else "observed_probe_count",
            "segments": feats,
            "citywide_summary": {"counts": {LEVELS[k]: int((lv == k).sum()) for k in range(4)},
                                 "mean_ci": round(float(np.nanmean(c["ci"])), 4),
                                 "share_high_or_severe_pct": round(float((lv >= 2).mean() * 100), 2),
                                 "n_segments": int(len(lv))}}


@app.post("/api/route")
def route(req: RouteReq):
    _need_ready()
    fc = get_forecaster()
    model = _model_or_default(fc, req.model)
    r = compute_routes((req.origin.lat, req.origin.lon), (req.destination.lat, req.destination.lon), req.departure, model)
    r["events"] = events_svc.events_overlapping(r["tdsp"]["geometry"] + r["shortest"]["geometry"],
                                                r["tdsp"]["departure"], max(r["tdsp"]["arrival"], r["shortest"]["arrival"]))
    log_query("route", req.model_dump(mode="json"), model, f"saved {r['difference']['time_saved_min']} min")
    return r


@app.post("/api/departure-analysis")
def departure(req: DepartureReq):
    _need_ready()
    fc = get_forecaster()
    model = _model_or_default(fc, req.model)
    out = analyse((req.origin.lat, req.origin.lon), (req.destination.lat, req.destination.lon), req.departure, model)
    log_query("departure", req.model_dump(mode="json"), model, out["recommendation"].get("best_offset_min"))
    return out


@app.get("/api/weather")
def weather(start: date, end: date):
    if end < start or (end - start).days > 31:
        raise HTTPException(422, "end must be >= start and the window at most 31 days")
    return weather_svc.fetch_hourly(start, end)


@app.get("/api/events")
def events():
    init_db(reload_events=False)
    return {"events": events_svc.list_events(),
            "note": "Manually curated context list. Impact radii are documented assumptions; no automatic event detection exists."}


# ---------------------------------------------------------------- static frontend (after `npm run build`)
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")
