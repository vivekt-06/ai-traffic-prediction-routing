"""API tests on the real artifacts (skipped when they are missing)."""
import pytest
from pathlib import Path
from src.config import MODEL_DIR, PROCESSED_DIR

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "panel.npz").exists() or not (MODEL_DIR / "metrics.json").exists(),
                                reason="artifacts missing")
O = {"lat": 28.6315, "lon": 77.2195}; D = {"lat": 28.5245, "lon": 77.2100}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_models_dataset(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["default_model"] in h["servable_models"]
    m = client.get("/api/models").json()
    keys = {x["key"]: x for x in m["models"]}
    assert {"historical_average", "linear_regression", "random_forest", "xgboost", "lstm", "gru", "st_gcn_gru"} <= set(keys)
    ds = client.get("/api/dataset/status").json()
    assert ds["records"] == 24938 * 456 and ds["quality"]["duplicate_keys"] == 0 and ds["resolution_minutes"] == 60
    mm = client.get("/api/model-metrics").json()
    assert mm["registry"] and "default_model_rule" in mm


def test_route_basic_invariants(client):
    r = client.post("/api/route", json={"origin": O, "destination": D, "departure": "2024-08-28T08:30:00", "model": "xgboost"})
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "xgboost" and j["split_of_departure_hour"] == "test"
    t, s = j["tdsp"], j["shortest"]
    assert t["duration_min"] <= s["duration_min"] + 1e-6, "TDSP is optimal under the same forecast"
    assert s["distance_km"] <= t["distance_km"] + 1e-6, "shortest-distance route is shortest by distance"
    assert j["difference"]["time_saved_min"] == pytest.approx(s["duration_min"] - t["duration_min"], abs=0.02)
    assert len(t["geometry"]) > 10 and t["n_segments"] == len(t["segments"])
    # route is a connected polyline: consecutive segments share an endpoint
    for a, b in zip(t["segments"], t["segments"][1:]):
        assert a["points"][-1] == b["points"][0]


def test_map_and_route_use_same_congestion_values(client):
    j = client.post("/api/route", json={"origin": O, "destination": D, "departure": "2024-08-28T08:00:00", "model": "random_forest"}).json()
    first = j["tdsp"]["segments"][0]
    # first segment is entered at 08:00 -> same hour as the map layer
    pts = first["points"]
    lat0, lon0 = pts[0]
    m = client.get("/api/map/segments", params={"west": lon0 - .002, "south": lat0 - .002, "east": lon0 + .002, "north": lat0 + .002,
                                                "timestamp": "2024-08-28T08:00:00", "model": "random_forest"}).json()
    match = [s for s in m["segments"] if s["points"] in (pts, pts[::-1])]
    assert match and match[0]["level"] == first["level"] and match[0]["ci"] == pytest.approx(first["ci"], abs=1e-3)


def test_departure_scenarios_are_recomputed_and_best_is_min(client):
    r = client.post("/api/departure-analysis", json={"origin": O, "destination": D, "departure": "2024-08-28T07:40:00", "model": "random_forest"}).json()
    av = [s for s in r["scenarios"] if s["available"]]
    assert [s["label"] for s in r["scenarios"]] == ["Now", "+15 min", "+30 min", "+45 min"]
    # each scenario equals an independent route computation at that departure time
    for s in av:
        one = client.post("/api/route", json={"origin": O, "destination": D, "departure": s["departure"], "model": "random_forest"}).json()
        assert s["duration_min"] == pytest.approx(one["tdsp"]["duration_min"], abs=1e-6)
    best = min(av, key=lambda s: (s["duration_min"], s["offset_min"]))
    assert r["recommendation"]["best_offset_min"] in (best["offset_min"], 0)


def test_forecast_availability_disclosed_at_end_of_data(client):
    r = client.post("/api/departure-analysis", json={"origin": O, "destination": D, "departure": "2024-08-30T23:10:00", "model": "xgboost"}).json()
    na = [s for s in r["scenarios"] if not s["available"]]
    assert na and all("outside dataset coverage" in s["reason"] for s in na)


def test_error_paths_are_clean(client):
    r = client.post("/api/route", json={"origin": O, "destination": D, "departure": "2030-01-01T08:00:00"})
    assert r.status_code == 409 and r.json()["kind"] == "forecast_unavailable"
    r = client.post("/api/route", json={"origin": {"lat": 28.7328, "lon": 77.12}, "destination": D, "departure": "2024-08-28T08:00:00"})
    assert r.status_code == 422 and "km from the nearest road" in r.json()["detail"]
    r = client.post("/api/route", json={"origin": O, "destination": D, "departure": "2024-08-28T08:00:00", "model": "lstm"})
    assert r.status_code == 422 and "not available" in r.json()["detail"]
    r = client.post("/api/route", json={"origin": {"lat": 999, "lon": 0}, "destination": D, "departure": "2024-08-28T08:00:00"})
    assert r.status_code == 422
    assert "Traceback" not in r.text


def test_predict_returns_model_and_actual(client):
    s = client.get("/api/segments/search", params={"q": "Ring Road"}).json()[0]
    p = client.post("/api/predict", json={"segment_id": s["segment_id"], "timestamp": "2024-08-28T08:30:00", "model": "random_forest"}).json()
    h0 = p["hourly"][0]
    assert p["model_label"] == "Random Forest" and h0["split"] == "test" and h0["actual_probe_count"] >= 0
    assert 0 <= h0["congestion_index"] <= 1 and h0["congestion_level"] in ("LOW", "MODERATE", "HIGH", "SEVERE")
    assert client.post("/api/predict", json={"segment_id": 1, "timestamp": "2024-08-28T08:30:00"}).status_code == 422


def test_weather_failure_does_not_break_api(client, monkeypatch):
    import httpx
    from src import weather
    monkeypatch.setattr(weather.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    j = client.get("/api/weather", params={"start": "2031-01-01", "end": "2031-01-01"}).json()
    assert j["available"] is False
