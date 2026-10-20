import json
import httpx, pytest
from datetime import date
from src import db, events as ev, weather as w


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()


class FakeResp:
    def __init__(self, data, code=200): self._d, self.status_code = data, code
    def json(self): return self._d
    def raise_for_status(self):
        if self.status_code >= 400: raise httpx.HTTPStatusError("bad", request=None, response=None)

class FakeClient:
    def __init__(self, resp=None, exc=None): self.resp, self.exc, self.calls = resp, exc, 0
    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.exc: raise self.exc
        return self.resp

PAYLOAD = {"hourly": {"time": ["2024-08-28T00:00", "2024-08-28T01:00"], "temperature_2m": [30.1, 29.5],
                       "relative_humidity_2m": [70, 72], "precipitation": [0.0, 1.2], "wind_speed_10m": [5.0, 6.1]},
           "hourly_units": {"temperature_2m": "°C"}}


def test_weather_parse_cache_and_provenance():
    c = FakeClient(FakeResp(PAYLOAD))
    r1 = w.fetch_hourly(date(2024, 8, 28), date(2024, 8, 28), client=c, today=date(2026, 10, 16))
    assert r1["available"] and r1["cache"] == "miss" and "Historical" in r1["source"]
    assert r1["hourly"][1]["precipitation"] == 1.2 and r1["fetched_at"].endswith("+00:00")
    r2 = w.fetch_hourly(date(2024, 8, 28), date(2024, 8, 28), client=c, today=date(2026, 10, 16))
    assert r2["cache"] == "hit" and c.calls == 1                      # no duplicate request


def test_weather_failure_is_graceful_and_not_a_stack_trace():
    r = w.fetch_hourly(date(2024, 8, 28), date(2024, 8, 28), client=FakeClient(exc=httpx.ConnectError("x")), today=date(2026, 10, 16))
    assert r["available"] is False and "Traceback" not in r["error"]


def test_weather_stale_cache_served_when_service_down(monkeypatch):
    w.fetch_hourly(date(2024, 8, 28), date(2024, 8, 28), client=FakeClient(FakeResp(PAYLOAD)), today=date(2026, 10, 16))
    monkeypatch.setattr(w, "time", type("T", (), {"time": staticmethod(lambda: 1e12)}))   # expire cache
    r = w.fetch_hourly(date(2024, 8, 28), date(2024, 8, 28), client=FakeClient(exc=httpx.ConnectError("x")), today=date(2026, 10, 16))
    assert r["available"] and r["cache"] == "stale" and "cached" in r["warning"]


def test_forecast_api_used_for_recent_dates():
    c = FakeClient(FakeResp(PAYLOAD))
    r = w.fetch_hourly(date(2026, 10, 16), date(2026, 10, 17), client=c, today=date(2026, 10, 16))
    assert "Forecast" in r["source"]


def test_events_loaded_with_documented_fields():
    e = ev.list_events()
    assert len(e) >= 3 and all(x["verification_status"] and x["source"] for x in e)


def test_event_overlap_requires_time_and_space():
    red_fort_route = [[28.6562, 77.2410], [28.6500, 77.2300]]
    far_route = [[28.52, 77.20], [28.53, 77.21]]
    inside = ev.events_overlapping(red_fort_route, "2024-08-15 08:00", "2024-08-15 09:00")
    assert any(x["event_name"].startswith("Independence") for x in inside)
    assert not any(x["event_name"].startswith("Independence") for x in ev.events_overlapping(far_route, "2024-08-15 08:00", "2024-08-15 09:00"))
    assert not ev.events_overlapping(red_fort_route, "2024-08-16 08:00", "2024-08-16 09:00")           # wrong day
    assert any(x["citywide"] for x in ev.events_overlapping(far_route, "2024-08-19 10:00", "2024-08-19 11:00"))
