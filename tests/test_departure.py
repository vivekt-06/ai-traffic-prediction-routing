import pandas as pd
from src.departure import recommend

def row(off, dur, arr, ok=True):
    return {"offset_min": off, "label": "Now" if off == 0 else f"+{off} min", "available": ok,
            "duration_min": dur, "arrival": arr}

T0 = pd.Timestamp("2024-08-28 08:00")
def arr(off, dur): return str(T0 + pd.Timedelta(minutes=off + dur))

def test_recommends_later_departure_when_clearly_faster():
    rows = [row(0, 40, arr(0, 40)), row(15, 35, arr(15, 35)), row(30, 28, arr(30, 28)), row(45, 33, arr(45, 33))]
    r = recommend(rows, 1.0)
    assert r["best_offset_min"] == 30 and abs(r["time_saved_vs_now_min"] - 12) < 1e-9
    assert "12.0 min less driving" in r["explanation"] and "later" in r["explanation"]
    assert abs(r["arrival_delta_vs_now_min"] - 18) < 1e-9

def test_recommends_now_when_waiting_is_worse():
    rows = [row(0, 30, arr(0, 30)), row(15, 34, arr(15, 34)), row(30, 38, arr(30, 38)), row(45, 41, arr(45, 41))]
    assert recommend(rows, 1.0)["best_offset_min"] == 0

def test_negligible_saving_is_not_recommended_as_meaningful():
    r = recommend([row(0, 30, arr(0, 30)), row(15, 29.6, arr(15, 29.6))], 1.0)
    assert r["negligible"] is True and "no meaningful benefit" in r["explanation"]

def test_unavailable_scenarios_are_excluded_and_all_unavailable_is_reported():
    rows = [row(0, 30, arr(0, 30)), row(15, 1, "x", ok=False)]
    assert recommend(rows, 1.0)["best_offset_min"] == 0
    assert recommend([row(0, 0, "x", ok=False)], 1.0)["best_offset_min"] is None

def test_tie_prefers_earlier_departure():
    assert recommend([row(0, 30, arr(0, 30)), row(15, 30, arr(15, 30))], 1.0)["best_offset_min"] == 0
