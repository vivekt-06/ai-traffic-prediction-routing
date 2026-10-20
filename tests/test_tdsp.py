import itertools, math, random
import pytest
from src.forecast import ForecastUnavailable
from src.tdsp import NoRoute, simulate, static_dijkstra, time_dependent_dijkstra, traverse
from tests.toy import ToyGraph, speeds_fn

H = 3600.0


def test_traverse_constant_speed():
    f = speeds_fn([[60.0]] * 3)
    assert abs(traverse(30, 0, 0, f) - 1800) < 1e-9        # 30 km at 60 km/h = 30 min


def test_traverse_integrates_across_hour_boundary_by_hand():
    # edge 10 km. Enter at 0:50. Hour 0 speed 30 km/h -> 10 min covers 5 km. Hour 1 speed 10 km/h -> 5 km takes 30 min.
    f = speeds_fn([[30.0], [10.0], [10.0]])
    assert abs(traverse(10, 50 * 60, 0, f) - (60 + 30) * 60) < 1e-6


def test_traverse_fifo():
    f = speeds_fn([[60.0], [5.0], [60.0], [60.0]])
    arr = [traverse(8, t, 0, f) for t in range(0, 3 * 3600, 120)]
    assert all(b >= a - 1e-9 for a, b in zip(arr, arr[1:])), "leaving later must never arrive earlier"


def test_forecast_unavailable_beyond_horizon():
    f = speeds_fn([[10.0]])                                   # only one hour of forecast
    with pytest.raises(ForecastUnavailable):
        traverse(20, 0, 0, f)                                  # needs 2 h at 10 km/h


@pytest.fixture
def diamond():
    # A0 -e0(10km)- B1 -e1(10km)- D3 ; A0 -e2(12km)- C2 -e3(12km)- D3
    g = ToyGraph(4, [(0, 1, 10), (1, 3, 10), (0, 2, 12), (2, 3, 12)])
    table = [[60, 60, 60, 60], [60, 10, 60, 60], [60, 10, 60, 60], [60, 10, 60, 60]]
    return g, speeds_fn(table)


def test_tdsp_differs_from_static_departure_hour_dijkstra(diamond):
    """Depart 00:50. At departure-hour speeds B-route looks fastest (20 km @60), but its second edge is
    entered at 01:00 when it has slowed to 10 km/h. A truly time-dependent search must pick the C route."""
    g, f = diamond
    t0 = 50 * 60.0
    arcs, arr = time_dependent_dijkstra(g, 0, 3, t0, f)
    assert [g.edge_l[a] for a in arcs] == [2, 3]
    assert abs(arr - (t0 + 24 * 60)) < 1e-6                    # 24 km at 60 km/h
    # 'ordinary Dijkstra with static departure-hour weights' would choose edges 0,1:
    static_cost = {e: g.len_l[e] / f(0)[e] for e in range(4)}
    assert static_cost[0] + static_cost[1] < static_cost[2] + static_cost[3]
    ev = simulate(g, static_dijkstra(g, 0, 3), t0, f, lambda h: {"ci": [0] * 4, "level": [0] * 4})
    assert ev.arrive_s > arr + 30 * 60, "shortest-distance route is much slower under the time-dependent forecast"


def test_tdsp_matches_brute_force_on_random_graphs():
    rnd = random.Random(7)
    for trial in range(25):
        n = 6
        edges = {(min(a, b), max(a, b)) for a, b in (rnd.sample(range(n), 2) for _ in range(9))}
        edges = [(a, b, rnd.uniform(1, 9)) for a, b in sorted(edges)]
        g = ToyGraph(n, edges)
        table = [[rnd.uniform(4, 60) for _ in edges] for _ in range(8)]
        f = speeds_fn(table)
        t0 = rnd.uniform(0, 3000)
        adj = {}
        for e, (u, v, _) in enumerate(edges):
            adj.setdefault(u, []).append((v, e)); adj.setdefault(v, []).append((u, e))
        best = math.inf
        def dfs(u, t, seen):
            nonlocal best
            if u == n - 1:
                best = min(best, t); return
            for v, e in adj.get(u, []):
                if v not in seen:
                    dfs(v, traverse(edges[e][2], t, e, f), seen | {v})
        dfs(0, t0, {0})
        try:
            _, arr = time_dependent_dijkstra(g, 0, n - 1, t0, f)
        except NoRoute:
            assert best == math.inf
            continue
        assert abs(arr - best) < 1e-6, f"trial {trial}: TDSP {arr} vs brute force {best}"


def test_unreachable_destination():
    g = ToyGraph(4, [(0, 1, 1), (2, 3, 1)])
    with pytest.raises(NoRoute):
        time_dependent_dijkstra(g, 0, 3, 0, speeds_fn([[30, 30]] * 2))
    with pytest.raises(NoRoute):
        static_dijkstra(g, 0, 3)


def test_tdsp_never_worse_than_static_route(diamond):
    g, f = diamond
    ci = lambda h: {"ci": [0] * 4, "level": [0] * 4}
    for t0 in range(0, 3 * 3600, 600):
        _, arr = time_dependent_dijkstra(g, 0, 3, float(t0), f)
        assert arr <= simulate(g, static_dijkstra(g, 0, 3), float(t0), f, ci).arrive_s + 1e-6
