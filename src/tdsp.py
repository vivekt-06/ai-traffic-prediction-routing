"""
Time-Dependent Shortest Path (TDSP) - custom implementation.

Edge cost is a FUNCTION OF THE TIME THE VEHICLE ENTERS THE EDGE. Forecast speeds are piecewise
constant per hour. A vehicle entering an edge at time t covers distance at the speed of the hour
it is currently in, rolling over into the next hour's speed if the edge spans the boundary
(travel-time integration). This makes the cost function FIFO (leaving later can never mean
arriving earlier), which is the condition under which time-dependent Dijkstra - settle nodes in
order of earliest arrival time - is exactly optimal.

This is NOT Dijkstra on static weights: the weight of an edge is only known once its entry time is
known, and the entry time depends on the path taken so far.

Time axis: seconds since dataset hour 0 (panel index * 3600). `speed_at(h)` must return a python
list of km/h for dataset hour h and raise ForecastUnavailable beyond the covered range.
"""
import heapq
import math
from dataclasses import dataclass, field

from src.forecast import ForecastUnavailable


class NoRoute(Exception):
    pass


def traverse(length_km: float, t: float, edge: int, speed_at) -> float:
    """Arrival time (s) after entering `edge` at time t (s). Integrates across hourly buckets."""
    remaining = length_km
    while True:
        h = int(t // 3600)
        v = speed_at(h)[edge]
        if not (v > 0) or not math.isfinite(v):
            raise ForecastUnavailable(f"invalid forecast speed for edge {edge} at hour {h}")
        bucket_end = (h + 1) * 3600.0
        coverable = v * (bucket_end - t) / 3600.0
        if coverable >= remaining:
            return t + remaining / v * 3600.0
        remaining -= coverable
        t = bucket_end


def _reconstruct(pred, src, dst):
    arcs, n = [], dst
    while n != src:
        p, arc = pred[n]
        arcs.append(arc)
        n = p
    arcs.reverse()
    return arcs


def time_dependent_dijkstra(g, src: int, dst: int, t_depart: float, speed_at):
    """Earliest-arrival path. Returns (arc_index_list, arrival_time_s)."""
    if src == dst:
        return [], t_depart
    best = {src: t_depart}
    pred, done = {}, set()
    heap = [(t_depart, src)]
    ptr, to, edge, ln = g.ptr_l, g.to_l, g.edge_l, g.len_l
    while heap:
        t, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            return _reconstruct(pred, src, dst), t
        for a in range(ptr[u], ptr[u + 1]):
            w = to[a]
            if w in done:
                continue
            e = edge[a]
            t2 = traverse(ln[e], t, e, speed_at)
            if t2 < best.get(w, math.inf):
                best[w] = t2
                pred[w] = (u, a)
                heapq.heappush(heap, (t2, w))
    raise NoRoute("destination is not reachable from the origin")


def static_dijkstra(g, src: int, dst: int):
    """Plain shortest-DISTANCE path (ignores traffic) - the comparison route."""
    if src == dst:
        return []
    best, pred, done = {src: 0.0}, {}, set()
    heap = [(0.0, src)]
    ptr, to, edge, ln = g.ptr_l, g.to_l, g.edge_l, g.len_l
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            return _reconstruct(pred, src, dst)
        for a in range(ptr[u], ptr[u + 1]):
            w = to[a]
            d2 = d + ln[edge[a]]
            if d2 < best.get(w, math.inf):
                best[w] = d2; pred[w] = (u, a); heapq.heappush(heap, (d2, w))
    raise NoRoute("destination is not reachable from the origin")


@dataclass
class RouteEval:
    arcs: list
    depart_s: float
    arrive_s: float
    distance_km: float
    edge_rows: list = field(default_factory=list)   # per-edge: edge, enter_s, exit_s, speed, ci, level


def simulate(g, arcs, t_depart: float, speed_at, ci_at) -> RouteEval:
    """Drive a fixed path through the time-dependent forecast and record per-edge conditions."""
    t, dist, rows = t_depart, 0.0, []
    for a in arcs:
        e = g.edge_l[a]
        t2 = traverse(g.len_l[e], t, e, speed_at)
        h = int(t // 3600)
        rows.append({"edge": e, "arc": a, "reverse": bool(g.rev_l[a]), "enter_s": t, "exit_s": t2,
                     "length_km": g.len_l[e], "speed_kmh": speed_at(h)[e], "ci": float(ci_at(h)["ci"][e]),
                     "level": int(ci_at(h)["level"][e])})
        dist += g.len_l[e]
        t = t2
    return RouteEval(arcs, t_depart, t, dist, rows)
