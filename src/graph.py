"""
Road graph built from the dataset's own segment geometry (real Delhi road segments).

Why not OSMnx: the traffic observations are keyed to THESE segments (segment_id). An OSM
graph would need map-matching that the data cannot support, and the sandbox/offline
requirement rules out downloads. Node = junction (endpoint rounded to 5 dp, ~1 m);
edge = one dataset segment. The dataset has no one-way flag, so every segment is
traversable in both directions (disclosed limitation).
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from src.config import congestion_config
from src.panel import SEGMENTS_PATH

ROUND_DP = 5


def _haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class RoadGraph:
    def __init__(self, static: pd.DataFrame):
        """static: segments table in PANEL order (row i == panel column i)."""
        self.static = static.reset_index(drop=True)
        self.n_seg = len(static)
        key = {}
        def nid(lon, lat):
            return key.setdefault((round(lon, ROUND_DP), round(lat, ROUND_DP)), len(key))
        self.u = np.empty(self.n_seg, dtype=np.int64)
        self.v = np.empty(self.n_seg, dtype=np.int64)
        for i, r in enumerate(static.itertuples(index=False)):
            self.u[i], self.v[i] = nid(r.start_lon, r.start_lat), nid(r.end_lon, r.end_lat)
        self.n_nodes = len(key)
        xy = np.array(list(key.keys()), dtype=np.float64)
        self.node_lon, self.node_lat = xy[:, 0], xy[:, 1]
        self.routable = self.u != self.v                      # degenerate zero-length segments excluded
        self.length_km = static["distance_km"].to_numpy(np.float64)
        self.free_flow = static["speed_limit"].to_numpy(np.float64)
        self.street = static["street_name"].fillna("Unnamed road").to_numpy()
        self.frc = static["frc"].to_numpy()
        # CSR adjacency over directed arcs (2 per routable edge)
        e = np.flatnonzero(self.routable)
        src = np.concatenate([self.u[e], self.v[e]])
        dst = np.concatenate([self.v[e], self.u[e]])
        eid = np.concatenate([e, e])
        rev = np.concatenate([np.zeros(len(e), np.int8), np.ones(len(e), np.int8)])
        order = np.argsort(src, kind="stable")
        self.arc_to, self.arc_edge, self.arc_rev = dst[order], eid[order], rev[order]
        self.ptr = np.searchsorted(src[order], np.arange(self.n_nodes + 1))
        # connectivity
        a = coo_matrix((np.ones(len(e)), (self.u[e], self.v[e])), shape=(self.n_nodes,) * 2)
        n_comp, labels = connected_components(a, directed=False)
        sizes = np.bincount(labels)
        self.component = labels
        self.main_component = int(np.argmax(sizes))
        self.n_components = int(n_comp)
        self.main_nodes = np.flatnonzero(labels == self.main_component)
        # edge bounding boxes for viewport queries
        self.bbox = np.stack([np.minimum(static.start_lon, static.end_lon), np.minimum(static.start_lat, static.end_lat),
                              np.maximum(static.start_lon, static.end_lon), np.maximum(static.start_lat, static.end_lat)], 1)
        self._geom = {}
        # python-list adjacency for the (scalar-heavy) routing loops
        self.ptr_l, self.to_l, self.edge_l, self.rev_l = (self.ptr.tolist(), self.arc_to.tolist(),
                                                           self.arc_edge.tolist(), self.arc_rev.tolist())
        self.len_l = self.length_km.tolist()

    # -- geometry -----------------------------------------------------------
    def geometry(self, edge: int, reverse=False):
        """[[lat, lon], ...] polyline of a segment, oriented along travel direction."""
        g = self._geom.get(edge)
        if g is None:
            coords = json.loads(self.static["geometry_coords"].iat[edge])
            g = [[c[1], c[0]] for c in coords]
            self._geom[edge] = g
        return g[::-1] if reverse else g

    # -- snapping -------------------------------------------------------------
    def nearest_node(self, lat: float, lon: float):
        """Nearest junction of the main connected component. Returns (node, distance_m)."""
        d = _haversine_m(lon, lat, self.node_lon[self.main_nodes], self.node_lat[self.main_nodes])
        k = int(np.argmin(d))
        return int(self.main_nodes[k]), float(d[k])

    def snap(self, lat, lon):
        node, d = self.nearest_node(lat, lon)
        if d > congestion_config()["max_snap_distance_m"]:
            raise ValueError(f"({lat:.4f}, {lon:.4f}) is {d/1000:.1f} km from the nearest road in the dataset's coverage area")
        return node, d

    def edges_in_bbox(self, west, south, east, north):
        b = self.bbox
        return np.flatnonzero((b[:, 2] >= west) & (b[:, 0] <= east) & (b[:, 3] >= south) & (b[:, 1] <= north))

    def node_xy(self, n):
        return float(self.node_lat[n]), float(self.node_lon[n])


def segment_adjacency(g: RoadGraph) -> csr_matrix:
    """Line-graph adjacency among segments (two segments are neighbours if they share a junction).
    Used by the ST-GCN-GRU model: nodes = road segments, edges = road connectivity."""
    e = np.flatnonzero(g.routable)
    nodes = np.concatenate([g.u[e], g.v[e]])
    segs = np.concatenate([e, e])
    inc = coo_matrix((np.ones(len(segs)), (nodes, segs)), shape=(g.n_nodes, g.n_seg)).tocsr()
    adj = (inc.T @ inc).tocsr()
    adj.setdiag(0); adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


@lru_cache(maxsize=1)
def get_graph() -> RoadGraph:
    from src.panel import Panel
    from src.training import load_static
    return RoadGraph(load_static(Panel()))
