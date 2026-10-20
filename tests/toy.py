"""Tiny graph stand-in exposing the attributes the TDSP code uses."""
import numpy as np

class ToyGraph:
    def __init__(self, n_nodes, edges):                      # edges: [(u, v, length_km)]
        src, dst, eid, rev = [], [], [], []
        for e, (u, v, _) in enumerate(edges):
            src += [u, v]; dst += [v, u]; eid += [e, e]; rev += [0, 1]
        order = np.argsort(src, kind="stable")
        src, dst, eid, rev = (np.array(a)[order] for a in (src, dst, eid, rev))
        self.ptr_l = np.searchsorted(src, np.arange(n_nodes + 1)).tolist()
        self.to_l, self.edge_l, self.rev_l = dst.tolist(), eid.tolist(), rev.tolist()
        self.len_l = [l for _, _, l in edges]
        self.edges = edges

def speeds_fn(table):
    from src.forecast import ForecastUnavailable
    def f(h):
        if not 0 <= h < len(table):
            raise ForecastUnavailable(f"hour {h} outside forecast")
        return table[h]
    return f
