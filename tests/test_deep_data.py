import numpy as np, pandas as pd, pytest
from scipy.sparse import csr_matrix
from src.deep_data import MIN_TARGET_HOUR, SEQ_LEN, normalized_adjacency, window, window_hours, time_features


def test_window_ends_24h_before_target_and_excludes_target():
    S = np.arange(24 + 100, dtype=float)[:, None]          # series idx i holds value i; hour h <-> idx h+24
    t = 60
    w = window(S, t)
    assert w.shape == (SEQ_LEN, 1)
    assert w[-1, 0] == (t - 24) + 24 and w[0, 0] == (t - 47) + 24
    assert max(window_hours(t)) == t - 24 < t


def test_window_requires_full_history():
    with pytest.raises(ValueError):
        window(np.zeros((30, 2)), MIN_TARGET_HOUR - 1)


def test_normalized_adjacency_is_symmetric_with_unit_scale():
    a = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float))
    n = normalized_adjacency(a).toarray()
    assert np.allclose(n, n.T) and n[0, 0] > 0
    assert np.allclose(n[0, 1], 1 / np.sqrt(2 * 3))


def test_segment_adjacency_on_toy_line_graph():
    import pandas as pd
    from src.graph import RoadGraph, segment_adjacency
    st = pd.DataFrame({"segment_id": [1, 2, 3], "street_name": ["a", "b", "c"], "speed_limit": [50] * 3, "frc": [3] * 3,
                       "distance_km": [1.0] * 3, "start_lon": [77.0, 77.1, 77.2], "start_lat": [28.5] * 3,
                       "end_lon": [77.1, 77.2, 77.3], "end_lat": [28.5] * 3,
                       "geometry_coords": ["[[77.0,28.5],[77.1,28.5]]", "[[77.1,28.5],[77.2,28.5]]", "[[77.2,28.5],[77.3,28.5]]"]})
    A = segment_adjacency(RoadGraph(st)).toarray()
    assert A.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_time_features_shape_and_ranges():
    f = time_features(pd.date_range("2024-08-12", periods=24 * 7, freq="h"))
    assert f.shape == (168, 4) and f[:, 0].min() >= -1 and f[:, 3].max() == 1
