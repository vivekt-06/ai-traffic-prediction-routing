"""
Train LSTM, GRU and ST-GCN-GRU (PyTorch) on the same chronological split used by the tree models,
evaluate on the held-out test days, and write (all under models/trained/):
    <name>.pt                      weights (state_dict; the fixed road-graph matrix is NOT stored in it)
    deep_config_<name>.json        architecture, hyper-parameters, preprocessing constants, param shapes,
                                   panel fingerprint, per-epoch training curve, exact command line
    deep_pred_<name>.npy           predictions [hours, segments] (the web app serves these; no torch needed to serve)
    metrics.json                   merged with the classical-model results (val + test metrics, computed here)

`--smoke` writes ONLY to models/trained/_smoke/ and never touches metrics.json, so a pipeline check can never make
a model look trained.

STATUS: this script has NOT been executed by the author's tooling (no PyTorch / GPU / network in the sandbox it was
reviewed in). Run `--smoke` first (~1 min), then the full command:

    python scripts/train_deep.py --smoke
    python scripts/train_deep.py --models lstm gru st_gcn_gru --epochs 30        # GPU strongly recommended
    python scripts/train_deep.py --models st_gcn_gru --epochs 30 --seed 7        # a single model / another seed

Afterwards the backend decides readiness itself (src/model_status.py): the model is "Ready" only if the weights file,
config and predictions are consistent and the stored test MAE is reproduced from the saved predictions.
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch is required for training only (not for serving). Install it from https://pytorch.org "
                     "(or `pip install -r requirements-deep.txt`), preferably on a GPU machine / Colab.")

from src.config import MODEL_DIR
from src.congestion import compute_congestion
from src.deep_data import MIN_TARGET_HOUR, SEQ_LEN, normalized_adjacency, time_features, window
from src.graph import RoadGraph, segment_adjacency
from src.metrics import regression_metrics
from src.model_status import panel_fingerprint
from src.panel import Panel
from src.training import load_static
import pandas as pd

SCALE = 1000.0   # global input/target scale (train P99 ~ 950 probes/hour)


def feats(x):                                # x: [..., ] raw counts -> 2 channels
    return torch.stack([x / SCALE, torch.log1p(x) / 8.0], dim=-1)


class SeqModel(nn.Module):
    """Per-segment sequence model (LSTM or GRU) + target-time and static features."""
    def __init__(self, cell="lstm", hidden=64, n_static=3, n_time=4):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(2, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + n_static + n_time, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, static, tfeat):                   # seq [B,L], static [B,3], tfeat [B,4]
        out, _ = self.rnn(feats(seq))
        return self.head(torch.cat([out[:, -1], static, tfeat], 1)).squeeze(-1) * SCALE


class STGCNGRU(nn.Module):
    """Spatial-temporal: at every input step a 2-hop graph convolution over ROAD CONNECTIVITY
    (segments are nodes, neighbours share a junction) mixes each segment with its neighbours;
    a GRU then runs over the 24 steps of each node. NOT a renamed GRU: the spatial operator A_hat
    is applied inside the forward pass."""
    def __init__(self, a_hat, hidden=32, n_static=3, n_time=4):
        super().__init__()
        self.register_buffer("A", a_hat, persistent=False)    # sparse [N,N]; rebuilt from the road graph, not stored in .pt
        self.inp = nn.Linear(2, hidden)
        self.g1, self.g2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static + n_time, 64), nn.ReLU(), nn.Linear(64, 1))

    def _prop(self, z):                                       # z [B,L,N,H] -> A_hat applied over N
        B, L, N, H = z.shape
        z = z.permute(2, 0, 1, 3).reshape(N, B * L * H)
        z = torch.sparse.mm(self.A, z)
        return z.reshape(N, B, L, H).permute(1, 2, 0, 3)

    def forward(self, seq, static, tfeat):                   # seq [B,L,N], static [N,3], tfeat [B,4]
        B, L, N = seq.shape
        z = torch.relu(self.inp(feats(seq)))                  # [B,L,N,H]
        z = torch.relu(self.g1(self._prop(z))) + z            # GCN layer 1 (+ residual)
        z = torch.relu(self.g2(self._prop(z))) + z            # GCN layer 2
        H = z.shape[-1]
        z = z.permute(0, 2, 1, 3).reshape(B * N, L, H)
        _, h = self.gru(z)
        h = h[-1].reshape(B, N, H)
        extra = torch.cat([static.unsqueeze(0).expand(B, -1, -1), tfeat.unsqueeze(1).expand(-1, N, -1)], -1)
        return self.head(torch.cat([h, extra], -1)).squeeze(-1) * SCALE   # [B,N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "gru", "st_gcn_gru"], choices=["lstm", "gru", "st_gcn_gru"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline (writes to models/trained/_smoke only)")
    ap.add_argument("--hidden", type=int, default=None, help="hidden size (default 64 for LSTM/GRU, 32 for ST-GCN-GRU)")
    ap.add_argument("--steps", type=int, default=400, help="optimizer steps per epoch for LSTM/GRU")
    ap.add_argument("--batch", type=int, default=8192, help="(hour, segment) samples per step for LSTM/GRU")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 1
    out_dir = MODEL_DIR / "_smoke" if a.smoke else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    print("device:", dev)

    panel = Panel()
    static_df = load_static(panel)
    series = panel.full_series()
    if not np.isfinite(series).all():
        raise SystemExit("panel.npz contains NaN/inf probe counts; rebuild it with scripts/build_panel.py before training")
    S = torch.tensor(series, dtype=torch.float32, device=dev)                      # [24+T, N]
    T, N = panel.T, panel.N
    bounds = panel.split_bounds()
    stat = torch.tensor(np.stack([static_df.speed_limit / 80.0, static_df.frc / 6.0,
                                  np.minimum(static_df.distance_km, 1.0)], 1), dtype=torch.float32, device=dev)
    tf = torch.tensor(time_features(panel.timestamps), device=dev)
    y_all = S[24:]                                                                  # [T,N] observed targets
    tr_hours = list(range(max(MIN_TARGET_HOUR, bounds["train"][0]), bounds["train"][1]))
    sat = pd.read_parquet(MODEL_DIR / "segment_stats.parquet")["saturation_ref"].to_numpy()
    ff = static_df["speed_limit"].to_numpy(float)

    def batch_seq(ts, segs):          # LSTM/GRU: sample (hour, segment) pairs
        ts, segs = torch.as_tensor(ts, device=dev), torch.as_tensor(segs, device=dev)
        idx = ts[:, None] + torch.arange(-(SEQ_LEN - 1), 1, device=dev)[None, :]    # series idx t-23..t
        return S[idx, segs[:, None]], stat[segs], tf[ts], y_all[ts, segs]

    def predict_hours(model, kind, hours):
        model.eval(); out = []
        with torch.no_grad():
            for t in hours:
                if kind == "st_gcn_gru":
                    seq = S[t - (SEQ_LEN - 1): t + 1].unsqueeze(0)
                    p = model(seq, stat, tf[t:t + 1])[0]
                else:
                    ps = []
                    for lo in range(0, N, 8192):
                        segs = np.arange(lo, min(N, lo + 8192))
                        s_, st_, tf_, _ = batch_seq(np.full(len(segs), t), segs)
                        ps.append(model(s_, st_, tf_))
                    p = torch.cat(ps)
                out.append(p.clamp(min=0).cpu().numpy())
        return np.stack(out)

    def split_mae(model, kind, name):
        lo, hi = bounds[name]; hrs = list(range(max(lo, MIN_TARGET_HOUR), hi))
        p = predict_hours(model, kind, hrs)
        return float(np.mean(np.abs(p - panel.target[hrs]))), p, hrs

    results = {}
    for name in a.models:
        print(f"\n=== {name} ===")
        if name == "st_gcn_gru":
            g = RoadGraph(static_df)
            A = normalized_adjacency(segment_adjacency(g)).tocoo()
            a_hat = torch.sparse_coo_tensor(torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long), torch.tensor(A.data, dtype=torch.float32), (N, N)).coalesce().to(dev)
            hidden = a.hidden or 32
            model, kind = STGCNGRU(a_hat, hidden=hidden).to(dev), "st_gcn_gru"
        else:
            hidden = a.hidden or 64
            model, kind = SeqModel(name, hidden=hidden).to(dev), "seq"
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        best, best_state, best_epoch, curve, t0 = float("inf"), None, 0, [], time.time()
        for ep in range(a.epochs):
            model.train(); losses = []
            order = np.random.permutation(tr_hours)
            if kind == "st_gcn_gru":
                step = 2
                chunks = [order[i:i + step] for i in range(0, len(order), step)]
                if a.smoke: chunks = chunks[:3]
                for ch in chunks:
                    seq = torch.stack([S[t - (SEQ_LEN - 1): t + 1] for t in ch])
                    pred = model(seq, stat, tf[torch.as_tensor(ch, device=dev)])
                    loss = nn.functional.mse_loss(pred / SCALE, y_all[torch.as_tensor(ch, device=dev)] / SCALE)
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            else:
                n_steps = 4 if a.smoke else a.steps
                for _ in range(n_steps):
                    ts = np.random.choice(tr_hours, a.batch); segs = np.random.randint(0, N, a.batch)
                    s_, st_, tf_, y = batch_seq(ts, segs)
                    loss = nn.functional.mse_loss(model(s_, st_, tf_) / SCALE, y / SCALE)
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            val_mae, _, _ = split_mae(model, kind, "val")
            print(f"epoch {ep+1}/{a.epochs} train_mse(scaled)={np.mean(losses):.5f} val_MAE={val_mae:.2f}", flush=True)
            curve.append({"epoch": ep + 1, "train_mse_scaled": float(np.mean(losses)), "val_MAE": val_mae})
            if val_mae < best:
                best, best_epoch, best_state = val_mae, ep + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / f"{name}.pt")
        cfg = {"name": name, "class": type(model).__name__, "hidden": hidden, "seq_len": SEQ_LEN, "lag_hours": 24,
               "input_scale": SCALE, "input_channels": ["count/SCALE", "log1p(count)/8"],
               "static_features": ["speed_limit/80", "frc/6", "min(distance_km,1)"],
               "time_features": ["hour_sin", "hour_cos", "dayofweek/6", "is_weekend"],
               "target": "hourly probe_count", "target_scale": SCALE, "lr": a.lr, "optimizer": "Adam",
               "epochs": a.epochs, "best_epoch": best_epoch, "steps_per_epoch": (a.steps if kind == "seq" else None),
               "batch": (a.batch if kind == "seq" else 2), "seed": a.seed, "torch": torch.__version__, "python": platform.python_version(),
               "device": str(dev), "command": " ".join(sys.argv), "n_parameters": int(sum(p.numel() for p in model.parameters())),
               "param_shapes": {k: list(v.shape) for k, v in best_state.items()},
               "panel_fingerprint": panel_fingerprint(panel.seg_ids, panel.T),
               "splits": {k: [str(panel.timestamps[lo]), str(panel.timestamps[hi - 1])] for k, (lo, hi) in bounds.items()},
               "model_selection": "epoch with the lowest validation MAE", "training_curve": curve,
               "adjacency": ("D^-1/2 (A+I) D^-1/2 over segments that share a junction (src/graph.py::segment_adjacency)" if kind == "st_gcn_gru" else None)}
        (out_dir / f"deep_config_{name}.json").write_text(json.dumps(cfg, indent=2))

        # full prediction matrix (hours without a complete window stay NaN -> app reports 'no forecast')
        all_hours = list(range(MIN_TARGET_HOUR, T))
        pred = np.full((T, N), np.nan, dtype=np.float32)
        pred[all_hours] = predict_hours(model, kind, all_hours)
        out = {"status": "trained", "servable": True, "uses_weather": False,
               "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_seconds": round(time.time() - t0, 1),
               "epochs": a.epochs, "best_epoch": best_epoch, "device": str(dev), "seed": a.seed, "smoke_run": bool(a.smoke),
               "n_parameters": int(sum(p.numel() for p in model.parameters())), "train_command": " ".join(sys.argv)}
        for split in ("val", "test"):
            lo, hi = bounds[split]; hrs = list(range(max(lo, MIN_TARGET_HOUR), hi))
            y, p = panel.target[hrs], pred[hrs]
            out[split] = regression_metrics(y, p)
            if split == "test":
                satt, fft = np.tile(sat, len(hrs)), np.tile(ff, len(hrs))
                la, lp = compute_congestion(y.ravel(), satt, fft)["level"], compute_congestion(p.ravel(), satt, fft)["level"]
                out[split]["congestion_level_agreement"] = float(np.mean(la == lp))
        np.save(out_dir / f"deep_pred_{name}.npy", pred)
        results[name] = out
        print(name, "test:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["test"].items()})

    if a.smoke:
        print(f"\nSMOKE RUN OK - artifacts in {out_dir}; metrics.json was NOT modified and no model is marked trained.")
        return
    mp = MODEL_DIR / "metrics.json"
    doc = json.loads(mp.read_text())
    doc["models"].update(results)
    ok = {k: v for k, v in doc["models"].items() if v.get("status") == "trained" and v.get("servable") and not v.get("smoke_run")}
    doc["default_model"] = min(ok, key=lambda k: ok[k]["val"]["MAE"])
    mp.write_text(json.dumps(doc, indent=2))
    print("default model:", doc["default_model"])


if __name__ == "__main__":
    main()
