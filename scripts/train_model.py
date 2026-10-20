"""
Train + evaluate Historical Average, Linear Regression, Random Forest, XGBoost (CPU, a few minutes).
Usage:  python scripts/train_model.py [--with-weather] [--skip random_forest ...]
Writes models/trained/{*.joblib,*.json,*.npy,segment_stats.parquet,metrics.json}.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-weather", action="store_true", help="also train *_weather variants (needs weather_history.parquet)")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    run_training(with_weather=a.with_weather, skip=tuple(a.skip))
