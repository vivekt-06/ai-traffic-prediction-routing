"""
scripts/preprocess.py
----------------------
Run once (or whenever the raw data changes) to build the processed, model-ready
dataset. Saves parquet files so the Streamlit app never has to re-parse the raw
GeoJSON at startup.

Usage:
    python scripts/preprocess.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import run_preprocessing
from src.raw_features import build_feature_table

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full, seg_static = run_preprocessing()

    print("Building feature table (lag/rolling features)...")
    features = build_feature_table(full)

    print("Saving processed data...")
    seg_static.to_parquet(PROCESSED_DIR / "segments_static.parquet", index=False)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    print(f"Done. {len(features):,} model-ready rows saved to {PROCESSED_DIR}")
    print(f"Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")


if __name__ == "__main__":
    main()
