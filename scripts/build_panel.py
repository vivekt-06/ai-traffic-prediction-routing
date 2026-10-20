"""Build data/processed/panel.npz from features.parquet (fast; run after preprocess.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.panel import build_panel
if __name__ == "__main__":
    print("panel shape (hours, segments):", build_panel())
