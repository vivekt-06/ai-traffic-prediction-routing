"""Compute dataset validation/quality statistics -> data/processed/dataset_status.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset_quality import write_status
if __name__ == "__main__":
    print(json.dumps(write_status(), indent=2))
