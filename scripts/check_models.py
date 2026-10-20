"""
Print the REAL readiness of every model (same checks the API runs at startup): weights load, held-out validation
prediction passes, stored metrics are reproduced.   Usage:  python scripts/check_models.py
Exit code 0 if at least one model is ready, 1 otherwise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forecast import get_forecaster

ICON = {"ready": "READY    ", "untrained": "UNTRAINED", "error": "ERROR    "}

if __name__ == "__main__":
    fc = get_forecaster()
    for r in fc.registry():
        print(f"{ICON[r['status']]}  {r['key']:<22} {r['status_detail']}")
        for c in r["checks"]:
            print(f"             [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    ready = fc.servable_models()
    print("\nready models:", ready or "none")
    sys.exit(0 if ready else 1)
