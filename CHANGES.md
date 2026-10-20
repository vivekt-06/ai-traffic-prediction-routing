# Files changed and why
**Removed**: `app.py`, `pages/*`, `common.py` (Streamlit UI → replaced by React); `src/prediction.py`, `inference.py`, `context.py`, `visualization.py`, `departure_analysis.py` (replaced); `data/contextual/weather.csv` (unsourced hand-typed values; weather now comes from Open-Meteo).
**Kept (original raw pipeline)**: `src/preprocessing.py`, `src/data_loader.py`, `src/raw_features.py` (was `feature_engineering.py`), `scripts/preprocess.py`.
**New backend**: `api/main.py`; `src/{config,metrics,congestion,panel,feature_engineering,training,forecast,graph,tdsp,routing,departure,weather,events,db,dataset_quality,deep_data}.py`.
**Fixes to the original**: seg_mean target leakage (now train-only); Congestion Index capped at 0.5 so SEVERE was unreachable; routing used one static hour (now true TDSP); silent clamping of out-of-range hours (now disclosed as unavailable); first-day-warm-up/roll3 docs corrected.
**Scripts**: `build_panel.py, train_model.py, train_deep.py (untested), init_db.py, dataset_status.py, fetch_weather_history.py`.
**Config/data**: `configs/dataset.json` (column mapping, splits), `configs/congestion.json`, `data/contextual/events.csv` (radius, source, verification columns), `data/processed/{panel.npz,dataset_status.json}`, `models/trained/*` (retrained).
**Frontend**: `frontend/` (Vite, React, TS, Tailwind, Leaflet, Recharts; 7 sections).
**Tests**: `tests/` (50). **Docs**: `README.md, CHANGES.md, .env.example, requirements*.txt`.

# Change set: real model status, working OpenStreetMap map (this session)
**Root cause of the broken map**: `@import "leaflet/dist/leaflet.css"` sat after the `@tailwind` rules in `index.css` (CSS ignores late `@import`), so Leaflet's stylesheet was missing from the build. Fix: import it in `frontend/src/main.tsx`.
**Backend**: new `src/model_status.py` (readiness checks); `src/forecast.py` (status registry: ready/untrained/error, runtime demotion on inference failure); `api/main.py` (`/api/models?refresh`, reasons in 422s, tolerant startup, `/api/map/segments?source=predicted|observed`, attribution link); `scripts/train_deep.py` (smoke runs isolated, per-model config/preprocessing JSON, panel fingerprint, training curve, tunable hyper-parameters, non-persistent adjacency buffer); new `scripts/check_models.py`.
**Frontend**: `components/MapView.tsx` (predicted/observed layer, click popups, provenance label, tile-failure notice, ResizeObserver, no canvas-renderer leak), new `components/MapSourceToggle.tsx`, `components/ui.tsx` (`StatusBadge`, wrapping card headers), `pages/Models.tsx` (backend status, error reasons), `pages/Dashboard.tsx`, `pages/Routes.tsx`, `pages/Methodology.tsx`, `App.tsx`, `store.tsx`, `api.ts`, `types.ts`, `index.css`, `main.tsx`, new `frontend/.env.example`.
**Tests**: new `tests/test_model_status.py`; `tests/test_pipeline_real.py` updated to the new status vocabulary.
**Removed**: stale `frontend/dist/` (built with the CSS bug; rebuild).
