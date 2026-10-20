"""
data_loader.py
---------------
Loads the raw "New Delhi Traffic Probe & Analytics 2024" dataset.

DATASET FORMAT (confirmed by direct inspection of the Kaggle download):
    data/raw/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__<date>_to_<date>_.geojson

Each daily GeoJSON file is a FeatureCollection where:
  - features[0] is a METADATA feature (geometry=None). Its "timeSets" list maps a
    timeSet id (2..25) to an hour-of-day bucket (e.g. id=2 -> "0:00-1:00").
  - features[1:] are ROAD SEGMENT features, one per road segment, each with:
        geometry            : LineString [[lon, lat], ...]
        properties.segmentId       : unique road-segment id (stable across all days)
        properties.streetName      : road name (may be empty)
        properties.speedLimit      : posted/map speed limit in km/h (static)
        properties.frc             : Functional Road Class, 1 (motorway) .. 6 (local street)
        properties.distance        : segment length in km (dataset is configured to KILOMETERS)
        properties.segmentProbeCounts : list of {timeSet, dateRange, probeCount} - ONE entry
                                         per hour of that day. probeCount is the number of
                                         GPS probes ("pings") observed on that segment in that
                                         hour - this is the dataset's actual measured traffic
                                         signal (a traffic-VOLUME/activity proxy).

IMPORTANT - there is no directly-measured per-segment speed in this dataset. Only a static
speedLimit (free-flow reference) and hourly probeCount are available. prediction.py predicts
probeCount; congestion.py derives an estimated speed/congestion from it. This is documented
explicitly so the modelling choice can be defended in the viva.
"""

import json
import glob
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "new_delhi_traffic_dataset"
PROBE_DIR = RAW_DIR / "probe_counts" / "geojson"


def list_daily_files():
    """Return sorted list of daily probe-count geojson file paths."""
    files = sorted(glob.glob(str(PROBE_DIR / "new_delhi__*.geojson")))
    if not files:
        raise FileNotFoundError(
            f"No daily probe-count files found under {PROBE_DIR}. "
            "Unzip the Kaggle download into data/raw/ so the path above exists."
        )
    return files


def _date_from_filename(path):
    """Extract the YYYY-MM-DD date encoded in the filename, e.g.
    new_delhi__2024-08-12_to_2024-08-12_.geojson -> '2024-08-12'."""
    name = Path(path).stem
    return name.split("__")[1].split("_to_")[0]


def load_segments_static(one_file_path=None):
    """Load the STATIC per-segment attributes (geometry, speedLimit, frc, distance,
    streetName) once from a single daily file - these do not change day to day."""
    path = one_file_path or list_daily_files()[0]
    with open(path) as f:
        data = json.load(f)

    rows = []
    for feat in data["features"][1:]:
        p = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        rows.append({
            "segment_id": p["segmentId"],
            "street_name": p.get("streetName", ""),
            "speed_limit": p["speedLimit"],
            "frc": p["frc"],
            # NOTE: despite the file's userPreference.distanceUnit saying
            # "KILOMETERS", direct verification against the haversine distance
            # between each segment's own start/end coordinates shows the raw
            # `distance` value is actually in METERS (ratio ~1000x, confirmed
            # across the whole file) - so we convert to km here at load time.
            "distance_km": p["distance"] / 1000.0,
            "start_lon": coords[0][0],
            "start_lat": coords[0][1],
            "end_lon": coords[-1][0],
            "end_lat": coords[-1][1],
            "geometry_coords": json.dumps(coords),
        })
    return pd.DataFrame(rows)


def load_all_probe_counts(cache_dir=None):
    """Load every daily file and build a LONG dataframe:
    one row per (segment_id, date, hour_of_day, probe_count).

    Memory-conscious implementation: each daily file is parsed straight into
    fixed-size NumPy arrays (no giant list of Python tuples), immediately
    downcast to compact dtypes, and optionally cached to a per-day parquet
    file on disk so re-runs don't need to re-parse the raw GeoJSON.

    Returns
    -------
    pd.DataFrame with columns: segment_id (int64), date (datetime64), hour (int8),
    probe_count (int32)
    """
    import numpy as np

    files = list_daily_files()
    day_frames = []
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        date_str = _date_from_filename(path)
        cache_path = cache_dir / f"{date_str}.parquet" if cache_dir else None

        if cache_path and cache_path.exists():
            day_frames.append(pd.read_parquet(cache_path))
            print(f"  loaded {date_str} from cache")
            continue

        with open(path) as f:
            data = json.load(f)

        meta = data["features"][0]["properties"]
        timeset_to_hour = {ts["@id"]: i for i, ts in enumerate(meta["timeSets"])}

        n_segments = len(data["features"]) - 1
        seg_ids = np.empty(n_segments * 24, dtype="int64")
        hours = np.empty(n_segments * 24, dtype="int8")
        counts = np.empty(n_segments * 24, dtype="int32")

        pos = 0
        for feat in data["features"][1:]:
            p = feat["properties"]
            seg_id = p["segmentId"]
            for entry in p["segmentProbeCounts"]:
                hour = timeset_to_hour.get(entry["timeSet"])
                if hour is None:
                    continue
                seg_ids[pos] = seg_id
                hours[pos] = hour
                counts[pos] = entry["probeCount"]
                pos += 1

        day_df = pd.DataFrame({
            "segment_id": seg_ids[:pos],
            "hour": hours[:pos],
            "probe_count": counts[:pos],
        })
        day_df["date"] = pd.Timestamp(date_str)

        if cache_path:
            day_df.to_parquet(cache_path, index=False)

        day_frames.append(day_df)
        print(f"  loaded {date_str}: {n_segments} segments")

    df = pd.concat(day_frames, ignore_index=True)
    return df


if __name__ == "__main__":
    print("Discovering daily files...")
    files = list_daily_files()
    print(f"Found {len(files)} daily files.")

    print("Loading static segment attributes...")
    seg_static = load_segments_static(files[0])
    print(f"  {len(seg_static)} road segments")

    print("Loading hourly probe counts (this can take a couple of minutes)...")
    probes = load_all_probe_counts()
    print(f"  {len(probes):,} (segment, date, hour) rows")
