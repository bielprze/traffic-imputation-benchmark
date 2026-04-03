"""
helper.py

Utility functions for loading and preprocessing sensor and GeoJSON data.
"""

import json
import os
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
from pyproj import Transformer

EXCLUDE_KEYS = ['A173']  # sometimes, some sensors must be excluded due to corruption in data


def load_and_normalize_geojson(
    file_path: str, width: int = 256, height: int = 256
) -> Dict[str, Tuple[float, float]]:
    """
    Load a GeoJSON file and normalize the point coordinates to fit within
    a width x height bitmap.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    coords = [feat["geometry"]["coordinates"] for feat in data.get("features", [])]
    if not coords:
        return {}

    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    normalized: Dict[str, Tuple[float, float]] = {}
    for feat in data.get("features", []):
        sensor_id = feat["properties"].get("sensor_id")
        lon, lat = feat["geometry"]["coordinates"]

        norm_x = ((lon - min_x) / (max_x - min_x) * width) if max_x != min_x else width / 2
        norm_y = ((lat - min_y) / (max_y - min_y) * height) if max_y != min_y else height / 2

        normalized[sensor_id] = (norm_x, norm_y)

    return normalized


def load_data_and_combine_with_geojson(
    folder_to_load: str,
    geojson_path: str
) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, List[str]]:
    """
    OLD PIPELINE (per-sensor CSV in folder).

    Load sensor locations (projected), time series data, filter out
    zero-only sensors, and assemble into a full data matrix.

    Returns:
      - dict of sensor_id → (x, y)
      - 2D numpy array of shape (time_steps, num_sensors)
      - list of all originally found sensor_ids
    """
    sensor_points = _load_geojson_recalculate(geojson_path)
    sensor_values, max_len = _load_sensor_data(folder_to_load)

    all_keys = list(sensor_values.keys())
    clean_points = _clean_keys_remove_zero(
        sensor_points,
        all_keys,
        sensor_values,
        exclude_keys=EXCLUDE_KEYS
    )
    kept_keys = list(clean_points.keys())

    full_data = np.zeros((max_len, len(kept_keys)), dtype=float)
    for idx, key in enumerate(kept_keys):
        vals = sensor_values.get(key, [])
        n = len(vals)
        full_data[:n, idx] = vals

    return clean_points, full_data, all_keys


# -------------------------- NEW PIPELINE (wide CSV) -------------------------- #

def load_wide_csv_and_filter_geojson(
    csv_path: str,
    geojson_path: str,
    timestamp_col: str = "timestamp",
    exclude_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[float, float]]]:
    """
    New loader for wide CSV format:
      timestamp, A001, A002, ...

    It also loads geojson and filters sensors to those present in CSV headers.

    Returns:
      timestamps: (T,) numpy array of datetime64 / objects (from pandas)
      data:       (T, N) float array aligned to sensor_names
      sensor_names: list[str] in the same order as columns in the CSV
      sensors_points: dict sensor_id -> (x,y) projected points, in the same order
    """
    exclude_keys = exclude_keys or []

    df = pd.read_csv(csv_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"CSV must contain '{timestamp_col}' column. Found: {list(df.columns)[:10]}...")

    # parse timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # sensors from CSV headers (keep order)
    sensor_names = [c for c in df.columns if c != timestamp_col]

    # apply exclude list
    sensor_names = [s for s in sensor_names if s not in exclude_keys]

    if len(sensor_names) == 0:
        raise ValueError("No sensor columns found in CSV (after exclusions).")

    # load geojson points (projected)
    points_all = _load_geojson_recalculate(geojson_path)

    # filter to sensors present in BOTH csv and geojson
    missing_in_geo = [s for s in sensor_names if s not in points_all]
    if missing_in_geo:
        # If some CSV columns have no geojson entry -> we drop them to keep graph consistent
        # Alternatively you could raise; dropping is usually more convenient.
        sensor_names = [s for s in sensor_names if s in points_all]

    if len(sensor_names) == 0:
        raise ValueError("After filtering by geojson, no sensors remain. Check headers vs geojson sensor_id.")

    # build ordered sensors dict
    sensors_points = {sid: points_all[sid] for sid in sensor_names}

    # data to float (NaNs preserved)
    data = df[sensor_names].to_numpy(dtype=float)
    timestamps = df[timestamp_col].to_numpy()

    return timestamps, data, sensor_names, sensors_points


# -------------------------- Internals (unchanged) --------------------------- #

def _load_geojson_recalculate(file_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Internal: load a GeoJSON file and project coordinates from
    EPSG:4326 to EPSG:2100 (Greek Grid).
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    points: Dict[str, Tuple[float, float]] = {}

    for feat in data.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        sensor_id = feat["properties"].get("sensor_id")
        x, y = transformer.transform(lon, lat)
        points[sensor_id] = (x, y)

    return points


def _load_sensor_data(folder_path: str) -> Tuple[Dict[str, List[int]], int]:
    """
    Internal: read all CSVs in a folder, extract the 'Total' column
    as int lists, return mapping sensor_id → values and the maximum series length.
    """
    data: Dict[str, List[int]] = {}
    max_len = 0

    for fname in os.listdir(folder_path):
        sensor_id, ext = os.path.splitext(fname)
        full_path = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(full_path)
            if 'Total' in df.columns:
                vals = df['Total'].dropna().astype(int).tolist()
                data[sensor_id] = vals
                max_len = max(max_len, len(vals))
            else:
                print(f"Warning: 'Total' column not found in {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return data, max_len


def _clean_keys_remove_zero(
    sensors: Dict[str, Tuple[float, float]],
    keys: List[str],
    sensor_values: Dict[str, List[float]],
    exclude_keys: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Internal: remove sensors that are in exclude_keys or whose entire
    value series is zero.
    """
    clean: Dict[str, Tuple[float, float]] = {}

    for key in keys:
        if key in exclude_keys or key not in sensors:
            continue
        vals = sensor_values.get(key, [])
        if any(abs(v) > 1e-6 for v in vals):
            clean[key] = sensors[key]

    return clean
