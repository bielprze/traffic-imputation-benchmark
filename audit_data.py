"""
Descriptive NaN-only source audit, separate from model preprocessing.
"""

import json
import hashlib
import collections
from pathlib import Path
import numpy as np
import pandas as pd

import argparse

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-root", type=Path, default=ROOT / "runs/data_audit"
)
OUT = parser.parse_args().output_root.resolve()
OUT.mkdir(parents=True, exist_ok=True)
geo = json.loads((ROOT / "sensors_location.geojson").read_text())
features = geo["features"]
ids = [f["properties"].get("sensor_id") for f in features]
counts = collections.Counter(ids)
report = {
    "geojson": {
        "sha256": (
            hashlib.sha256(
                (ROOT / "sensors_location.geojson").read_bytes()
            ).hexdigest()
        ),
        "n_features": len(features),
        "unique_ids": len(set(ids)),
        "duplicate_ids": {k: v for k, v in counts.items() if v > 1},
        "geometry_types": dict(
            collections.Counter(f["geometry"]["type"] for f in features)
        ),
        "crs": geo.get("crs"),
        "property_keys": sorted(
            set(k for f in features for k in f["properties"])
        ),
    },
    "datasets": [],
}


def stats(x):
    return {
        "shape": list(x.shape),
        "nan": int(np.isnan(x).sum()),
        "posinf": int(np.isposinf(x).sum()),
        "neginf": int(np.isneginf(x).sum()),
        "zero": int((x == 0).sum()),
        "negative_finite": int((np.isfinite(x) & (x < 0)).sum()),
        "above1500_finite": int((np.isfinite(x) & (x > 1500)).sum()),
        "minimum_finite": float(x[np.isfinite(x)].min()),
        "maximum_finite": float(x[np.isfinite(x)].max()),
    }


for name, file in [
    ("jan2023", "combined_sensors_2023-01-01_2023-01-31.csv"),
    ("octdec2022", "combined_sensors_2022-10-01_2022-12-31.csv"),
]:
    p = ROOT / file
    df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    expected = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    missing_times = expected.difference(df.index)
    df = df.reindex(expected)
    x = df.to_numpy(float)
    a, b = int(len(x) * 0.5), int(len(x) * 0.7)
    tr = x[:a]
    observed = ~np.isnan(tr)
    allzero = observed.any(0) & np.all(~observed | (tr == 0), axis=0)
    availability = observed.mean(0)
    keep = (availability >= 0.9) & ~allzero
    eligible = df.columns[keep].tolist()
    constant_positive = (np.nanstd(tr, axis=0) == 0) & (
        np.nanmean(tr, axis=0) > 0
    )
    m = np.isnan(tr[:, keep])
    lengths = []
    for c in range(m.shape[1]):
        d = np.diff(np.r_[False, m[:, c], False].astype(int))
        lengths.extend(
            (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()
        )
    row = {
        "name": name,
        "path": str(p.relative_to(ROOT)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "timestamp_duplicates": int(df.index.duplicated().sum()),
        "introduced_missing_timestamps": len(missing_times),
        "all": stats(x),
        "train": stats(tr),
        "validation": stats(x[a:b]),
        "test": stats(x[b:]),
        "splits": {"train": [0, a], "validation": [a, b], "test": [b, len(x)]},
        "split_dates": {
            k: [str(df.index[i]), str(df.index[j - 1])]
            for k, (i, j) in {
                "train": [0, a],
                "validation": [a, b],
                "test": [b, len(x)],
            }.items()
        },
        "train_allzero_sensors": df.columns[allzero].tolist(),
        "train_availability_below90": df.columns[availability < 0.9].tolist(),
        "train_constant_positive_sensors": (
            df.columns[constant_positive].tolist()
        ),
        "eligible_n": len(eligible),
        "eligible_ids": eligible,
        "eligible_missing_geometry": sorted(set(eligible) - set(ids)),
        "eligible_duplicate_geometry": [s for s in eligible if counts[s] > 1],
        "train_eligible": stats(tr[:, keep]),
        "natural_nan_profile": {
            "source": "training rows, eligible sensors, np.isnan only",
            "nan_cells": int(m.sum()),
            "run_count": len(lengths),
            "lengths": lengths,
            "per_sensor_nan_count": m.sum(0).tolist(),
            "simultaneous_max": int(m.sum(1).max()),
            "empirical_replay_supported": bool(m.any()),
        },
    }
    report["datasets"].append(row)
(OUT / "raw_data_profile.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "geojson": report["geojson"],
            "datasets": [
                {
                    k: r[k]
                    for k in [
                        "name",
                        "all",
                        "train",
                        "validation",
                        "test",
                        "eligible_n",
                        "train_allzero_sensors",
                        "train_constant_positive_sensors",
                        "eligible_missing_geometry",
                        "eligible_duplicate_geometry",
                        "natural_nan_profile",
                    ]
                }
                for r in report["datasets"]
            ],
        },
        indent=2,
    )
)
