"""
Train-only sensor qualification, chronological splits, profiles, masks
and metrics.
"""

import hashlib
import json
import warnings
import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def metrics(y, p):
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Empty/nonfinite evaluation")
    e = np.abs(y - p)
    nz = y != 0
    return {
        "MAE": float(e.mean()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MRE": float(np.mean(e[nz] / np.abs(y[nz]))) if nz.any() else np.nan,
    }


def load_dataset(cfg, out):
    path = ROOT / cfg["path"]
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(expected)
    x = df.to_numpy(float)
    if np.isinf(x).any() or (x < 0).any():
        raise ValueError(
            "Unexpected infinite/negative counts: stop for data audit, do not"
            " convert to missing"
        )
    if cfg.get("qc_upper") is not None:
        raise ValueError(
            "This protocol counts only NaN as missing; QC threshold forbidden"
        )
    a, b = int(len(x) * 0.5), int(len(x) * 0.7)
    train = x[:a]
    observed = ~np.isnan(train)
    allzero = observed.any(0) & np.all(np.isnan(train) | (train == 0), axis=0)
    keep = (observed.mean(0) >= 0.9) & ~allzero
    selection = [
        {
            "sensor": name,
            "train_observed_fraction": float(observed[:, i].mean()),
            "train_all_observed_zero": bool(allzero[i]),
            "included": bool(keep[i]),
        }
        for i, name in enumerate(df.columns)
    ]
    if cfg.get("density", 1) < 1:
        idx = np.flatnonzero(keep)
        chosen = np.random.default_rng(20261007).choice(
            idx, int(len(idx) * cfg["density"]), replace=False
        )
        keep[:] = False
        keep[chosen] = True
    x = x[:, keep]
    names = df.columns[keep].tolist()
    train = x[:a]
    if (~np.isfinite(train).any(0)).any():
        raise ValueError("Sensor without training observations")
    mu = np.nanmean(train, 0)
    sd = np.nanstd(train, 0)
    sd[sd < 1e-6] = 1
    med = np.nanmedian(train, 0)
    edges = pd.read_csv(ROOT / cfg["edges"])
    idx = {s: i for i, s in enumerate(names)}
    A = np.zeros((len(names), len(names)))
    for u, v in edges.iloc[:, :2].itertuples(index=False, name=None):
        if u in idx and v in idx and u != v:
            A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1
    splits = {"train": [0, a], "validation": [a, b], "test": [b, len(x)]}
    meta = {
        "config": cfg,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sensors": names,
        "splits": splits,
        "dates": {
            k: [str(df.index[i]), str(df.index[j - 1])]
            for k, (i, j) in splits.items()
        },
        "profile_source": "train only; NaN only",
        "selection_source": "train only",
        "selection": selection,
        "isolates": int((A.sum(0) == 0).sum()),
        "qc_note": (
            "No upper threshold. Zeros and large counts are observed; only NaN"
            " missing. Constant positive sensors retained."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "dataset.json", meta)
    tmp = out / ("data_" + uuid.uuid4().hex + ".npz")
    np.savez_compressed(
        tmp,
        values=x,
        timestamps=df.index.to_numpy()
        .astype("datetime64[ns]")
        .astype("int64"),
        mean=mu,
        std=sd,
        median=med,
        adj=A,
    )
    os.replace(tmp, out / "data.npz")
    profile = profile_missing(train, df.index[:a])
    atomic_json(out / "profile.json", profile)
    return dict(
        x=x,
        index=df.index,
        names=names,
        a=a,
        b=b,
        mu=mu,
        sd=sd,
        med=med,
        A=A,
        profile=profile,
        meta=meta,
    )


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    s = np.flatnonzero(changes == 1)
    e = np.flatnonzero(changes == -1)
    return s, e


def profile_missing(train, times):
    natural = np.isnan(train)
    lens = []
    hour = np.zeros(24, int)
    for c in range(train.shape[1]):
        s, e = runs(natural[:, c])
        lens.extend((e - s).tolist())
        for t in s:
            hour[times[t].hour] += 1
    corr = pd.DataFrame(natural.astype(float)).corr().to_numpy()
    upper = corr[np.triu_indices(train.shape[1], 1)]
    upper = upper[np.isfinite(upper)]
    return {
        "source_rows": [0, len(train)],
        "missing_definition": "NaN only",
        "natural_rate_per_sensor": natural.mean(0).tolist(),
        "natural_lengths": lens,
        "start_hour_counts": hour.tolist(),
        "simultaneous_missing_counts": natural.sum(1).tolist(),
        "mean_missingness_correlation": (
            float(upper.mean()) if len(upper) else None
        ),
        "empirical_replay_supported": bool(natural.any()),
        "interpretation": (
            "Observed unavailability, unknown cause; zeros always legitimate"
        ),
    }


def make_mask(x, scenario, seed, profile=None, train=None):
    rng = np.random.default_rng(seed)
    obs = np.isfinite(x)
    mask = np.zeros_like(obs)
    T, N = x.shape
    kind = scenario["kind"]
    if kind == "mcar":
        idx = np.flatnonzero(obs)
        mask.flat[
            rng.choice(idx, round(len(idx) * scenario["rate"]), replace=False)
        ] = True
    elif kind in ["block", "synchronous"]:
        shared = (
            int(rng.integers(1, T - scenario.get("length", 144)))
            if kind == "synchronous"
            else None
        )
        for c in range(N):
            length = scenario["length"]
            if length >= T - 2:
                raise ValueError("Gap too large")
            start = (
                shared
                if shared is not None
                else int(rng.integers(1, T - length))
            )
            mask[start : start + length, c] = obs[start : start + length, c]
    elif kind == "empirical_replay":
        # Only train NaN, never zeros or quality-threshold exclusions.
        natural = np.isnan(train)
        if not natural.any():
            raise ValueError(
                "No natural missingness in training: empirical replay"
                " unavailable"
            )
        for t in range(0, T, 144):
            n = min(144, T - t)
            s = int(rng.integers(0, len(train) - n + 1))
            mask[t : t + n] = natural[s : s + n] & obs[t : t + n]
    else:
        raise ValueError(kind)
    if not mask.any():
        raise ValueError("No evaluable artificially masked cells")
    assert not (mask & ~obs).any()
    return mask
