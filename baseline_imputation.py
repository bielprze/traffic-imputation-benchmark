#!/usr/bin/env python3
"""
baseline_imputation.py

Baseline imputation methods compatible with data_imputation.py outputs.

Reads ONE wide CSV: timestamp + sensor columns
Optionally applies masking (same rules as your data_imputation.py):
  - choose columns (impute_columns OR impute_num_columns / "all")
  - for each chosen column: pick random start in [input_seq_len, T - num_records]
  - mask a contiguous block of length num_records
  - seed controls determinism -> SAME holes for same seed/params

Runs an imputation method:
  - per-sensor: ffill_bfill, median, rolling_mean, linear, spline, knn_columns
  - graph-aware (requires edges.csv): graph_neighbor, graph_laplacian, graph_temporal

Outputs (same convention as data_imputation.py):
1) data_filled.csv  (wide, ints)
2) data_masked.csv  (wide, with NaNs)
3) imputed_values.csv (long: timestamp,sensor,predicted,original,error sorted by timestamp)
4) imputation_summary.csv (same but sorted by sensor then timestamp)
5) metrics.json (MAE, RMSE, MRE + masking metadata)

NOTE:
- Final filled values are forced to non-negative integers (vehicle counts).
- Metrics are computed ONLY on newly-masked cells (not on pre-existing NaNs).
"""

from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as spla

try:
    from sklearn.impute import KNNImputer
except Exception:
    KNNImputer = None

import yaml


# ----------------------------- logging --------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ----------------------------- utils ----------------------------------

def cg_compat(A, b, x0=None, maxiter=200, rtol=1e-6, atol=0.0, M=None, callback=None):
    """SciPy cg compat wrapper."""
    try:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, rtol=rtol, atol=atol, M=M, callback=callback)
    except TypeError:
        return spla.cg(A, b, x0=x0, maxiter=maxiter, tol=rtol, M=M, callback=callback)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def finalize_to_vehicle_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Clip <0 to 0, round, cast to int64."""
    out = df.clip(lower=0)
    # for safety: keep NaNs if any still exist in filled (should not happen for masked cells)
    # but wide filled output in data_imputation is numeric; here we fill remaining NaNs with 0.
    out = out.fillna(0.0)
    arr = np.rint(out.values)
    return pd.DataFrame(arr, index=out.index, columns=out.columns).astype(np.int64)


def load_wide_csv(path: Path, timestamp_col: str = "timestamp") -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Expected wide CSV with at least 2 columns: timestamp + sensors")
    if timestamp_col not in df.columns:
        # fallback: first column
        timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col, kind="mergesort")
    timestamps = df[timestamp_col].copy()
    sensors_df = df.drop(columns=[timestamp_col])
    # ensure numeric (NaNs allowed)
    sensors_df = sensors_df.apply(pd.to_numeric, errors="coerce")
    return timestamps, sensors_df


def load_edges_csv(path: Path, undirected: bool = True) -> nx.Graph:
    edf = pd.read_csv(path)
    if edf.shape[1] < 2:
        raise ValueError("Edges CSV needs at least 2 columns: src,dst[,weight]")
    src_col, dst_col = edf.columns[0], edf.columns[1]
    w_col = edf.columns[2] if edf.shape[1] > 2 else None
    G = nx.Graph() if undirected else nx.DiGraph()
    for _, row in edf.iterrows():
        u = row[src_col]
        v = row[dst_col]
        w = float(row[w_col]) if w_col is not None else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(G[u][v].get("weight", 1.0), w)
        else:
            G.add_edge(u, v, weight=w)
    return G


# ------------------------- baseline methods ---------------------------

def impute_per_sensor(
    df: pd.DataFrame,
    method: str = "linear",
    limit: Optional[int] = None,
    rolling_window: int = 5,
    knn_k: int = 5
) -> pd.DataFrame:
    X = df.copy()
    if method == "ffill_bfill":
        return X.ffill(limit=limit).bfill(limit=limit)
    if method == "median":
        med = X.median(axis=0, skipna=True)
        return X.fillna(med).interpolate(method="time", limit=limit)
    if method == "rolling_mean":
        rm = X.rolling(rolling_window, min_periods=1).mean()
        return X.fillna(rm).interpolate(method="time", limit=limit)
    if method == "linear":
        return X.interpolate(method="time", limit=limit).ffill().bfill()
    if method == "spline":
        try:
            return X.interpolate(method="spline", order=3, limit=limit).ffill().bfill()
        except Exception:
            return X.interpolate(method="time", limit=limit).ffill().bfill()
    if method == "knn_columns":
        if KNNImputer is None:
            raise ImportError("scikit-learn is required for KNNImputer")
        imp = KNNImputer(n_neighbors=knn_k)
        arr = imp.fit_transform(X.values)
        return pd.DataFrame(arr, index=X.index, columns=X.columns)
    raise ValueError(f"Unknown method: {method}")


# ------------------------- graph helpers ------------------------------

def graph_laplacian(G: nx.Graph, nodes: Iterable[str]) -> sparse.csr_matrix:
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    data, row, col = [], [], []
    deg = np.zeros(len(nodes))
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            w = float(d.get("weight", 1.0))
            row += [i, j]
            col += [j, i]
            data += [-w, -w]
            deg[i] += w
            deg[j] += w
    row += list(range(len(nodes)))
    col += list(range(len(nodes)))
    data += deg.tolist()
    return sparse.coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes))).tocsr()


def impute_graph_neighbor_average(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    out = df.copy()
    nodes = list(out.columns)
    neigh = {n: list(G[n].items()) if n in G else [] for n in nodes}

    for t in range(len(out)):
        r = out.iloc[t]
        for n in nodes:
            if pd.isna(r[n]):
                s, wsum = 0.0, 0.0
                for v, attrs in neigh.get(n, []):
                    w = float(attrs.get("weight", 1.0))
                    val = r.get(v)
                    if not pd.isna(val):
                        s += w * val
                        wsum += w
                if wsum > 0:
                    out.iat[t, out.columns.get_loc(n)] = s / wsum

    if out.isna().to_numpy().any():
        out = out.interpolate(method="time").ffill().bfill()
    return out


def impute_graph_laplacian(df: pd.DataFrame, G: nx.Graph, lam: float = 1.0) -> pd.DataFrame:
    nodes = list(df.columns)
    L = graph_laplacian(G, nodes)
    out = df.copy().astype(float)
    warm = impute_graph_neighbor_average(out, G)

    for t_idx in range(len(out)):
        y = warm.iloc[t_idx].values.copy()
        obs_mask = ~df.iloc[t_idx].isna().values
        M = sparse.diags(obs_mask.astype(float), format="csr")
        A = M + lam * L
        b = M @ y

        x, info = cg_compat(A, b, x0=y, maxiter=200, rtol=1e-6, atol=0.0)
        if info != 0:
            try:
                x = spla.spsolve(A, b)
            except Exception:
                x = y

        out.iloc[t_idx] = x
        out.iloc[t_idx, obs_mask] = df.iloc[t_idx, obs_mask]
    return out


def impute_graph_temporal(df: pd.DataFrame, G: nx.Graph, lam_space: float = 1.0, lam_time: float = 0.1) -> pd.DataFrame:
    nodes = list(df.columns)
    T, N = df.shape
    L = graph_laplacian(G, nodes)
    I = sparse.identity(N, format="csr")

    masks = (~df.isna()).astype(float).values
    Y = df.fillna(0.0).values

    blocks, b_list = [], []
    for t in range(T):
        M = sparse.diags(masks[t], format="csr")
        D = M + lam_space * L
        if t > 0:
            D = D + lam_time * I
        if t < T - 1:
            D = D + lam_time * I
        blocks.append(D)
        b_list.append(M @ Y[t])

    # Build in LIL for efficient block assignments, then convert for solving
    A = sparse.block_diag(blocks, format="lil")

    off = (-lam_time * I).tocsr()  # keep off as sparse

    for t in range(T - 1):
        r0, r1 = t * N, (t + 1) * N
        c0, c1 = (t + 1) * N, (t + 2) * N

        # upper block
        A[r0:r1, c0:c1] = off
        # lower block
        A[c0:c1, r0:r1] = off

    # Convert once at the end (solver likes CSR/CSC)
    A = A.tocsr()

    b = np.concatenate(b_list)
    x, info = cg_compat(A, b, maxiter=500, rtol=1e-6, atol=0.0)
    if info != 0:
        try:
            x = spla.spsolve(A.tocsc(), b)
        except Exception:
            return impute_graph_laplacian(df, G, lam=lam_space)

    X = x.reshape(T, N)
    out = pd.DataFrame(X, index=df.index, columns=df.columns)

    obs_mask = ~df.isna().values
    out.values[obs_mask] = df.values[obs_mask]
    return out


# ------------------------- masking (same as data_imputation) ----------

def choose_columns_to_mask(
    sensor_names: List[str],
    impute_num_columns: Any,
    impute_columns: Optional[List[str]]
) -> List[str]:
    if impute_columns:
        missing = [c for c in impute_columns if c not in sensor_names]
        if missing:
            raise ValueError(f"impute_columns contains sensors not in CSV: {missing}")
        return list(impute_columns)

    if isinstance(impute_num_columns, str):
        if impute_num_columns.lower() == "all":
            return list(sensor_names)
        raise ValueError(f"Invalid impute_num_columns string: {impute_num_columns!r}. Use int or 'all'.")

    if not isinstance(impute_num_columns, int):
        raise ValueError(f"impute_num_columns must be int or 'all', got {type(impute_num_columns)}")

    if impute_num_columns <= 0:
        return []
    if impute_num_columns > len(sensor_names):
        raise ValueError(f"impute_num_columns={impute_num_columns} > num sensors={len(sensor_names)}")

    return random.sample(sensor_names, k=impute_num_columns)


def apply_block_masking(
    data: np.ndarray,                 # (T,N)
    sensor_names: List[str],
    num_records: int,
    columns_to_mask: List[str],
    input_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Same behavior as data_imputation.py:
    - per column, random start in [input_seq_len, T - num_records]
    - mask contiguous block length num_records
    - newly_masked_mask marks only cells that were NOT already NaN
    """
    T, N = data.shape
    if num_records <= 0:
        return data.copy(), np.zeros_like(data, dtype=bool), {}
    if num_records > T:
        raise ValueError("impute_num_records > number of rows")

    col_to_idx = {c: i for i, c in enumerate(sensor_names)}
    data_masked = data.copy()
    newly_masked = np.zeros_like(data_masked, dtype=bool)
    blocks: Dict[str, Tuple[int, int]] = {}

    min_start = input_seq_len
    max_start = T - num_records
    if max_start < min_start:
        raise ValueError(f"Not enough rows for masking: need T >= input_seq_len+num_records. T={T}")

    for col in columns_to_mask:
        j = col_to_idx[col]
        start = random.randint(min_start, max_start)
        end = start + num_records

        original_slice = data_masked[start:end, j]
        to_mask = ~np.isnan(original_slice)

        data_masked[start:end, j] = np.nan
        newly_masked[start:end, j] = to_mask

        blocks[col] = (start, end)

    return data_masked, newly_masked, blocks


# ------------------------- metrics + outputs --------------------------

def compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, Any]:
    mae = float(np.mean(np.abs(pred_vals - true_vals)))
    rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))

    abs_err = np.abs(pred_vals - true_vals)
    nonzero_mask = (np.abs(true_vals) > 0)
    num_mre_points = int(np.sum(nonzero_mask))
    num_mre_skipped_zeros = int(true_vals.size - num_mre_points)
    if num_mre_points == 0:
        mre = float("nan")
    else:
        mre = float(np.mean(abs_err[nonzero_mask] / np.abs(true_vals[nonzero_mask])))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MRE": mre,
        "num_imputed_points": int(true_vals.size),
        "num_mre_points": num_mre_points,
        "num_mre_skipped_zeros": num_mre_skipped_zeros,
    }


def save_outputs(
    timestamps: pd.Series,
    sensor_names: List[str],
    data_original: np.ndarray,          # (T,N)
    data_masked: np.ndarray,            # (T,N)
    newly_masked_mask: np.ndarray,      # (T,N) bool
    data_filled: np.ndarray,            # (T,N) ints
    out_filled_csv: Path,
    out_masked_csv: Path,
    out_imputed_values_csv: Path,
    out_summary_csv: Path,
    out_metrics_json: Path,
    apply_masking: bool,
    impute_num_records: int,
    impute_num_columns: Any,
    impute_columns: Optional[List[str]],
    mask_blocks: Dict[str, Tuple[int, int]],
):
    # 1) wide masked
    if apply_masking:
        df_masked = pd.DataFrame(data_masked, columns=sensor_names)
        df_masked.insert(0, "timestamp", timestamps.values)
        ensure_parent(out_masked_csv)
        df_masked.to_csv(out_masked_csv, index=False)
        logging.info("Saved masked wide CSV: %s", out_masked_csv)

    # 2) wide filled
    df_filled = pd.DataFrame(data_filled, columns=sensor_names)
    df_filled.insert(0, "timestamp", timestamps.values)
    ensure_parent(out_filled_csv)
    df_filled.to_csv(out_filled_csv, index=False)
    logging.info("Saved filled wide CSV: %s", out_filled_csv)

    # long rows only for newly masked cells
    idxs = np.where(newly_masked_mask)
    rows = []
    if idxs[0].size > 0:
        for r, c in zip(idxs[0], idxs[1]):
            orig = int(round(float(data_original[r, c])))
            pred = int(data_filled[r, c])
            rows.append({
                "timestamp": timestamps.iloc[r],
                "sensor": sensor_names[c],
                "predicted": pred,
                "original": orig,
                "error": int(pred - orig),
            })

    df_long = pd.DataFrame(rows, columns=["timestamp", "sensor", "predicted", "original", "error"])

    # 3) imputed_values.csv sorted by timestamp
    ensure_parent(out_imputed_values_csv)
    if not df_long.empty:
        df_time = df_long.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
        df_time = df_time.sort_values(by=["timestamp", "sensor"], kind="mergesort")
        df_time.to_csv(out_imputed_values_csv, index=False)
    else:
        df_long.to_csv(out_imputed_values_csv, index=False)
    logging.info("Saved imputed_values.csv: %s", out_imputed_values_csv)

    # 4) imputation_summary.csv sorted by sensor then timestamp
    ensure_parent(out_summary_csv)
    if not df_long.empty:
        df_sum = df_long.copy()
        df_sum["timestamp"] = pd.to_datetime(df_sum["timestamp"])
        df_sum = df_sum.sort_values(by=["sensor", "timestamp"], kind="mergesort")
        df_sum.to_csv(out_summary_csv, index=False)
    else:
        df_long.to_csv(out_summary_csv, index=False)
    logging.info("Saved imputation_summary.csv: %s", out_summary_csv)

    # 5) metrics.json
    if idxs[0].size == 0:
        metrics_core = {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MRE": float("nan"),
            "num_imputed_points": 0,
            "num_mre_points": 0,
            "num_mre_skipped_zeros": 0,
        }
    else:
        true_vals = data_original[idxs]
        pred_vals = data_filled[idxs].astype(float)
        metrics_core = compute_metrics(true_vals=true_vals, pred_vals=pred_vals)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    method = str(cfg.get("baseline_method", "linear"))
    
    metrics = {
        **metrics_core,
        "method": method,
        "apply_masking": apply_masking,
        "impute_num_records": int(impute_num_records),
        "impute_num_columns": (impute_num_columns if isinstance(impute_num_columns, str) else int(impute_num_columns)),
        "impute_columns": list(impute_columns) if impute_columns else None,
        "mask_blocks": {k: {"start": v[0], "end_exclusive": v[1]} for k, v in mask_blocks.items()},
    }

    ensure_parent(out_metrics_json)
    out_metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Saved metrics.json: %s", out_metrics_json)


# ------------------------------ main ----------------------------------

def run_imputer(
    method: str,
    df_masked_indexed: pd.DataFrame,
    edges_path: Optional[Path],
    lam: float,
    lam_time: float,
) -> pd.DataFrame:
    if method in {"linear", "ffill_bfill", "median", "rolling_mean", "spline", "knn_columns"}:
        return impute_per_sensor(df_masked_indexed, method=method)
    if method == "graph_neighbor":
        if not edges_path:
            raise SystemExit("--edges (or edges_csv in config) required for graph methods")
        G = load_edges_csv(edges_path)
        return impute_graph_neighbor_average(df_masked_indexed, G)
    if method == "graph_laplacian":
        if not edges_path:
            raise SystemExit("--edges (or edges_csv in config) required for graph methods")
        G = load_edges_csv(edges_path)
        return impute_graph_laplacian(df_masked_indexed, G, lam=lam)
    if method == "graph_temporal":
        if not edges_path:
            raise SystemExit("--edges (or edges_csv in config) required for graph methods")
        G = load_edges_csv(edges_path)
        return impute_graph_temporal(df_masked_indexed, G, lam_space=lam, lam_time=lam_time)
    raise SystemExit(f"Unknown method: {method}")


def main():
    setup_logging()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Inputs
    data_csv = Path(cfg["data_csv"])
    # Optional graph edges source (only for graph_* methods)
    edges_csv = Path(cfg["edges_csv"]) if cfg.get("edges_csv") else None

    # Masking config (same keys as data_imputation.py)
    apply_masking = bool(cfg.get("apply_masking", False))
    impute_num_records = int(cfg.get("impute_num_records", 0))
    impute_num_columns = cfg.get("impute_num_columns", 0)   # int or "all"
    impute_columns = cfg.get("impute_columns", None)        # list or null
    input_seq_len = int(cfg.get("input_seq_len", 12))

    # Seed (same convention: SEEDS[0])
    seeds = cfg.get("seeds", [0])
    seed = int(seeds[0]) if isinstance(seeds, list) and len(seeds) > 0 else int(seeds)
    set_seed(seed)

    # Outputs (same naming style as data_imputation.py)
    out_filled_csv = Path(cfg.get("output_csv", "data_filled.csv"))
    out_masked_csv = Path(cfg.get("masked_csv", "data_masked.csv"))
    out_imputed_values_csv = Path(cfg.get("filled_csv", "imputed_values.csv"))
    out_summary_csv = Path(cfg.get("summary_csv", "imputation_summary.csv"))
    out_metrics_json = Path(cfg.get("metrics_json", "metrics.json"))

    # Method config
    method = str(cfg.get("baseline_method", "linear"))
    lam = float(cfg.get("lam", 1.0))
    lam_time = float(cfg.get("lam_time", 0.1))

    logging.info("Loaded config. method=%s, seed=%d, apply_masking=%s", method, seed, apply_masking)

    # Load data
    timestamps, sensors_df = load_wide_csv(data_csv, timestamp_col="timestamp")
    sensor_names = list(sensors_df.columns)
    T, N = sensors_df.shape
    logging.info("Loaded data: T=%d, N=%d", T, N)

    data_original = sensors_df.to_numpy(dtype=float)  # (T,N) may contain NaNs

    # Apply masking (or not)
    if apply_masking:
        if impute_num_records <= 0:
            raise ValueError("apply_masking=true but impute_num_records <= 0")
        cols_to_mask = choose_columns_to_mask(sensor_names, impute_num_columns, impute_columns)
        data_masked, newly_masked_mask, mask_blocks = apply_block_masking(
            data=data_original,
            sensor_names=sensor_names,
            num_records=impute_num_records,
            columns_to_mask=cols_to_mask,
            input_seq_len=input_seq_len,
        )
        logging.info("Masking enabled: columns=%d, block_len=%d", len(cols_to_mask), impute_num_records)
    else:
        data_masked = data_original.copy()
        newly_masked_mask = np.zeros_like(data_masked, dtype=bool)
        mask_blocks = {}
        logging.info("Masking disabled.")

    # Run imputer
    # Create indexed DF for time interpolation methods
    df_masked_indexed = pd.DataFrame(data_masked, index=pd.to_datetime(timestamps.values), columns=sensor_names)
    df_masked_indexed.index.name = "timestamp"

    filled_df = run_imputer(
        method=method,
        df_masked_indexed=df_masked_indexed,
        edges_path=edges_csv,
        lam=lam,
        lam_time=lam_time,
    )

    # Force integers (vehicle counts)
    filled_int_df = finalize_to_vehicle_counts(filled_df)
    data_filled = filled_int_df.to_numpy(dtype=np.int64)

    # Save outputs + metrics (metrics only on newly masked cells)
    save_outputs(
        timestamps=timestamps,
        sensor_names=sensor_names,
        data_original=data_original,
        data_masked=data_masked,
        newly_masked_mask=newly_masked_mask,
        data_filled=data_filled,
        out_filled_csv=out_filled_csv,
        out_masked_csv=out_masked_csv,
        out_imputed_values_csv=out_imputed_values_csv,
        out_summary_csv=out_summary_csv,
        out_metrics_json=out_metrics_json,
        apply_masking=apply_masking,
        impute_num_records=impute_num_records,
        impute_num_columns=impute_num_columns,
        impute_columns=impute_columns,
        mask_blocks=mask_blocks,
    )

    logging.info("DONE.")


if __name__ == "__main__":
    main()
