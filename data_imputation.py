
"""
data_imputation.py

Imputation using trained SAGE-Voronoi / ST-GCN model.

NEW:
- Reads one wide CSV: timestamp + sensor columns.
- Filters geojson to sensors present in CSV.
- Optional masking (apply_masking flag):
  For each masked column: choose random start index, then mask a contiguous block
  of impute_num_records rows: start..start+K-1
- Imputes ONLY NaNs (does not overwrite existing values).
"""

import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import yaml
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.losses import MeanSquaredError
from keras.optimizers import RMSprop

import helper
from gcnnmodel import (
    GraphInfo,
    LSTMGC,
    compute_adjacency_matrix,
    calculate_adjacency_matrix_Delaunay,
    calculate_route_distance,
    preprocess,
)

# ─── Configuration ──────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_CSV     = Path(cfg["data_csv"])
GEOJSON_PATH = Path(cfg["geojson_path"])
SAVE_FOLDER  = Path(cfg["save_folder"])

# outputs
OUTPUT_CSV   = Path(cfg.get("output_csv", "data_filled.csv"))  # full filled wide csv
MASKED_CSV   = Path(cfg.get("masked_csv", "data_masked.csv"))  # optional write-out
METRICS_JSON = Path(cfg.get("metrics_json", "metrics.json"))
FILLED_CSV   = Path(cfg.get("filled_csv", "imputed_values.csv"))  # list of imputed points

TRAIN_SIZE = cfg["train_size"]
VAL_SIZE   = cfg["val_size"]

SIGMA2  = cfg["sigma2"]
EPSILON = cfg["epsilon"]

USE_DELAUNAY       = cfg["use_delaunay"]
MAX_DEPTH          = cfg["max_depth"]
ADJ_SCALER_OPTIONS = cfg["adj_scaler_options"]

IN_FEAT           = cfg["in_feat"]
OUT_FEAT          = cfg["out_feat"]
LSTM_UNITS        = cfg["lstm_units"]
GRAPH_CONV_PARAMS = cfg["graph_conv_params"]
INPUT_SEQ_LEN     = cfg["input_seq_len"]

HORIZON = 1
SEEDS   = cfg["seeds"]

AUTOREGRESSION = bool(cfg.get("autoregressive_imputation", False))
print(f"\n\n{AUTOREGRESSION}\n\n")
# masking config (NEW)
APPLY_MASKING      = bool(cfg.get("apply_masking", False))
IMPUTE_NUM_RECORDS = int(cfg.get("impute_num_records", 0))
IMPUTE_NUM_COLUMNS = cfg.get("impute_num_columns", 0)
IMPUTE_COLUMNS     = cfg.get("impute_columns", None)  # optional list

MASKED_VALUES_CSV = Path(cfg.get("masked_values_csv", "masked_values.csv"))
MASKED_VALUES_CSV.parent.mkdir(parents=True, exist_ok=True)

GUI_RESULTS_CSV = Path(cfg.get("gui_results_csv", "gui_results.csv"))
# ─── Utilities ─────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs():
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FILLED_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    MASKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    GUI_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_data_from_wide_csv() -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[float, float]]]:
    """
    Load wide csv + filtered geojson points.

    Returns:
      timestamps (T,)
      data (T,N)
      sensor_names list length N
      sensors_points dict length N (ordered)
    """
    timestamps, data, sensor_names, sensors_points = helper.load_wide_csv_and_filter_geojson(
        str(DATA_CSV),
        str(GEOJSON_PATH),
        timestamp_col="timestamp",
        exclude_keys=helper.EXCLUDE_KEYS,
    )
    logging.info("Loaded CSV: T=%d, N=%d", data.shape[0], data.shape[1])
    return timestamps, data, sensor_names, sensors_points


def choose_columns_to_mask(sensor_names: List[str]) -> List[str]:
    """
    Determine which columns to mask based on config.
    Rules:
      - If IMPUTE_COLUMNS is provided → use exactly those
      - If IMPUTE_NUM_COLUMNS == "all" → use all available sensors
      - If IMPUTE_NUM_COLUMNS == 0 → mask none
      - Else → random sample of size IMPUTE_NUM_COLUMNS
    """
    if IMPUTE_COLUMNS:
        missing = [c for c in IMPUTE_COLUMNS if c not in sensor_names]
        if missing:
            raise ValueError(f"impute_columns contains sensors not in CSV: {missing}")
        return list(IMPUTE_COLUMNS)

    if isinstance(IMPUTE_NUM_COLUMNS, str):
        if IMPUTE_NUM_COLUMNS.lower() == "all":
            return list(sensor_names)
        else:
            raise ValueError(
                f"Invalid impute_num_columns string value: {IMPUTE_NUM_COLUMNS!r}. "
                "Use an integer or 'all'."
            )

    # numeric case
    if not isinstance(IMPUTE_NUM_COLUMNS, int):
        raise ValueError(
            f"impute_num_columns must be int or 'all', got {type(IMPUTE_NUM_COLUMNS)}"
        )

    if IMPUTE_NUM_COLUMNS <= 0:
        return []

    if IMPUTE_NUM_COLUMNS > len(sensor_names):
        raise ValueError(
            f"impute_num_columns={IMPUTE_NUM_COLUMNS} > number of available sensors={len(sensor_names)} "
            "(after geojson/exclude filtering)"
        )

    return random.sample(sensor_names, k=IMPUTE_NUM_COLUMNS)


def apply_block_masking(
    data: np.ndarray,
    sensor_names: List[str],
    num_records: int,
    columns_to_mask: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    For each chosen column, select a random start index and mask a contiguous block:
    start..start+num_records-1 (inclusive).

    Returns:
      data_masked: copy with NaNs inserted
      newly_masked_mask: boolean mask True where we *newly* inserted NaN
      blocks_info: dict col -> (start, end_exclusive)
    """
    T, N = data.shape
    if num_records <= 0:
        return data.copy(), np.zeros_like(data, dtype=bool), {}

    if num_records > T:
        raise ValueError("impute_num_records is larger than number of rows in data.")

    col_to_idx = {c: i for i, c in enumerate(sensor_names)}
    data_masked = data.copy()
    newly_masked = np.zeros_like(data_masked, dtype=bool)
    blocks_info: Dict[str, Tuple[int, int]] = {}
    min_start = INPUT_SEQ_LEN
    max_start = T - num_records
    for col in columns_to_mask:
        j = col_to_idx[col]
        start = random.randint(min_start, max_start)
        end = start + num_records  # exclusive

        # only mark positions that are not already NaN as "newly masked"
        original_slice = data_masked[start:end, j]
        to_mask = ~np.isnan(original_slice)

        data_masked[start:end, j] = np.nan
        newly_masked[start:end, j] = to_mask

        blocks_info[col] = (start, end)

    return data_masked, newly_masked, blocks_info


def build_graph(sensors_points: Dict[str, Tuple[float, float]]) -> GraphInfo:
    """Rebuild graph structure from filtered sensors."""
    route_distances, all_points = calculate_route_distance(sensors_points)

    if USE_DELAUNAY:
        adj = calculate_adjacency_matrix_Delaunay(all_points, MAX_DEPTH, ADJ_SCALER_OPTIONS[0])
    else:
        adj = compute_adjacency_matrix(route_distances, SIGMA2, EPSILON)

    idx, nbr = np.where(adj > 0.0)
    weights = adj[idx, nbr]

    graph = GraphInfo(
        edges=(idx.tolist(), nbr.tolist(), weights.tolist()),
        num_nodes=adj.shape[0],
    )
    logging.info("Graph: %d nodes, %d edges", graph.num_nodes, len(idx))
    return graph


def build_model(graph: GraphInfo, horizon: int) -> Model:
    stgcn = LSTMGC(
        IN_FEAT,
        OUT_FEAT,
        LSTM_UNITS,
        INPUT_SEQ_LEN,
        horizon,
        graph,
        GRAPH_CONV_PARAMS,
    )
    inputs = layers.Input(shape=(INPUT_SEQ_LEN, graph.num_nodes, IN_FEAT))
    outputs = stgcn(inputs)
    model = Model(inputs, outputs)
    model.compile(optimizer=RMSprop(2e-4), loss=MeanSquaredError())
    return model


def scale_full_series(data_for_scaler: np.ndarray):
    """
    Fit scaler exactly as in training (via preprocess on full series),
    then transform full series.

    IMPORTANT: preprocess expects no NaNs. We use a NaN-filled dataset only for imputation,
    so here we fit scaler on a *complete* reference. We choose to fit on the original data
    with NaNs replaced by 0 for scaler-fitting only.

    If your original data has NaNs, consider preprocessing them properly (ffill/median etc).
    """
    data_tmp = data_for_scaler.copy()
    if np.isnan(data_tmp).any():
        # minimal, deterministic fill just for scaler fitting
        data_tmp = np.nan_to_num(data_tmp, nan=0.0)

    _, _, _, scaler = preprocess(data_tmp, TRAIN_SIZE, VAL_SIZE)
    scaled_full = scaler.transform(data_tmp)
    return scaled_full, scaler


def build_input_windows(scaled_full: np.ndarray, graph: GraphInfo):
    """
    Build windows over the entire scaled series.

    Returns:
      X : (num_windows, INPUT_SEQ_LEN, num_nodes, IN_FEAT)
      target_time_indices : global timestep indices for prediction
    """
    T, N = scaled_full.shape
    assert N == graph.num_nodes, "Data sensors must match graph num_nodes"

    windows = []
    target_indices = []

    for t in range(INPUT_SEQ_LEN, T):
        start = t - INPUT_SEQ_LEN
        end = t
        window = scaled_full[start:end, :]      # (INPUT_SEQ_LEN, N)
        windows.append(window[..., np.newaxis]) # -> (INPUT_SEQ_LEN, N, 1)
        target_indices.append(t)

    X = np.stack(windows, axis=0)
    target_indices = np.array(target_indices)
    return X, target_indices


def predict_next_step(model: Model, X: np.ndarray) -> np.ndarray:
    """
    Predict in scaled space, return (num_windows, N) for horizon=1.
    Handles both y_pred shapes.
    """
    y_pred = model.predict(X, verbose=0)

    if y_pred.ndim == 3:
        # (batch, horizon, N)
        y_pred_scaled = y_pred[:, 0, :]
    elif y_pred.ndim == 4:
        # (batch, horizon, N, out_feat)
        y_pred_scaled = y_pred[:, 0, :, 0]
    else:
        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

    return y_pred_scaled


def impute_only_nans(
    data_masked: np.ndarray,
    y_pred_unscaled: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """
    Create filled data by replacing ONLY NaNs at target times t>=INPUT_SEQ_LEN.

    data_masked: (T,N)
    y_pred_unscaled: (num_windows, N) corresponds to targets at target_indices
    """
    filled = data_masked.copy()
    for i, t in enumerate(target_indices):
        nan_mask = np.isnan(filled[t, :])
        if nan_mask.any():
            filled[t, nan_mask] = y_pred_unscaled[i, nan_mask]
    return filled


def evaluate_and_save(
    timestamps: np.ndarray,
    sensor_names: List[str],
    data_original: np.ndarray,
    data_masked: np.ndarray,
    newly_masked_mask: np.ndarray,
    filled: np.ndarray,
    blocks_info: Dict[str, Tuple[int, int]],
):
    """
    Compute metrics only on points that were newly masked (not original NaNs),
    and save:
      - metrics.json
      - imputed_values.csv (long format)
      - output_csv (wide filled)
      - masked_csv (wide masked) if APPLY_MASKING
    """
    # Save masked (optional)
    if APPLY_MASKING:
        df_masked = pd.DataFrame(data_masked, columns=sensor_names)
        df_masked.insert(0, "timestamp", timestamps)
        df_masked.to_csv(MASKED_CSV, index=False)
        logging.info("Saved masked CSV to %s", MASKED_CSV)

    # Save filled wide CSV
    df_filled_wide = pd.DataFrame(filled, columns=sensor_names)
    df_filled_wide.insert(0, "timestamp", timestamps)
    df_filled_wide.to_csv(OUTPUT_CSV, index=False)
    logging.info("Saved filled wide CSV to %s", OUTPUT_CSV)

    # Metrics on newly masked positions
    idxs = np.where(newly_masked_mask)
    if idxs[0].size == 0:
        mae = float("nan")
        rmse = float("nan")
        mre = float("nan")
        num_mre_points = 0
        num_mre_skipped_zeros = 0
        num_pts = 0
        rows = []
        logging.warning("No newly masked points to evaluate (maybe masking off or all were already NaN).")
    else:
        true_vals = data_original[idxs]
        pred_vals = filled[idxs]
        mae = float(np.mean(np.abs(pred_vals - true_vals)))
        rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        num_pts = int(true_vals.size)

        # long-format rows
        rows = []
        for r, c in zip(idxs[0], idxs[1]):
            rows.append({
                "timestamp": timestamps[r],
                "sensor": sensor_names[c],
                "predicted": float(filled[r, c]),
                "original": float(data_original[r, c]),
                "error": float(filled[r, c] - data_original[r, c]),
            })
        
        abs_err = np.abs(pred_vals - true_vals)
        # MRE: mean(|err| / |true|), skipping true == 0 to avoid division by zero
        nonzero_mask = (np.abs(true_vals) > 0)
        num_mre_points = int(np.sum(nonzero_mask))
        num_mre_skipped_zeros = int(true_vals.size - num_mre_points)

        if num_mre_points == 0:
            mre = float("nan")
        else:
            mre = float(np.mean(abs_err[nonzero_mask] / np.abs(true_vals[nonzero_mask])))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MRE": mre,
        "num_imputed_points": num_pts,
        "num_mre_points": num_mre_points,
        "num_mre_skipped_zeros": num_mre_skipped_zeros,
        "apply_masking": APPLY_MASKING,
        "impute_num_records": IMPUTE_NUM_RECORDS,
        "impute_num_columns": IMPUTE_NUM_COLUMNS if not IMPUTE_COLUMNS else None,
        "impute_columns": list(IMPUTE_COLUMNS) if IMPUTE_COLUMNS else None,
        "mask_blocks": {k: {"start": v[0], "end_exclusive": v[1]} for k, v in blocks_info.items()},
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    logging.info("Saved metrics to %s", METRICS_JSON)

    df_long = pd.DataFrame(rows)
    df_long.to_csv(FILLED_CSV, index=False)
    # --- Save masked values list (long format) ---
    # --- Additional GUI-format CSV: timestamp, sensor, predicted, original, error (sorted) ---
    if not df_long.empty:
        df_gui = df_long[["timestamp", "sensor", "predicted", "original", "error"]].copy()
        df_gui["timestamp"] = pd.to_datetime(df_gui["timestamp"])

        # Sort: sensor first, then timestamp
        df_gui = df_gui.sort_values(
            by=["sensor", "timestamp"],
            ascending=[True, True],
            kind="mergesort"
        )

        df_gui.to_csv(GUI_RESULTS_CSV, index=False)
        logging.info("Saved GUI results CSV to %s", GUI_RESULTS_CSV)
    else:
        # still create an empty file with headers, so GUI doesn't crash
        pd.DataFrame(columns=["timestamp", "sensor", "predicted", "original", "error"]).to_csv(
            GUI_RESULTS_CSV, index=False
        )
        logging.info("Saved empty GUI results CSV to %s", GUI_RESULTS_CSV)

    
    rows_masked = []
    idxs = np.where(newly_masked_mask)
    for r, c in zip(idxs[0], idxs[1]):
        rows_masked.append({
            "timestamp": timestamps[r],
            "row_index": int(r),
            "sensor": sensor_names[c],
            "original_value": float(data_original[r, c]),
            "masked_value": np.nan,
        })

    df_masked_long = pd.DataFrame(rows_masked)
    df_masked_long.to_csv(MASKED_VALUES_CSV, index=False)
    logging.info("Saved masked values list to %s", MASKED_VALUES_CSV)

    logging.info("Saved imputed points list to %s", FILLED_CSV)
    logging.info("DONE saving all outputs.")


def autoregressive_fill_batched(
    model: Model,
    data_masked: np.ndarray,     # (T, N) in original units, may contain NaNs
    scaler,
    input_seq_len: int,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Autoregressive imputation, but batched to avoid calling model.predict() per-timestep.

    Strategy:
      - Iterate time forward (AR is preserved).
      - Collect up to `batch_size` timestamps that currently have NaNs.
      - For each timestamp t, build its window from the CURRENT scaled series.
      - Predict all windows in one model.predict() call.
      - Fill NaNs at those timestamps, update scaled series, continue.

    This avoids thousands of tiny predict() calls that can crash TF on Windows.
    """
    filled = data_masked.copy()
    T, N = filled.shape

    # Prepare scaled series from current filled, with NaNs replaced for scaling only
    filled_for_scale = np.nan_to_num(filled.copy(), nan=0.0)
    scaled = scaler.transform(filled_for_scale)  # (T, N)

    def _predict_scaled_batch(Xb: np.ndarray) -> np.ndarray:
        """
        Predict in scaled space for a batch:
          Xb: (B, seq, N, 1)
        Return:
          y_scaled: (B, N) for horizon=1, using the first output feature if OUT_FEAT>1.
        """
        y = model.predict(Xb, verbose=0)

        # Common shapes:
        # (B, horizon, N)
        # (B, horizon, N, out_feat)
        if y.ndim == 3:
            y_scaled = y[:, 0, :]          # (B, N)
        elif y.ndim == 4:
            y_scaled = y[:, 0, :, :].mean(axis=-1)       # (B, N) - first feature channel
        else:
            raise ValueError(f"Unexpected y_pred shape: {y.shape}")

        if y_scaled.shape[1] != N:
            raise ValueError(f"Pred shape mismatch: got {y_scaled.shape}, expected (B,{N})")

        return y_scaled

    t = input_seq_len
    while t < T:
        # Build a batch of timestamps that currently need filling
        t_list = []
        while t < T and len(t_list) < batch_size:
            if np.isnan(filled[t, :]).any():
                t_list.append(t)
            t += 1

        if not t_list:
            continue  # no missing in this range, keep scanning

        # Build batch windows from CURRENT scaled series
        X_batch = np.stack(
            [scaled[tt - input_seq_len : tt, :] for tt in t_list],
            axis=0
        )[..., np.newaxis]  # (B, seq, N, 1)

        # Predict scaled
        y_pred_scaled = _predict_scaled_batch(X_batch)  # (B, N)

        # Inverse transform -> original units (B, N)
        y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

        # Enforce non-negative integer vehicle counts
        y_pred_unscaled = np.clip(y_pred_unscaled, 0, None)
        y_pred_unscaled = np.rint(y_pred_unscaled).astype(int)

        # Fill only NaNs at each timestamp, then update scaled series for AR
        for bi, tt in enumerate(t_list):
            nan_mask = np.isnan(filled[tt, :])
            if not nan_mask.any():
                continue

            filled[tt, nan_mask] = y_pred_unscaled[bi, nan_mask]

            # update both filled_for_scale and scaled for future windows
            filled_for_scale[tt, nan_mask] = filled[tt, nan_mask]
            scaled[tt, :] = scaler.transform(filled_for_scale[tt, :].reshape(1, -1))[0]

        # Optional progress log
        # logging.info("AR batch filled: %d timestamps, t=%d/%d", len(t_list), t, T)

    return filled


def main():
    setup_logging()
    ensure_dirs()

    # deterministic seed base
    seed = SEEDS[0]
    set_seed(seed)

    # 1) Load data + filtered geojson
    timestamps, data_original, sensor_names, sensors_points = load_data_from_wide_csv()

    # 2) Optional masking
    columns_to_mask = []
    blocks_info = {}
    if APPLY_MASKING:
        if IMPUTE_NUM_RECORDS <= 0:
            raise ValueError("apply_masking=true but impute_num_records <= 0")
        columns_to_mask = choose_columns_to_mask(sensor_names)
        if not columns_to_mask:
            logging.warning("apply_masking=true but no columns selected to mask.")
        data_masked, newly_masked_mask, blocks_info = apply_block_masking(
            data_original,
            sensor_names,
            IMPUTE_NUM_RECORDS,
            columns_to_mask,
        )
        logging.info("Masking enabled. Masked columns=%d", len(columns_to_mask))
    else:
        data_masked = data_original.copy()
        newly_masked_mask = np.zeros_like(data_masked, dtype=bool)
        blocks_info = {}

    # Also respect any pre-existing NaNs: imputation should fill them too (but metrics only on newly_masked)
    # We'll fill *all* NaNs; newly_masked_mask controls evaluation points.

    # 3) Build graph & model, load weights
    graph = build_graph(sensors_points)
    model = build_model(graph, HORIZON)

    horizon = HORIZON
    adj_scaler = ADJ_SCALER_OPTIONS[0]

    weight_file = (
        SAVE_FOLDER
        / f"darmstadt_h{horizon}_s{seed}_"
        f"{'D' if USE_DELAUNAY else 'S'}_"
        f"d{MAX_DEPTH}_a{adj_scaler}.weights.h5"
    )
    model.load_weights(str(weight_file))
    logging.info("Loaded weights from %s", weight_file)

    # 4) Fit scaler + build windows
    logging.info("STEP 4: fitting scaler + building windows...")
    scaled_full, scaler = scale_full_series(data_original)
    X, target_indices = build_input_windows(scaled_full, graph)
    logging.info("STEP 4 DONE: X shape=%s, targets=%s", X.shape, target_indices.shape)
    logging.info("STEP 5: imputation (autoregression=%s)...", AUTOREGRESSION)
    # 5) Predict + inverse transform
    if AUTOREGRESSION:
        filled = autoregressive_fill_batched(
            model=model,
            data_masked=data_masked,
            scaler=scaler,
            input_seq_len=INPUT_SEQ_LEN,
            batch_size=int(cfg.get("ar_batch_size", 64)),
    )
    else:
        y_pred_scaled = predict_next_step(model, X)  # (num_windows, N)
        y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)
        y_pred_unscaled = np.clip(y_pred_unscaled, 0, None)
        y_pred_unscaled = np.rint(y_pred_unscaled).astype(int)
        # 6) Fill only NaNs in masked data (and any existing NaNs)
        filled = impute_only_nans(data_masked, y_pred_unscaled, target_indices)
    logging.info("STEP 5 DONE: filled shape=%s, num_nans_after=%d", filled.shape, int(np.isnan(filled).sum()))
    # 7) Save results + metrics
    logging.info("STEP 7: saving outputs...")
    evaluate_and_save(
        timestamps=timestamps,
        sensor_names=sensor_names,
        data_original=data_original,
        data_masked=data_masked,
        newly_masked_mask=newly_masked_mask,
        filled=filled,
        blocks_info=blocks_info,
    )


if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    try:
        main()
    except Exception as e:
        logging.exception("FATAL ERROR: %s", e)
        raise
