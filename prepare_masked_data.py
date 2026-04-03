#!/usr/bin/env python3
"""
prepare_masked_data.py
----------------------
Create a masked version of a sensor CSV by randomly hiding a percentage of
observed values (turning them into NaN). The first column is assumed to be a
Timestamp and is preserved. All other columns are treated as sensors.

Outputs:
  - masked CSV with the same layout as the input
  - optional mask map CSV (same shape as data columns only) with 1 where this
    script masked a value, else 0
  - optional "removed values" CSV with the same layout as the input but only the
    removed values retained (non-removed cells are NaN)
  - optional "removed values (long)" CSV with rows: [timestamp, sensor, value]

Usage examples:
  python prepare_masked_data.py \
      --input original_data.csv \
      --output masked_data.csv \
      --percent 10 \
      --maskmap mask_map.csv \
      --removed removed_values.csv \
      --removed_long removed_values_long.csv \
      --seed 123

Notes:
  * Only non-NaN cells are eligible to be masked.
  * The target number of masked cells is rounded to the nearest integer.
  * By default, at least one value per sensor column is kept unmasked if possible
    (use --allow_empty_columns to disable this constraint).
"""
from __future__ import annotations

import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def load_sensor_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (timestamp + >=1 sensor)")
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        # If parsing failed entirely, leave as is but warn in printout later.
        pass
    return df


def choose_mask_indices(
    data_vals: np.ndarray,
    percent: float,
    rng: np.random.Generator,
    ensure_nonempty_cols: bool = True,
) -> Tuple[np.ndarray, int]:
    """Return a boolean mask (same shape as data_vals) indicating positions to mask.

    Only True where data_vals is not NaN. `percent` is in [0, 100].
    If ensure_nonempty_cols, we avoid masking all non-NaN entries of any column
    when feasible.
    """
    if not (0.0 <= percent <= 100.0):
        raise ValueError("percent must be between 0 and 100")

    observed = ~np.isnan(data_vals)
    M = observed.sum()
    if M == 0:
        return np.zeros_like(observed, dtype=bool), 0

    target = int(np.rint(M * (percent / 100.0)))
    if target == 0:
        return np.zeros_like(observed, dtype=bool), 0

    mask = np.zeros_like(observed, dtype=bool)

    # Flatten indices of observed cells
    obs_idx = np.flatnonzero(observed.ravel())
    if target > len(obs_idx):
        target = len(obs_idx)

    # Optionally reserve one observed per column
    if ensure_nonempty_cols:
        rows, cols = observed.shape
        for c in range(cols):
            col_obs = np.flatnonzero(observed[:, c])
            if col_obs.size > 0:
                # Reserve one random observed index in this column to remain unmasked
                keep_r = int(rng.choice(col_obs))
                # Remove this (row, col) from candidate list if present
                flat_k = keep_r * cols + c
                obs_idx = obs_idx[obs_idx != flat_k]
        # Recompute target in case we removed many candidates
        target = min(target, len(obs_idx))

    if target > 0:
        chosen = rng.choice(obs_idx, size=target, replace=False)
        mask_flat = np.zeros(observed.size, dtype=bool)
        mask_flat[chosen] = True
        mask = mask_flat.reshape(observed.shape)

    return mask, int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description="Randomly mask a percentage of sensor data.")
    ap.add_argument("--input", required=True, help="Path to original_data.csv")
    ap.add_argument("--output", required=True, help="Where to write masked_data.csv")
    ap.add_argument("--percent", type=float, required=True, help="Percent of observed values to mask (0-100)")
    ap.add_argument("--maskmap", default=None, help="Optional path to write mask_map.csv (1=masked by this script, 0=otherwise)")
    ap.add_argument("--removed", default=None, help="Optional path to write removed_values.csv (same layout; NaN elsewhere)")
    ap.add_argument("--removed_long", default=None, help="Optional path to write removed values in long format [timestamp,sensor,value]")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--allow_empty_columns", action="store_true", help="Allow entire columns to become fully masked if necessary")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = load_sensor_csv(args.input)
    ts_col = df.columns[0]
    sensors = df.columns[1:]

    data = df[sensors].astype(float)
    vals = data.values.astype(float)

    mask, k = choose_mask_indices(
        vals,
        percent=args.percent,
        rng=rng,
        ensure_nonempty_cols=(not args.allow_empty_columns),
    )

    masked_vals = vals.copy()
    masked_vals[mask] = np.nan
    masked_df = pd.concat([df[[ts_col]], pd.DataFrame(masked_vals, columns=sensors)], axis=1)

    masked_df.to_csv(args.output, index=False)
    print(f"Wrote masked data to {args.output} (masked {k} cells = {k/np.isfinite(vals).sum()*100:.2f}% of observed)")

    if args.maskmap:
        mask_map_df = pd.DataFrame(mask.astype(int), columns=sensors)
        mask_map_df.to_csv(args.maskmap, index=False)
        print(f"Wrote mask map to {args.maskmap}")

    # Optional: write removed values in wide format (same schema as input)
    if args.removed:
        removed_vals = np.where(mask, vals, np.nan)
        removed_wide_df = pd.concat([df[[ts_col]], pd.DataFrame(removed_vals, columns=sensors)], axis=1)
        removed_wide_df.to_csv(args.removed, index=False)
        print(f"Wrote removed values (wide) to {args.removed}")

    # Optional: write removed values in long (tidy) format
    if args.removed_long:
        rows_idx, cols_idx = np.where(mask)
        if rows_idx.size > 0:
            long_df = pd.DataFrame({
                ts_col: df[ts_col].iloc[rows_idx].values,
                "sensor": sensors[cols_idx],
                "value": vals[rows_idx, cols_idx],
            })
        else:
            long_df = pd.DataFrame(columns=[ts_col, "sensor", "value"])
        long_df.to_csv(args.removed_long, index=False)
        print(f"Wrote removed values (long) to {args.removed_long}")

    if pd.isna(df[ts_col]).all():
        print("[warning] Timestamp column could not be parsed; left untouched as text.")


if __name__ == "__main__":
    main()
