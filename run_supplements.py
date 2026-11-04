"""
Repeat graph/scaling ablations on prepared data; fixed choices, no test
selection.
"""

import os
import json
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from helper import load_dataset, metrics, digest
from baseline_imputation import Classical
from data_imputation import execution_fingerprint

OUTPUT_DIR = (
    Path(
        os.environ.get(
            "TRAFFIC_RUN_ROOT",
            Path(__file__).resolve().parent / "runs/reproduction",
        )
    )
    / "benchmark"
)


def main():
    cfg = json.loads((OUTPUT_DIR / "configs/main.json").read_text())
    fits = [
        OUTPUT_DIR / "models" / d["name"] / f"{m}_s{s}" / "fit.json"
        for d in cfg["datasets"]
        for m in ["graph_temporal", "knn"]
        for s in cfg["seeds"]
    ]
    loaded = {
        p.relative_to(OUTPUT_DIR).as_posix(): json.loads(p.read_text())
        for p in fits
    }  # Fail clearly if prerequisite fits are missing.
    assert execution_fingerprint(cfg) == json.loads(
        (OUTPUT_DIR / "execution_manifest.json").read_text()
    )
    manifest = {
        "source_execution": execution_fingerprint(cfg),
        "script_sha256": (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "fits": {
            str(p.relative_to(OUTPUT_DIR)): (
                hashlib.sha256(p.read_bytes()).hexdigest()
            )
            for p in fits
        },
    }
    path = OUTPUT_DIR / "supplements_manifest.json"
    if path.exists():
        assert (
            json.loads(path.read_text()) == manifest
        ), "Changed supplement inputs"
    else:
        path.write_text(json.dumps(manifest, indent=2))
    rows = []
    variants = [
        ("road_normalized", {}),
        ("road_raw", {"normalize": False}),
        ("no_edges", {"graph": "none"}),
        ("train_correlation", {"graph": "correlation"}),
    ]
    for dc in cfg["datasets"]:
        d = load_dataset(dc, OUTPUT_DIR / "audit/supplement_data" / dc["name"])
        te = d["x"][d["b"] :]
        ti = d["index"][d["b"] :]
        for seed in cfg["seeds"]:
            gp = loaded[
                f"models/{dc['name']}/graph_temporal_s{seed}/fit.json"
            ]["chosen"]
            kp = loaded[f"models/{dc['name']}/knn_s{seed}/fit.json"]["chosen"]
            for sc in ["block_16", "block_144", "mcar_30"]:
                mask = np.load(
                    OUTPUT_DIR
                    / "data"
                    / dc["name"]
                    / "masks"
                    / f"{sc}_s{seed}.npz"
                )["test_mask"]
                x = te.copy()
                x[mask] = np.nan
                tasks = [
                    ("graph_temporal", v, {**gp, **p}) for v, p in variants
                ]
                if sc != "block_16":
                    tasks += [
                        ("knn", "standardized_fixed_k", kp),
                        ("knn_unscaled", "unscaled_fixed_k", kp),
                    ]
                for method, variant, params in tasks:
                    dest = (
                        OUTPUT_DIR
                        / "supplements"
                        / dc["name"]
                        / sc
                        / f"{variant}_s{seed}"
                    )
                    dest.mkdir(parents=True, exist_ok=True)
                    rp = dest / "result.json"
                    if rp.exists():
                        r = json.loads(rp.read_text())
                    else:
                        t = time.perf_counter()
                        model = Classical(d, method, params)
                        pred, info = model.predict(x, ti)
                        seconds = time.perf_counter() - t
                        rr, cc = np.where(mask)
                        np.savez_compressed(
                            dest / "predictions.npz",
                            rows=rr,
                            cols=cc,
                            truth=te[mask],
                            prediction=pred[mask],
                        )
                        r = {
                            "dataset": dc["name"],
                            "scenario": sc,
                            "method": method,
                            "variant": variant,
                            "seed": seed,
                            "params": params,
                            "metrics": metrics(te[mask], pred[mask]),
                            "seconds_fit_and_predict": seconds,
                            "solver": info,
                            "mask_sha256": digest(mask),
                        }
                        rp.write_text(json.dumps(r, indent=2))
                    for metric, value in r["metrics"].items():
                        rows.append(
                            {
                                k: r[k]
                                for k in [
                                    "dataset",
                                    "scenario",
                                    "method",
                                    "variant",
                                    "seed",
                                ]
                            }
                            | {
                                "metric": metric,
                                "value": value,
                                "seconds_fit_and_predict": r[
                                    "seconds_fit_and_predict"
                                ],
                            }
                        )
                    pd.DataFrame(rows).to_csv(
                        OUTPUT_DIR / "supplements_raw.csv", index=False
                    )
                    print(dc["name"], seed, sc, variant, flush=True)
    from analyze_results import summarize

    summarize(
        pd.DataFrame(rows),
        ["dataset", "scenario", "method", "variant", "metric"],
        cfg["seeds"],
    ).to_csv(OUTPUT_DIR / "supplements_summary.csv", index=False)
    (OUTPUT_DIR / "supplements_complete.json").write_text(
        json.dumps(
            {
                "prediction_sets": len(rows) // 3,
                "graph_prediction_sets": 120,
                "knn_prediction_sets": 40,
                "knn_controls_recomputed_not_new_replicates": 20,
            },
            indent=2,
        )
    )
    print("Supplements complete", len(rows) // 3, flush=True)


if __name__ == "__main__":
    try:
        with threadpool_limits(limits=2):
            main()
    except Exception:
        (OUTPUT_DIR / "logs/supplements_failure.txt").write_text(
            traceback.format_exc()
        )
        raise
