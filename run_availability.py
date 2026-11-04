"""
Frozen-model stress test: lose25%/50% sensor streams during test.
Measures information availability, not physical density or retrained
smaller networks.
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keep imports after required runtime/environment initialization.
import json  # noqa: E402
import time  # noqa: E402
import hashlib  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from helper import load_dataset, metrics, digest  # noqa: E402
from baseline_imputation import Classical  # noqa: E402
from data_imputation import execution_fingerprint  # noqa: E402

OUTPUT_DIR = (
    Path(
        os.environ.get(
            "TRAFFIC_RUN_ROOT",
            Path(__file__).resolve().parent / "runs/reproduction",
        )
    )
    / "benchmark"
)
OUT = OUTPUT_DIR / "availability"


def ready(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main():
    base = json.loads((OUTPUT_DIR / "configs/main.json").read_text())
    cov = ready(OUTPUT_DIR / "coverage.json")
    expected = (
        len(base["datasets"])
        * len(base["methods"])
        * len(base["seeds"])
        * len(base["scenarios"])
    )
    if not cov or cov.get("successful_runs") != expected:
        raise RuntimeError("Complete primary evaluation and analysis first")
    assert execution_fingerprint(base) == json.loads(
        (OUTPUT_DIR / "execution_manifest.json").read_text()
    )
    cfg = {
        **base,
        "scenarios": [
            {
                "name": "sensor_streams_missing25",
                "kind": "whole_sensors",
                "rate": 0.25,
            },
            {
                "name": "sensor_streams_missing50",
                "kind": "whole_sensors",
                "rate": 0.5,
            },
        ],
        "notes": (
            "Secondary sensor-stream availability stress. Frozen"
            " train/validation-selected full-network models; no retraining or"
            " physical-density claim. Subsets fixed by seed independently of"
            " test outcomes."
        ),
    }
    manifest = execution_fingerprint(cfg)
    manifest["availability_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest["frozen_fit_sha256"] = {}
    for dc in cfg["datasets"]:
        for m in cfg["methods"]:
            for seed in cfg["seeds"]:
                mp = OUTPUT_DIR / "models" / dc["name"] / f"{m}_s{seed}"
                for name in ["fit.json"] + (
                    ["weights.pt"]
                    if m in ["GAIN"]
                    else ["sage.weights.h5"] if m == "sage_voronoi" else []
                ):
                    p = mp / name
                    manifest["frozen_fit_sha256"][
                        p.relative_to(OUTPUT_DIR).as_posix()
                    ] = hashlib.sha256(p.read_bytes()).hexdigest()
    OUT.mkdir(exist_ok=True)
    mp = OUT / "execution_manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "configs/resolved_main.json").write_text(json.dumps(cfg, indent=2))
    cs = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    # Import PyTorch first to avoid the host runtime's OpenMP
    # import-order conflict.
    from gainmodel import Neural
    from gcnnmodel import Sage

    for dc in cfg["datasets"]:
        d = load_dataset(dc, OUT / "data" / dc["name"])
        te = d["x"][d["b"] :]
        ti = d["index"][d["b"] :]
        md = OUT / "data" / dc["name"] / "masks"
        md.mkdir(exist_ok=True)
        for seed in cfg["seeds"]:
            masks = {}
            for i, sc in enumerate(cfg["scenarios"]):
                selected = np.random.default_rng(
                    seed + 960000 + i * 1000
                ).choice(
                    len(d["names"]),
                    round(len(d["names"]) * sc["rate"]),
                    replace=False,
                )
                mask = np.zeros(te.shape, bool)
                mask[:, selected] = True
                mask &= ~np.isnan(te)
                masks[sc["name"]] = mask
                vm = np.load(
                    OUTPUT_DIR
                    / "data"
                    / dc["name"]
                    / "masks"
                    / f"block_1_s{seed}.npz"
                )["validation_mask"]
                np.savez_compressed(
                    md / f"{sc['name']}_s{seed}.npz",
                    test_mask=mask,
                    validation_mask=vm,
                    unavailable_sensor_columns=selected,
                )
            for method in cfg["methods"]:
                source = (
                    OUTPUT_DIR / "models" / dc["name"] / f"{method}_s{seed}"
                )
                fit = json.loads((source / "fit.json").read_text())
                if method in ["GAIN"]:
                    model = Neural(d, method, cfg["neural"][method], seed)
                    model.restore(source)
                    params = cfg["neural"][method]
                elif method == "sage_voronoi":
                    model = Sage(d, cfg["sage"], seed)
                    model.restore(source)
                    params = cfg["sage"]
                else:
                    params = fit["chosen"]
                    model = Classical(d, method, params)
                for sc in cfg["scenarios"]:
                    dest = (
                        OUT
                        / "runs"
                        / dc["name"]
                        / sc["name"]
                        / f"{method}_s{seed}"
                    )
                    dest.mkdir(parents=True, exist_ok=True)
                    if (dest / "result.json").exists():
                        continue
                    mask = masks[sc["name"]]
                    x = te.copy()
                    x[mask] = np.nan
                    t = time.perf_counter()
                    pred, info = model.predict(x, ti)
                    elapsed = time.perf_counter() - t
                    assert np.array_equal(pred[~np.isnan(x)], x[~np.isnan(x)])
                    rr, cc = np.where(mask)
                    np.savez_compressed(
                        dest / "predictions.npz",
                        rows=rr,
                        cols=cc,
                        truth=te[mask],
                        prediction=pred[mask],
                    )
                    row = {
                        "dataset": dc["name"],
                        "method": method,
                        "seed": seed,
                        "scenario": sc["name"],
                        "masking": sc["kind"],
                        "missing_rate": sc["rate"],
                        "achieved_missing_rate": float(
                            mask.sum() / np.isfinite(te).sum()
                        ),
                        "gap_length": None,
                        "metrics": metrics(te[mask], pred[mask]),
                        "seconds": elapsed,
                        "fit": fit,
                        "params": params,
                        "solver": info,
                        "mask_sha256": digest(mask),
                        "config_sha256": cs,
                        "split": "test",
                        "status": "success",
                        "n_points": int(mask.sum()),
                        "n_mre": int((te[mask] != 0).sum()),
                    }
                    (dest / "result.json").write_text(
                        json.dumps(row, indent=2)
                    )
                    print(dc["name"], seed, method, sc["name"], flush=True)
    print(
        "Availability complete",
        len(cfg["datasets"]) * len(cfg["methods"]) * len(cfg["seeds"]) * 2,
        flush=True,
    )
    (OUT / "completion.json").write_text(
        json.dumps(
            {
                "successful_runs": (
                    len(cfg["datasets"])
                    * len(cfg["methods"])
                    * len(cfg["seeds"])
                    * 2
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=2):
            main()
    except Exception:
        (OUT / "logs/failure.txt").write_text(traceback.format_exc())
        raise
