"""
GRIN training and evaluation using shared splits and paired benchmark
masks.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from helper import load_dataset, make_mask, metrics, digest
from grinmodel import ReferenceGRIN, atomic_json

HERE = Path(__file__).resolve().parent
CODE = HERE
SHARED = (
    Path(os.environ.get("TRAFFIC_RUN_ROOT", HERE / "runs/reproduction"))
    / "benchmark"
)
ROOT = HERE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def fingerprint(cfg):
    sources = [
        HERE / "grinmodel.py",
        HERE / "grin_experiments.py",
        CODE / "helper.py",
        *sorted((HERE / "vendor").rglob("*.py")),
    ]
    inputs = [ROOT / d[k] for d in cfg["datasets"] for k in ["path", "edges"]]
    return {
        "config": cfg,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "pandas", "scipy", "einops"]
        },
    }


def guard(out, cfg):
    expected = fingerprint(cfg)
    p = out / "execution_manifest.json"
    if p.exists():
        if json.loads(p.read_text()) != expected:
            raise RuntimeError(
                "Source/input/config/runtime changed: use a new output root"
            )
    else:
        if list((out / "models").rglob("*.pt")) or list(
            (out / "runs").rglob("result.json")
        ):
            raise RuntimeError("Unmanifested results refused")
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(p, expected)
    (out / "configs").mkdir(exist_ok=True)
    atomic_json(out / "configs/resolved_main.json", cfg)


def progress(out, stage, details, major=False):
    state = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "stage": stage,
        **details,
    }
    atomic_json(out / "status.json", state)
    if major:
        text = (
            f'\n- {state["updated"]} GRIN — {stage}:'
            f" {json.dumps(details, ensure_ascii=False)}\n"
        )
        with (out / "progress.md").open("a", encoding="utf-8") as f:
            f.write(text)
        with (out.parent / "progress.md").open("a", encoding="utf-8") as f:
            f.write(text)


def dataset(dc, out):
    d = load_dataset(dc, out / "data" / dc["name"])
    # Verify shared split/panel/graph identity. No model decisions use
    # test.
    meta = json.loads(
        (SHARED / "data" / dc["name"] / "dataset.json").read_text()
    )
    if d["meta"] != meta:
        raise AssertionError("Dataset/panel/split drift")
    with np.load(SHARED / "data" / dc["name"] / "data.npz") as saved:
        np.testing.assert_array_equal(d["x"], saved["values"])
        np.testing.assert_array_equal(d["A"], saved["adj"])
    return d


def train_all(out, cfg):
    for dc in cfg["datasets"]:
        d = dataset(dc, out)
        train, val = d["x"][: d["a"]], d["x"][d["a"] : d["b"]]
        for seed in cfg["seeds"]:
            path = out / "models" / dc["name"] / f"GRIN_s{seed}"
            model = ReferenceGRIN(train, d["A"], cfg["neural"]["GRIN"], seed)
            vm = make_mask(val, cfg["validation_mask"], seed + 30000)
            with np.load(
                SHARED / "data" / dc["name"] / "masks" / f"block_1_s{seed}.npz"
            ) as saved:
                np.testing.assert_array_equal(vm, saved["validation_mask"])
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path / "validation_mask.npz", validation_mask=vm
            )
            if (path / "fit.json").exists():
                model.restore(path)
                continue
            progress(
                out,
                "training",
                {"dataset": dc["name"], "seed": seed},
                major=True,
            )

            def tick(row, best, stale):
                progress(
                    out,
                    "training",
                    {
                        "dataset": dc["name"],
                        "seed": seed,
                        **row,
                        "best_epoch": best,
                        "stale": stale,
                    },
                    major=row["epoch"] % 10 == 0,
                )

            fit = model.fit(train, val, vm, path, on_epoch=tick)
            progress(
                out,
                "fit_completed",
                {
                    "dataset": dc["name"],
                    "seed": seed,
                    "best_epoch": fit["best_epoch"],
                    "epochs_run": fit["epochs_run"],
                    "budget_limited": fit["budget_limited"],
                },
                major=True,
            )
    progress(
        out,
        "all_training_completed",
        {"fits": 10, "test_evaluation_started": False},
        major=True,
    )


def availability_config(cfg):
    return {
        **cfg,
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
    }


def evaluate(out, cfg):
    # Never begin test evaluation until every prespecified training run
    # is complete.
    for dc in cfg["datasets"]:
        for seed in cfg["seeds"]:
            if not (
                out / "models" / dc["name"] / f"GRIN_s{seed}" / "fit.json"
            ).exists():
                raise RuntimeError(
                    "All ten train/validation-selected fits are required first"
                )
    avcfg = availability_config(cfg)
    (out / "availability/configs").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "availability/configs/resolved_main.json", avcfg)
    for dc in cfg["datasets"]:
        d = dataset(dc, out)
        # Reuse this prepared data in the availability analysis without
        # a second copy.
        avdata = out / "availability/data" / dc["name"]
        avdata.mkdir(parents=True, exist_ok=True)
        import shutil

        for name in ["dataset.json", "data.npz", "profile.json"]:
            shutil.copyfile(out / "data" / dc["name"] / name, avdata / name)
        te = d["x"][d["b"] :]
        for seed in cfg["seeds"]:
            model = ReferenceGRIN(
                d["x"][: d["a"]], d["A"], cfg["neural"]["GRIN"], seed
            )
            fitpath = out / "models" / dc["name"] / f"GRIN_s{seed}"
            fit = model.restore(fitpath)
            for namespace, current, source in [
                (out, cfg, SHARED),
                (out / "availability", avcfg, SHARED / "availability"),
            ]:
                for sc in current["scenarios"]:
                    dest = (
                        namespace
                        / "runs"
                        / dc["name"]
                        / sc["name"]
                        / f"GRIN_s{seed}"
                    )
                    mp = (
                        source
                        / "data"
                        / dc["name"]
                        / "masks"
                        / f'{sc["name"]}_s{seed}.npz'
                    )
                    with np.load(mp) as z:
                        masks = {k: z[k] for k in z.files}
                    mask = masks["test_mask"]
                    if np.any(mask & ~np.isfinite(te)):
                        raise AssertionError("Mask targets unknown truth")
                    target = namespace / "data" / dc["name"] / "masks"
                    target.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target / mp.name, **masks)
                    if (dest / "result.json").exists():
                        r = json.loads((dest / "result.json").read_text())
                        assert r["config_sha256"] == config_sha(current)
                        assert r["checkpoint_sha256"] == sha(
                            fitpath / "weights.pt"
                        )
                        continue
                    progress(
                        out,
                        "test_evaluation",
                        {
                            "dataset": dc["name"],
                            "seed": seed,
                            "scenario": sc["name"],
                        },
                    )
                    masked = te.copy()
                    masked[mask] = np.nan
                    started = time.perf_counter()
                    pred, info = model.predict(masked)
                    seconds = time.perf_counter() - started
                    np.testing.assert_array_equal(
                        pred[np.isfinite(masked)], masked[np.isfinite(masked)]
                    )
                    vals = metrics(te[mask], pred[mask])
                    rows, cols = np.where(mask)
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dest / "predictions.npz",
                        rows=rows,
                        cols=cols,
                        truth=te[mask],
                        prediction=pred[mask],
                    )
                    r = dict(
                        dataset=dc["name"],
                        method="GRIN",
                        seed=seed,
                        scenario=sc["name"],
                        masking=sc["kind"],
                        missing_rate=sc.get("rate"),
                        gap_length=sc.get("length"),
                        achieved_missing_rate=float(
                            mask.sum() / np.isfinite(te).sum()
                        ),
                        metrics=vals,
                        n_points=int(mask.sum()),
                        n_mre=int((te[mask] != 0).sum()),
                        seconds=seconds,
                        fit=fit,
                        params=cfg["neural"]["GRIN"],
                        solver=info,
                        mask_sha256=digest(mask),
                        config_sha256=config_sha(current),
                        checkpoint_sha256=sha(fitpath / "weights.pt"),
                        split="test",
                        status="success",
                    )
                    atomic_json(dest / "result.json", r)
                    print(dc["name"], seed, sc["name"], vals, flush=True)
            progress(
                out,
                "evaluation_seed_completed",
                {"dataset": dc["name"], "seed": seed},
                major=True,
            )
    progress(
        out,
        "all_evaluation_completed",
        {"primary": 130, "availability": 20},
        major=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "analyze"], default="train"
    )
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    cfg = json.loads(Path(args.config).read_text())
    out.mkdir(parents=True, exist_ok=True)
    # OS lock releases even if process dies; stale lock files are
    # harmless.
    lock = (out / "execution.lock").open("a+b")
    lock.seek(0)
    if os.name == "nt":
        import msvcrt

        if not (out / "execution.lock").stat().st_size:
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        guard(out, cfg)
        if args.stage in ["train"]:
            train_all(out, cfg)
        if args.stage in ["evaluate"]:
            evaluate(out, cfg)
        if args.stage in ["analyze"]:
            from combine_results import analyze_all

            analyze_all(out)
    except Exception:
        error = traceback.format_exc()
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / f"failure_{time.time_ns()}.txt").write_text(
            error, encoding="utf-8"
        )
        progress(
            out,
            "failed",
            {
                "error": error,
                "continuation": (
                    "Fix recorded cause; resume same command only if frozen"
                    " source is unchanged."
                ),
            },
            major=True,
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
