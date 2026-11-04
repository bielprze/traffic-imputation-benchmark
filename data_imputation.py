"""
Validation selection, imputation runs and all-method evaluation
coordination.
"""

import argparse
import hashlib
import json
import time
import traceback
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keep imports after required runtime/environment initialization.
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from helper import load_dataset, make_mask, metrics, digest  # noqa: E402
from baseline_imputation import Classical  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
BASE = OUTPUT_DIR


def execution_fingerprint(cfg):
    root = BASE
    sources = {}
    for p in [
        BASE / "data_imputation.py",
        BASE / "helper.py",
        BASE / "baseline_imputation.py",
        BASE / "gainmodel.py",
        BASE / "gcnnmodel.py",
        *sorted((BASE / "vendor").rglob("*.py")),
    ]:
        sources[os.path.relpath(p, BASE)] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    inputs = {}
    for dc in cfg["datasets"]:
        for key in ["path", "edges"]:
            p = root / dc[key]
            inputs[dc[key]] = hashlib.sha256(p.read_bytes()).hexdigest()
    p = root / "sensors_location.geojson"
    inputs["sensors_location.geojson"] = hashlib.sha256(
        p.read_bytes()
    ).hexdigest()
    import importlib.metadata as md

    versions = {
        m: md.version(m)
        for m in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "torch",
            "tensorflow",
            "einops",
            "pyproj",
        ]
    }
    return {
        "config": cfg,
        "algorithm_sources_sha256": sources,
        "input_sha256": inputs,
        "runtime_versions": versions,
    }


def guard_execution(out, cfg, adopt=False):
    expected = execution_fingerprint(cfg)
    path = out / "execution_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(
                "Configuration, input or algorithm changed. Use a NEW"
                " --output-root; refusing before any artifact write."
            )
    else:
        old = (
            list((out / "runs").rglob("result.json"))
            if (out / "runs").exists()
            else []
        )
        if old and not adopt:
            raise RuntimeError(
                "Existing unmanifested artifacts; audit and explicitly"
                " --adopt-existing, or use a new output root."
            )
        cfgsha = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()
        for p in old:
            if json.loads(p.read_text()).get("config_sha256") != cfgsha:
                raise RuntimeError("Cannot adopt mismatching run " + str(p))
        for p in (
            (out / "models").rglob("fit.json")
            if (out / "models").exists()
            else []
        ):
            fit = json.loads(p.read_text())
            expected_cfg = (
                cfg["sage"]
                if fit.get("method") == "sage_voronoi"
                else cfg["neural"].get(fit.get("method"))
            )
            if "config" in fit and fit["config"] != expected_cfg:
                raise RuntimeError("Cannot adopt mismatching fit " + str(p))
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2))
    return expected


def run_engine():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--adopt-existing", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    OUTPUT_DIR = Path(args.output_root).resolve()
    cfg = json.loads(Path(args.config).read_text())
    cfgsha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()
    provenance = guard_execution(OUTPUT_DIR, cfg, args.adopt_existing)
    for sub in ["logs", "configs"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    invocation = f"{time.time_ns()}_{os.getpid()}"
    logpath = OUTPUT_DIR / "logs" / f"invocation_{invocation}.jsonl"

    def event(record):
        with logpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time_ns": time.time_ns(), **record}) + "\n")

    event(
        {
            "event": "start",
            "config_sha256": cfgsha,
            "methods": args.methods or cfg["methods"],
            "seeds": args.seeds or cfg["seeds"],
        }
    )
    runroot = OUTPUT_DIR / "runs"
    runroot.mkdir(exist_ok=True)
    fail = []
    (OUTPUT_DIR / "configs" / "resolved_main.json").write_text(
        json.dumps(cfg, indent=2)
    )
    for dc in cfg["datasets"]:
        if args.datasets and dc["name"] not in args.datasets:
            continue
        dd = OUTPUT_DIR / "data" / dc["name"]
        d = load_dataset(dc, dd)
        tr = d["x"][: d["a"]]
        va = d["x"][d["a"] : d["b"]]
        te = d["x"][d["b"] :]
        ti = d["index"][d["b"] :]
        for seed in (args.seeds if args.seeds is not None else cfg["seeds"]):
            vm = make_mask(va, cfg["validation_mask"], seed + 30000)
            vx = va.copy()
            vx[vm] = np.nan
            masks = {}
            for i, sc in enumerate(
                [] if args.train_only else cfg["scenarios"]
            ):
                try:
                    mask = make_mask(
                        te, sc, seed + 100000 + i * 1000, d["profile"], tr
                    )
                    masks[sc["name"]] = mask
                    mp = dd / "masks"
                    mp.mkdir(exist_ok=True)
                    np.savez_compressed(
                        mp / f"{sc['name']}_s{seed}.npz",
                        test_mask=mask,
                        validation_mask=vm,
                    )
                except Exception as e:
                    fail.append(
                        {
                            "dataset": dc["name"],
                            "seed": seed,
                            "scenario": sc["name"],
                            "stage": "mask",
                            "error": str(e),
                        }
                    )
                    event({"event": "failure", **fail[-1]})
            for method in args.methods or cfg["methods"]:
                fitpath = (
                    OUTPUT_DIR / "models" / dc["name"] / f"{method}_s{seed}"
                )
                fitpath.mkdir(parents=True, exist_ok=True)
                try:
                    start = time.perf_counter()
                    if method == "GAIN":
                        from gainmodel import Neural

                        model = Neural(d, method, cfg["neural"][method], seed)
                        if (fitpath / "fit.json").exists():
                            fitmeta = model.restore(fitpath)
                        else:
                            fitmeta = model.fit(vm, fitpath)
                        params = cfg["neural"][method]
                    elif method == "sage_voronoi":
                        from gcnnmodel import Sage

                        model = Sage(d, cfg["sage"], seed)
                        if (fitpath / "fit.json").exists():
                            fitmeta = model.restore(fitpath)
                        else:
                            fitmeta = model.fit(vm, fitpath)
                        params = cfg["sage"]
                    else:
                        candidates = []
                        for params in cfg["parameter_grid"][method]:
                            model = Classical(d, method, params)
                            p, info = model.predict(
                                vx, d["index"][d["a"] : d["b"]]
                            )
                            candidates.append(
                                {
                                    "params": params,
                                    "validation_MAE": metrics(va[vm], p[vm])[
                                        "MAE"
                                    ],
                                    "solver": info,
                                }
                            )
                        chosen = min(
                            candidates, key=lambda x: x["validation_MAE"]
                        )
                        params = chosen["params"]
                        model = Classical(d, method, params)
                        fitmeta = {
                            "selection": "minimum validation MAE only",
                            "candidates": candidates,
                            "chosen": params,
                            "training_seconds": time.perf_counter() - start,
                            "epochs_run": 0,
                            "early_stopping": "not applicable",
                        }
                        (fitpath / "fit.json").write_text(
                            json.dumps(fitmeta, indent=2)
                        )
                    if args.train_only:
                        continue
                    if method == "sage_voronoi":
                        fp = OUTPUT_DIR / "forecast" / dc["name"] / f"s{seed}"
                        fp.mkdir(parents=True, exist_ok=True)
                        if not (fp / "result.json").exists():
                            t = time.perf_counter()
                            fc = model.forecast(te)
                            fm = np.isfinite(te) & np.isfinite(fc)
                            rr, cc = np.where(fm)
                            np.savez_compressed(
                                fp / "predictions.npz",
                                rows=rr,
                                cols=cc,
                                truth=te[fm],
                                prediction=fc[fm],
                            )
                            (fp / "result.json").write_text(
                                json.dumps(
                                    {
                                        "dataset": dc["name"],
                                        "seed": seed,
                                        "method": method,
                                        "metrics": metrics(te[fm], fc[fm]),
                                        "n_points": int(fm.sum()),
                                        "seconds": time.perf_counter() - t,
                                        "config_sha256": cfgsha,
                                        "split": "test",
                                        "task": (
                                            "plain one-step forecast; past"
                                            " visible test observations only,"
                                            " first window excluded"
                                        ),
                                    },
                                    indent=2,
                                )
                            )
                    for sc in cfg["scenarios"]:
                        if sc["name"] not in masks:
                            continue
                        rp = (
                            runroot
                            / dc["name"]
                            / sc["name"]
                            / f"{method}_s{seed}"
                        )
                        rp.mkdir(parents=True, exist_ok=True)
                        if (rp / "result.json").exists():
                            old = json.loads((rp / "result.json").read_text())
                            if old.get("config_sha256") != cfgsha:
                                raise RuntimeError(
                                    "Config mismatch; use a new output run"
                                    " namespace"
                                )
                            continue
                        mask = masks[sc["name"]]
                        x = te.copy()
                        x[mask] = np.nan
                        t = time.perf_counter()
                        pred, info = model.predict(x, ti)
                        if not np.array_equal(
                            pred[np.isfinite(x)], x[np.isfinite(x)]
                        ):
                            raise AssertionError("Observed values altered")
                        vals = metrics(te[mask], pred[mask])
                        rr, cc = np.where(mask)
                        np.savez_compressed(
                            rp / "predictions.npz",
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
                            "missing_rate": sc.get("rate", None),
                            "achieved_missing_rate": float(
                                mask.sum() / np.isfinite(te).sum()
                            ),
                            "gap_length": sc.get("length", None),
                            "metrics": vals,
                            "n_points": int(mask.sum()),
                            "n_mre": int((te[mask] != 0).sum()),
                            "seconds": time.perf_counter() - t,
                            "fit": fitmeta,
                            "params": params,
                            "solver": info,
                            "mask_sha256": digest(mask),
                            "config_sha256": cfgsha,
                            "split": "test",
                            "status": "success",
                        }
                        (rp / "result.json").write_text(
                            json.dumps(row, indent=2)
                        )
                        print(
                            dc["name"],
                            seed,
                            method,
                            sc["name"],
                            vals,
                            flush=True,
                        )
                except Exception:
                    error = traceback.format_exc()
                    fail.append(
                        {
                            "dataset": dc["name"],
                            "seed": seed,
                            "method": method,
                            "stage": "fit_or_predict",
                            "error": error,
                        }
                    )
                    print(error, flush=True)
                    event({"event": "failure", **fail[-1]})
                (
                    OUTPUT_DIR / "logs" / f"failures_{invocation}.json"
                ).write_text(json.dumps(fail, indent=2))
            with (OUTPUT_DIR / "progress.md").open("a", encoding="utf-8") as f:
                f.write(
                    f'\n- Completed attempted dataset={dc["name"]},'
                    f' seed={seed}, methods={args.methods or cfg["methods"]}.'
                    " Failures: output/logs/failures.json; successful"
                    " artifacts: output/runs.\n"
                )
    print("Finished requested execution; failures:", len(fail))
    event({"event": "finished", "n_failures": len(fail)})
    if fail:
        raise SystemExit(1)


def evaluate_methods(work, methods, flags, subset, execute):
    """
    Evaluate selected imputers, then shared availability masks and GRIN.
    """
    non_grin = [m for m in methods if m != "GRIN"]
    if non_grin:
        execute(
            BASE / "data_imputation.py",
            "--engine",
            "--config",
            work / "benchmark/configs/main.json",
            "--output-root",
            work / "benchmark",
            *flags,
            work=work,
        )
    if not subset:
        execute(
            BASE / "analyze_results.py",
            "--output-root",
            work / "benchmark",
            work=work,
        )
        execute(BASE / "run_availability.py", work=work)
    if "GRIN" in methods:
        execute(
            BASE / "grin_experiments.py",
            "--config",
            work / "grin/configs/main.json",
            "--output-root",
            work / "grin",
            "--stage",
            "evaluate",
            work=work,
        )


if __name__ == "__main__":
    import sys

    if "--engine" in sys.argv:
        sys.argv.remove("--engine")
        with threadpool_limits(limits=2):
            run_engine()
    else:
        from run_experiments import main

        main(["--stage", "evaluate", *sys.argv[1:]])
