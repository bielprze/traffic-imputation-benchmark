"""
Portable entry point for the final twelve-method manuscript protocol.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_masked_data import prepare
from train_GNN import train_models, install_checkpoints
from data_imputation import evaluate_methods

ROOT = Path(__file__).resolve().parent


def read_config(path):
    # JSON is a YAML subset; avoid implicit YAML scalar conversions in
    # frozen configs.
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def execute(script, *args, work):
    env = os.environ.copy()
    env["TRAFFIC_RUN_ROOT"] = str(work)
    env["PYTHONUTF8"] = "1"
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, str(script), *map(str, args)]
    stamp = str(time.time_ns())
    write_json(
        logs / (stamp + ".json"), {"command": command, "cwd": str(ROOT)}
    )
    with (logs / (stamp + ".log")).open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = p.wait()
    if code:
        raise RuntimeError(
            f'{script.name} failed ({code}); see {logs/(stamp+".log")}'
        )


def guard(work, cfg):
    code = [*ROOT.glob("*.py"), *ROOT.glob("vendor/**/*.py")]
    inputs = {
        str(p.relative_to(ROOT)): sha(p)
        for d in cfg["datasets"]
        for p in [ROOT / d["path"], ROOT / d["edges"]]
    }
    inputs["sensors_location.geojson"] = sha(ROOT / "sensors_location.geojson")
    manifest = {
        "config": cfg,
        "code": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(code)},
        "inputs": inputs,
    }
    old = work / "package_execution.json"
    if old.exists() and read_config(old) != manifest:
        raise RuntimeError("Changed code/config/data; use a NEW --output-root")
    write_json(old, manifest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--output-root", default=str(ROOT / "runs/reproduction"))
    ap.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare",
            "train",
            "evaluate",
            "analyze",
            "supplements",
            "qa",
        ],
        default="all",
    )
    ap.add_argument(
        "--from-checkpoints",
        action="store_true",
        help="Use bundled selected weights instead of training new models",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        help="Train/evaluate subset; full analysis requires all 12 methods",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Train/evaluate subset; full analysis requires all five seeds",
    )
    args = ap.parse_args(argv)
    cfg = read_config(args.config)
    work = Path(args.output_root).resolve()
    # Outputs must not overwrite the immutable bundled reference
    # material or source.
    protected = [
        ROOT / "reference",
        ROOT / "results",
        ROOT / "vendor",
        ROOT / "docs",
    ]
    if work == ROOT or any(work == p or p in work.parents for p in protected):
        raise ValueError(
            "Choose an output folder outside source/reference/results"
            " (default: runs/reproduction)."
        )
    if args.methods and not set(args.methods) <= set(cfg["methods"]):
        raise ValueError("Unknown method")
    if args.seeds and not set(args.seeds) <= set(cfg["seeds"]):
        raise ValueError("Unknown seed")
    subset = bool(args.methods or args.seeds)
    if subset and args.stage not in ["prepare", "train", "evaluate"]:
        raise ValueError(
            "Subset execution supports prepare/train/evaluate only; analyze"
            " the complete grid afterwards."
        )
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "execution.lock").open("a+b")
    lock.seek(0)
    if os.name == "nt":
        import msvcrt

        if not (work / "execution.lock").stat().st_size:
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        guard(work, cfg)
        prepare(work, cfg)
        c3 = work / "grin/configs/main.json"
        methods = args.methods or cfg["methods"]
        non_grin = [m for m in methods if m != "GRIN"]
        flags = (["--methods", *non_grin] if non_grin else []) + (
            ["--seeds", *map(str, args.seeds)] if args.seeds else []
        )
        # Keep a full canonical config for manifests; GRIN seed subsets
        # are not used by its native runner.
        if args.seeds and "GRIN" in methods:
            raise ValueError(
                "GRIN seed subsets are not supported; omit --seeds (all five"
                " fixed seeds)."
            )
        if args.stage == "prepare":
            return
        if subset and "GRIN" in methods and args.stage == "evaluate":
            raise ValueError(
                "GRIN evaluation requires the full shared mask grid; omit"
                " --methods/--seeds."
            )
        if args.from_checkpoints:
            install_checkpoints(work, cfg)
        if args.stage in ["all", "train"]:
            train_models(work, methods, flags, execute)
        if args.stage in ["all", "evaluate"]:
            evaluate_methods(work, methods, flags, subset, execute)
        if args.stage in ["all", "supplements"]:
            execute(ROOT / "run_supplements.py", work=work)
        if args.stage in ["all", "analyze"]:
            execute(
                ROOT / "analyze_results.py",
                "--output-root",
                work / "benchmark",
                work=work,
            )
            execute(
                ROOT / "analyze_training.py",
                "--output-root",
                work / "benchmark",
                work=work,
            )
            execute(
                ROOT / "analyze_results.py",
                "--output-root",
                work / "benchmark/availability",
                work=work,
            )
            execute(
                ROOT / "grin_experiments.py",
                "--config",
                c3,
                "--output-root",
                work / "grin",
                "--stage",
                "analyze",
                work=work,
            )
            delivery = work / "results"
            (delivery / "README.md").write_text(
                "Recomputed final protocol results. GRIN: author-reference"
                " architecture; all methods use the shared split-safe"
                " protocol. See the repository README for protocol, run-level"
                " confidence intervals and limitations.\n",
                encoding="utf-8",
            )
            print("Combined results:", delivery)
        if args.stage in ["all", "qa"]:
            execute(
                ROOT / "verify_results.py",
                "--results",
                work / "results",
                "--work",
                work,
                work=work,
            )
    except Exception as e:
        write_json(
            work / f"failure_{time.time_ns()}.json",
            {"error": str(e), "stage": args.stage},
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
