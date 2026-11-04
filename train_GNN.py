"""
Train neural imputers or install verified, validation-selected
checkpoints.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from prepare_masked_data import write_json

ROOT = Path(__file__).resolve().parent


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_models(work, methods, flags, execute):
    non_grin = [m for m in methods if m != "GRIN"]
    if non_grin:
        execute(
            ROOT / "data_imputation.py",
            "--engine",
            "--config",
            work / "benchmark/configs/main.json",
            "--output-root",
            work / "benchmark",
            "--train-only",
            *flags,
            work=work,
        )
    if "GRIN" in methods:
        execute(
            ROOT / "grin_experiments.py",
            "--config",
            work / "grin/configs/main.json",
            "--output-root",
            work / "grin",
            "--stage",
            "train",
            work=work,
        )


def install_checkpoints(work, cfg):
    # Record code/config/input identities before placing any frozen
    # model weights.
    os.environ["TRAFFIC_RUN_ROOT"] = str(work)
    from data_imputation import guard_execution
    from grin_experiments import guard

    guard_execution(
        work / "benchmark", read_config(work / "benchmark/configs/main.json")
    )
    guard(work / "grin", read_config(work / "grin/configs/main.json"))
    seed_checkpoints(work, cfg)


def seed_checkpoints(work, cfg):
    # The manifest verifies every supplied checkpoint, not just its
    # filename.
    manifest = read_config(ROOT / "reference/manifest.json")
    for dc in cfg["datasets"]:
        for m in cfg["methods"]:
            for seed in cfg["seeds"]:
                source = (
                    ROOT
                    / "reference/checkpoints"
                    / dc["name"]
                    / f"{m}_s{seed}"
                )
                fit = read_config(source / "fit.json")
                expected = (
                    cfg["sage"]
                    if m == "sage_voronoi"
                    else cfg["neural"].get(m)
                )
                if expected is not None and fit["config"] != expected:
                    raise ValueError(
                        "Reference fit configuration mismatch: " + str(source)
                    )
                target = (
                    work
                    / ("grin" if m == "GRIN" else "benchmark")
                    / "models"
                    / dc["name"]
                    / source.name
                )
                for p in source.iterdir():
                    if (
                        sha(p)
                        != manifest[
                            p.relative_to(ROOT / "reference").as_posix()
                        ]
                    ):
                        raise ValueError(
                            "Reference checksum mismatch: " + str(p)
                        )
                if target.exists() and any(target.iterdir()):
                    continue  # Never overwrite a newly fitted model.
                shutil.copytree(source, target, dirs_exist_ok=True)
    write_json(
        work / "checkpoint_source.json",
        {
            "source": "bundled selected reference checkpoints",
            "note": (
                "No retraining; evaluate with frozen"
                " training/validation-selected models."
            ),
        },
    )


if __name__ == "__main__":
    import sys
    from run_experiments import main

    main(
        [
            "--stage",
            "train",
            "--methods",
            "sage_voronoi",
            "GAIN",
            "GRIN",
            *sys.argv[1:],
        ]
    )
