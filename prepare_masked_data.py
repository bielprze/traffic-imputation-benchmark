"""
Prepare training-only sensor selection, splits, profiles and shared
masks.
"""

import json
import os
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def split_configs(cfg):
    a = json.loads(json.dumps(cfg))
    a["methods"] = [m for m in cfg["methods"] if m != "GRIN"]
    a["neural"].pop("GRIN", None)
    b = {
        k: cfg[k]
        for k in [
            "protocol_version",
            "seeds",
            "datasets",
            "scenarios",
            "validation_mask",
            "notes",
        ]
    }
    b.update(methods=["GRIN"], neural={"GRIN": cfg["neural"]["GRIN"]})
    return a, b


def prepare(work, cfg):
    import numpy as np
    from helper import load_dataset, make_mask

    a, b = split_configs(cfg)
    for name, c in [("benchmark", a), ("grin", b)]:
        dest = work / name
        dest.mkdir(exist_ok=True)
        write_json(dest / "configs/main.json", c)
        write_json(dest / "configs/resolved_main.json", c)
        (dest / "logs").mkdir(exist_ok=True)
    for dc in cfg["datasets"]:
        dest = work / "benchmark/data" / dc["name"]
        d = load_dataset(dc, dest)
        tr = d["x"][: d["a"]]
        va = d["x"][d["a"] : d["b"]]
        te = d["x"][d["b"] :]
        md = dest / "masks"
        md.mkdir(exist_ok=True)
        for seed in cfg["seeds"]:
            vm = make_mask(va, cfg["validation_mask"], seed + 30000)
            for i, sc in enumerate(cfg["scenarios"]):
                mask = make_mask(
                    te, sc, seed + 100000 + i * 1000, d["profile"], tr
                )
                np.savez_compressed(
                    md / f"{sc['name']}_s{seed}.npz",
                    test_mask=mask,
                    validation_mask=vm,
                )
    print(
        "Prepared train-only sensor selection, splits, profiles and"
        " deterministic masks.",
        flush=True,
    )


if __name__ == "__main__":
    import sys
    from run_experiments import main

    main(["--stage", "prepare", *sys.argv[1:]])
