"""
Relocatable data/mask checks, optional real checkpoint replay, and
source integrity.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

from helper import load_dataset, make_mask, digest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true")
    a = parser.parse_args()
    cfg = json.loads((ROOT / "config.yaml").read_text())
    expected = json.loads(
        (ROOT / "reference/preparation_checks.json").read_text()
    )
    checks = {"datasets": 0, "masks": 0, "checkpoint_predictions": 0}
    if a.checkpoints:
        # Import torch before TensorFlow on the recorded Windows
        # runtime.
        from gainmodel import Neural
        from grinmodel import ReferenceGRIN
        from gcnnmodel import Sage
    with tempfile.TemporaryDirectory(prefix="traffic-package-") as tmp:
        for dc in cfg["datasets"]:
            d = load_dataset(dc, Path(tmp) / dc["name"])
            e = expected[dc["name"]]
            assert (
                d["names"] == e["sensors"]
                and [d["a"], d["b"], len(d["x"])] == e["boundaries"]
            )
            assert (
                digest(d["x"]) == e["values_sha256"]
                and digest(d["A"]) == e["adj_sha256"]
            )
            te = d["x"][d["b"] :]
            va = d["x"][d["a"] : d["b"]]
            tr = d["x"][: d["a"]]
            for seed in cfg["seeds"]:
                vm = make_mask(va, cfg["validation_mask"], seed + 30000)
                assert digest(vm) == e["validation_masks"][str(seed)]
                for i, sc in enumerate(cfg["scenarios"]):
                    mask = make_mask(
                        te, sc, seed + 100000 + i * 1000, d["profile"], tr
                    )
                    assert (
                        digest(mask)
                        == e["test_masks"][f"{sc['name']}_s{seed}"]
                    )
                    checks["masks"] += 1
                if a.checkpoints:
                    mask = make_mask(
                        te,
                        cfg["scenarios"][0],
                        seed + 100000,
                        d["profile"],
                        tr,
                    )
                    x = te.copy()
                    x[mask] = np.nan
                    for method in ["GAIN", "sage_voronoi", "GRIN"]:
                        cp = (
                            ROOT
                            / "reference/checkpoints"
                            / dc["name"]
                            / f"{method}_s{seed}"
                        )
                        if method == "GAIN":
                            model = Neural(
                                d, method, cfg["neural"][method], seed
                            )
                        elif method == "sage_voronoi":
                            model = Sage(d, cfg["sage"], seed)
                        else:
                            model = ReferenceGRIN(
                                tr, d["A"], cfg["neural"][method], seed
                            )
                        model.restore(cp)
                        pred, _ = model.predict(x, d["index"][d["b"] :])
                        z = np.load(
                            ROOT
                            / "reference/prediction_checks"
                            / dc["name"]
                            / f"{method}_s{seed}.npz"
                        )
                        np.testing.assert_array_equal(te[mask], z["truth"])
                        np.testing.assert_allclose(
                            pred[mask], z["prediction"], rtol=1e-5, atol=5e-4
                        )
                        np.testing.assert_array_equal(pred[~mask], x[~mask])
                        checks["checkpoint_predictions"] += 1
                        print(
                            dc["name"],
                            method,
                            seed,
                            "checkpoint replay PASS",
                            flush=True,
                        )
                        del model
            checks["datasets"] += 1
    print(json.dumps({"status": "PASS", **checks}, indent=2))


if __name__ == "__main__":
    main()
