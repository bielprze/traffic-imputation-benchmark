"""
Independent run-level statistics and optional prediction/mask audit.
"""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def verify(folder, expected):
    raw = pd.read_csv(folder / "results_raw.csv")
    summary = pd.read_csv(folder / "results_summary.csv")
    keys = ["dataset", "method", "scenario", "metric"]
    assert len(summary) == expected and len(raw) == expected * 5
    assert not raw.duplicated(keys + ["seed"]).any()
    computed = []
    for key, g in raw.groupby(keys):
        assert set(g.seed) == set(range(5))
        values = g.value.to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1)
        delta = stats.t.ppf(0.975, 4) * sd / np.sqrt(5)
        computed.append(
            dict(
                zip(keys, key),
                mean=mean,
                std=sd,
                ci_lower=mean - delta,
                ci_upper=mean + delta,
            )
        )
    computed = pd.DataFrame(computed).set_index(keys).sort_index()
    actual = summary.set_index(keys).sort_index()
    for col in ["mean", "std", "ci_lower", "ci_upper"]:
        np.testing.assert_allclose(
            computed[col], actual[col], rtol=1e-10, atol=1e-10
        )
    assert actual.n_runs.eq(5).all() and actual.complete.all()
    pairs = pd.read_csv(folder / "paired_comparisons.csv")
    calc = []
    for key, g in raw.groupby(["dataset", "scenario", "metric"]):
        linear = g[g.method.eq("linear")].set_index("seed").sort_index()
        for method, h in g.groupby("method"):
            if method == "linear":
                continue
            h = h.set_index("seed").sort_index()
            assert h.mask_sha256.equals(linear.mask_sha256)
            d = h.value.to_numpy() - linear.value.to_numpy()
            sd = d.std(ddof=1)
            delta = stats.t.ppf(0.975, 4) * sd / np.sqrt(5)
            calc.append(
                dict(
                    zip(["dataset", "scenario", "metric"], key),
                    method=method,
                    mean_difference=d.mean(),
                    ci_lower=d.mean() - delta,
                    ci_upper=d.mean() + delta,
                    p_raw=(
                        stats.ttest_1samp(d, 0).pvalue
                        if sd
                        else float(d.mean() == 0)
                    ),
                )
            )
    q = pd.DataFrame(calc)
    q["p_holm"] = np.nan
    for _, g in q.groupby(["dataset", "scenario", "metric"]):
        order = g.sort_values("p_raw").index
        q.loc[order, "p_holm"] = np.minimum(
            1,
            np.maximum.accumulate(
                q.loc[order, "p_raw"].to_numpy() * np.arange(len(order), 0, -1)
            ),
        )
    pk = ["dataset", "scenario", "metric", "method"]
    q = q.set_index(pk).sort_index()
    pairs = pairs.set_index(pk).sort_index()
    for c in ["mean_difference", "ci_lower", "ci_upper", "p_raw", "p_holm"]:
        np.testing.assert_allclose(q[c], pairs[c], rtol=1e-8, atol=1e-12)
    return {
        "summary_cells": len(summary),
        "raw_metric_rows": len(raw),
        "paired_contrasts": len(pairs),
    }


def artifacts(work):
    count = 0
    for engine in ["benchmark", "grin"]:
        for namespace in [work / engine, work / engine / "availability"]:
            if not (namespace / "runs").exists():
                raise ValueError("Missing evaluations: " + str(namespace))
            for p in (namespace / "runs").rglob("result.json"):
                r = json.loads(p.read_text())
                z = np.load(p.parent / "predictions.npz")
                m = np.load(
                    namespace
                    / "data"
                    / r["dataset"]
                    / "masks"
                    / f"{r['scenario']}_s{r['seed']}.npz"
                )["test_mask"]
                assert (
                    hashlib.sha256(
                        np.ascontiguousarray(m).tobytes()
                    ).hexdigest()
                    == r["mask_sha256"]
                )
                rows, cols = np.where(m)
                np.testing.assert_array_equal(rows, z["rows"])
                np.testing.assert_array_equal(cols, z["cols"])
                err = np.abs(z["truth"] - z["prediction"])
                nonzero = z["truth"] != 0
                vals = {
                    "MAE": err.mean(),
                    "RMSE": np.sqrt((err**2).mean()),
                    "MRE": (err[nonzero] / np.abs(z["truth"][nonzero])).mean(),
                }
                for k, v in vals.items():
                    np.testing.assert_allclose(
                        v, r["metrics"][k], rtol=1e-10, atol=1e-10
                    )
                assert r["status"] == "success" and r["split"] == "test"
                count += 1
    assert count == 1800, count
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    p.add_argument("--work", type=Path)
    a = p.parse_args()
    result = {
        "primary": verify(a.results, 936),
        "availability": verify(a.results / "availability", 144),
    }
    if a.work:
        result["prediction_sets"] = artifacts(a.work)
    print(json.dumps({"status": "PASS", **result}, indent=2))
