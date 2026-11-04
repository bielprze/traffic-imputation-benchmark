"""
Recompute metrics from saved predictions, run-level CIs, paired tests,
tables and plots.
"""

import json
import math
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
# Keep imports after required runtime/environment initialization.
import matplotlib.pyplot as plt  # noqa: E402
from helper import metrics  # noqa: E402
from helper import digest  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def summarize(df, keys, expected_seeds=None):
    r = []
    for key, g in df.groupby(keys, dropna=False):
        if "seed" in g and g.seed.duplicated().any():
            raise ValueError("Duplicate seed within summary cell")
        v = g.value.dropna().to_numpy()
        n = len(v)
        mean = v.mean() if n else np.nan
        sd = v.std(ddof=1) if n > 1 else np.nan
        delta = (
            stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        )
        wanted = set(
            expected_seeds if expected_seeds is not None else range(5)
        )
        complete = n == len(wanted)
        if "seed" in g:
            complete = (
                complete and set(g.loc[g.value.notna(), "seed"]) == wanted
            )
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(
            mean=mean,
            std=sd,
            ci_lower=mean - delta,
            ci_upper=mean + delta,
            n_runs=n,
            expected_runs=len(wanted),
            complete=complete,
        )
        r.append(row)
    return pd.DataFrame(r)


def main():
    cfg = json.loads(
        (OUTPUT_DIR / "configs" / "resolved_main.json").read_text()
    )
    cs = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    expected_keys = {
        (dc["name"], method, sc["name"], seed)
        for dc in cfg["datasets"]
        for method in cfg["methods"]
        for sc in cfg["scenarios"]
        for seed in cfg["seeds"]
    }
    raw = []
    sensor = []
    strata = []
    downstream = []
    rounded = []
    errors = []
    datasets = {}
    mask_cache = {}
    for p in sorted((OUTPUT_DIR / "runs").rglob("result.json")):
        try:
            r = json.loads(p.read_text())
            a = np.load(p.parent / "predictions.npz")
            y = a["truth"]
            pred = a["prediction"]
            rr = a["rows"]
            cc = a["cols"]
            ds = r["dataset"]
            if (
                ds,
                r["method"],
                r["scenario"],
                r["seed"],
            ) not in expected_keys:
                raise ValueError("Unexpected run identity")
            if r["config_sha256"] != cs:
                raise ValueError("Mixed configuration provenance")
            if ds not in datasets:
                meta = json.loads(
                    (OUTPUT_DIR / "data" / ds / "dataset.json").read_text()
                )
                data = np.load(OUTPUT_DIR / "data" / ds / "data.npz")
                tr = data["values"][: meta["splits"]["train"][1]]
                datasets[ds] = (
                    meta,
                    {k: data[k] for k in data.files},
                    np.nanquantile(tr, 0.25, axis=0),
                    np.nanquantile(tr, 0.75, axis=0),
                    data["adj"].sum(0),
                )
                data.close()
            meta, data, low, high, deg = datasets[ds]
            x = data["values"]
            train_end = meta["splits"]["train"][1]
            test_start = meta["splits"]["test"][0]
            if not np.array_equal(y, x[test_start:][rr, cc]):
                raise ValueError("Truth differs from prepared test data")
            mk = (ds, r["scenario"], r["seed"])
            if mk not in mask_cache:
                m = np.load(
                    OUTPUT_DIR
                    / "data"
                    / ds
                    / "masks"
                    / f"{r['scenario']}_s{r['seed']}.npz"
                )["test_mask"]
                mask_cache[mk] = (digest(m), np.where(m))
            mh, coords = mask_cache[mk]
            if (
                mh != r["mask_sha256"]
                or not np.array_equal(rr, coords[0])
                or not np.array_equal(cc, coords[1])
            ):
                raise ValueError("Evaluation mask mismatch")
            ids = {
                k: r.get(k)
                for k in [
                    "dataset",
                    "method",
                    "seed",
                    "scenario",
                    "masking",
                    "missing_rate",
                    "gap_length",
                    "achieved_missing_rate",
                ]
            }
            ids.update(
                n_points=len(y),
                seconds=r["seconds"],
                fit_seconds=r["fit"].get("training_seconds"),
                mask_sha256=r["mask_sha256"],
            )
            run_fields = dict(
                epochs_run=r["fit"].get("epochs_run"),
                best_epoch=r["fit"].get("best_epoch"),
                budget_limited=r["fit"].get("budget_limited", False),
                early_stopping=r["fit"].get("early_stopping"),
                params_json=json.dumps(r["params"], sort_keys=True),
                split="test",
            )
            for metric, value in metrics(y, pred).items():
                if not np.isclose(value, r["metrics"][metric], equal_nan=True):
                    raise ValueError("Stored metric mismatch")
                raw.append(
                    {
                        **ids,
                        **run_fields,
                        "metric": metric,
                        "value": value,
                        "n_mre": int((y != 0).sum()),
                    }
                )
            for metric, value in metrics(
                y, np.rint(np.maximum(pred, 0))
            ).items():
                rounded.append({**ids, "metric": metric, "value": value})
            for c in np.unique(cc):
                hit = cc == c
                for metric, value in metrics(y[hit], pred[hit]).items():
                    sensor.append(
                        {
                            **ids,
                            "sensor": meta["sensors"][c],
                            "metric": metric,
                            "value": value,
                        }
                    )
            for name, hit in [
                ("low_flow", y <= low[cc]),
                ("high_flow", y >= high[cc]),
                ("degree_0_2", deg[cc] <= 2),
                ("degree_3plus", deg[cc] >= 3),
            ]:
                if hit.any():
                    for metric, value in metrics(y[hit], pred[hit]).items():
                        strata.append(
                            {
                                **ids,
                                "stratum": name,
                                "metric": metric,
                                "value": value,
                                "n_stratum_points": int(hit.sum()),
                            }
                        )
            # Downstream retrospective traffic-volume aggregation over
            # known reference cells.
            # Natural missing cells are excluded from BOTH reconstructed
            # and true totals.
            dates = pd.to_datetime(data["timestamps"][test_start:])
            test = x[test_start:]
            reconstructed = test.copy()
            reconstructed[rr, cc] = pred
            truth_total = np.nansum(test, axis=1)
            recon_total = np.nansum(reconstructed, axis=1)
            daily_true = (
                pd.Series(truth_total, index=dates).resample("D").sum()
            )
            daily_pred = (
                pd.Series(recon_total, index=dates).resample("D").sum()
            )
            affected = (
                pd.Series(np.isin(np.arange(len(test)), rr), index=dates)
                .resample("D")
                .max()
                .to_numpy()
            )
            if affected.any():
                for metric, value in metrics(
                    daily_true.to_numpy()[affected],
                    daily_pred.to_numpy()[affected],
                ).items():
                    downstream.append(
                        {
                            **ids,
                            "metric": "daily_known_volume_" + metric,
                            "value": value,
                        }
                    )
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})
    if errors:
        (OUTPUT_DIR / "logs" / "analysis_errors.json").write_text(
            json.dumps(errors, indent=2)
        )
        raise RuntimeError(errors[:3])
    df = pd.DataFrame(raw)
    if df.duplicated(
        ["dataset", "method", "scenario", "seed", "metric"]
    ).any():
        raise ValueError("Duplicate run/metric identity")
    df.to_csv(OUTPUT_DIR / "results_raw.csv", index=False)
    keys = [
        "dataset",
        "method",
        "scenario",
        "masking",
        "missing_rate",
        "gap_length",
        "metric",
    ]
    summary = summarize(df, keys, cfg["seeds"])
    rate = (
        df.groupby(keys, dropna=False)
        .achieved_missing_rate.mean()
        .reset_index(name="mean_achieved_missing_rate")
    )
    summary = summary.merge(rate, on=keys, how="left")
    summary.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    for name, rows, extra in [
        ("per_sensor", sensor, ["sensor"]),
        ("strata", strata, ["stratum"]),
        ("downstream", downstream, []),
        ("rounding", rounded, []),
    ]:
        z = pd.DataFrame(rows)
        z.to_csv(OUTPUT_DIR / f"{name}_raw.csv", index=False)
        summarize(z, keys + extra, cfg["seeds"]).to_csv(
            OUTPUT_DIR / f"{name}_summary.csv", index=False
        )
    # Paired differences to preregistered linear reference; Holm
    # adjustment within dataset/scenario/metric.
    pairs = []
    for (ds, sc, metric), g in df.groupby(["dataset", "scenario", "metric"]):
        base = g[g.method == "linear"][["seed", "value", "mask_sha256"]]
        for method, h in g.groupby("method"):
            if method == "linear":
                continue
            z = h.merge(base, on="seed", suffixes=("", "_linear"))
            n_expected = len(cfg["seeds"])
            if (
                len(z) != n_expected
                or set(z.seed) != set(cfg["seeds"])
                or not (z.mask_sha256 == z.mask_sha256_linear).all()
            ):
                continue
            delta = z.value - z.value_linear
            sd = delta.std(ddof=1)
            ci = stats.t.ppf(0.975, n_expected - 1) * sd / np.sqrt(n_expected)
            pv = (
                stats.ttest_1samp(delta, 0).pvalue
                if sd
                else (1.0 if delta.mean() == 0 else 0.0)
            )
            pairs.append(
                dict(
                    dataset=ds,
                    scenario=sc,
                    metric=metric,
                    method=method,
                    reference="linear",
                    mean_difference=delta.mean(),
                    ci_lower=delta.mean() - ci,
                    ci_upper=delta.mean() + ci,
                    p_raw=pv,
                    n_pairs=n_expected,
                )
            )
    pair = pd.DataFrame(pairs)
    if len(pair):
        pair["p_holm"] = np.nan
        for _, g in pair.groupby(["dataset", "scenario", "metric"]):
            order = g.sort_values("p_raw").index
            pv = pair.loc[order, "p_raw"].values
            adj = np.minimum(
                1, np.maximum.accumulate(pv * np.arange(len(pv), 0, -1))
            )
            pair.loc[order, "p_holm"] = adj
    pair.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
    tables = OUTPUT_DIR / "tables"
    tables.mkdir(exist_ok=True)
    for (ds, metric), g in summary.groupby(["dataset", "metric"]):
        lines = [
            (
                "% Generated from results_summary.csv. Mean [95% run-level t"
                " CI]; only complete 5-run cells."
            ),
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\caption{"
            + ds.replace("_", r"\_")
            + " "
            + metric
            + r": mean [95\% CI], five runs.}",
            r"\begin{tabular}{lll}",
            r"\toprule",
            r"Method & Scenario & Mean [95\% CI] \\",
            r"\midrule",
        ]
        for _, r in g.iterrows():
            if r["complete"]:
                lines.append(
                    f"{r['method'].replace('_',chr(92)+'_')} &"
                    f" {r['scenario'].replace('_',chr(92)+'_')} &"
                    f" {r['mean']:.3f} [{r['ci_lower']:.3f},"
                    f" {r['ci_upper']:.3f}] "
                    + r"\\"
                )
        # Split long tables into chunks so artifacts remain practical
        # for manuscript integration.
        body = lines[9:]
        header = lines[:9]
        for i in range(0, len(body), 35):
            (tables / f"{ds}_{metric}_{i//35+1}.tex").write_text(
                "\n".join(
                    header
                    + body[i : i + 35]
                    + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
                ),
                encoding="utf-8",
            )
    plots = OUTPUT_DIR / "figures"
    plots.mkdir(exist_ok=True)
    per = pd.read_csv(OUTPUT_DIR / "per_sensor_summary.csv")
    for (ds, sens, metric), g in per[
        (per.sensor.isin(["A023", "A027", "A142"])) & (per.masking == "block")
    ].groupby(["dataset", "sensor", "metric"]):
        lens = sorted(g.gap_length.dropna().unique())
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{"
            + ds.replace("_", r"\_")
            + " "
            + sens
            + " "
            + metric
            + r": mean [95\% CI], five runs; -- denotes"
            r" undefined/incomplete.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "r" * len(lens) + "}",
            r"\toprule",
            "Method & "
            + " & ".join(str(int(length)) for length in lens)
            + r" \\",
            r"\midrule",
        ]
        for method, z in g.groupby("method"):
            cells = []
            for length in lens:
                q = z[z.gap_length == length]
                if len(q) != 1 or not q.iloc[0]["complete"]:
                    cells.append("--")
                else:
                    v = q.iloc[0]
                    cells.append(
                        f"{v['mean']:.2f} [{v.ci_lower:.2f}, {v.ci_upper:.2f}]"
                    )
            lines.append(
                method.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
            )
        (tables / f"{ds}_{sens}_{metric}.tex").write_text(
            "\n".join(
                lines + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
            ),
            encoding="utf-8",
        )
    for ds in df.dataset.unique():
        fig, ax = plt.subplots(figsize=(10, 5))
        g = summary[
            (summary.dataset == ds)
            & (summary.metric == "MAE")
            & (summary.masking == "block")
            & summary.complete
        ]
        for j, (method, z) in enumerate(g.groupby("method")):
            z = z.sort_values("gap_length")
            ax.errorbar(
                z.gap_length,
                z["mean"],
                yerr=z.ci_upper - z["mean"],
                label=method,
                color=plt.get_cmap("tab20")(j),
                linestyle="--" if j % 2 else "-",
                marker=".",
                capsize=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Gap length (10-minute samples)")
        ax.set_ylabel("Test MAE (vehicles)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.set_title(ds + " — five-run mean and 95% CI")
        fig.tight_layout()
        fig.savefig(plots / f"{ds}_gap_mae.png", dpi=150)
        plt.close(fig)
        meta, data, *_ = datasets[ds]
        b = meta["splits"]["test"][0]
        truth = data["values"][b:]
        times = pd.to_datetime(data["timestamps"][b:])
        for name in ["A023", "A027", "A142"]:
            if name not in meta["sensors"]:
                continue
            c = meta["sensors"].index(name)
            sc = "block_144"
            rp = (
                OUTPUT_DIR / "runs" / ds / sc / "linear_s0" / "predictions.npz"
            )
            if not rp.exists():
                continue
            a = np.load(rp)
            sel = a["cols"] == c
            r = a["rows"][sel]
            if not len(r):
                continue
            lo = max(0, r.min() - 36)
            hi = min(len(truth), r.max() + 37)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                times[lo:hi],
                truth[lo:hi, c],
                color="black",
                label="ground truth",
                linewidth=1.5,
            )
            for method in [
                "linear",
                "knn_unscaled",
                "historical",
                "GAIN",
                "GRIN",
                "sage_voronoi",
            ]:
                rp = (
                    OUTPUT_DIR
                    / "runs"
                    / ds
                    / sc
                    / f"{method}_s0"
                    / "predictions.npz"
                )
                if rp.exists():
                    z = np.load(rp)
                    s = z["cols"] == c
                    ax.plot(
                        times[z["rows"][s]],
                        z["prediction"][s],
                        label=method,
                        alpha=0.8,
                    )
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Vehicles per 10 minutes")
            ax.set_title(f"{ds}, {name}, seed 0 (fixed illustrative seed)")
            ax.legend(fontsize=8, ncol=3)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plots / f"{ds}_{name}_144.png", dpi=150)
            plt.close(fig)
    expected = (
        len(cfg["datasets"])
        * len(cfg["seeds"])
        * len(cfg["scenarios"])
        * len(cfg["methods"])
    )
    have = set(
        tuple(r)
        for r in (
            df[["dataset", "method", "scenario", "seed"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )
    inventory = []
    for dc in cfg["datasets"]:
        for method in cfg["methods"]:
            for sc in cfg["scenarios"]:
                for seed in cfg["seeds"]:
                    key = (dc["name"], method, sc["name"], seed)
                    inventory.append(
                        dict(
                            zip(["dataset", "method", "scenario", "seed"], key)
                        )
                        | {
                            "status": (
                                "success"
                                if key in have
                                else "missing_or_failed"
                            )
                        }
                    )
    pd.DataFrame(inventory).to_csv(
        OUTPUT_DIR / "run_inventory.csv", index=False
    )
    (OUTPUT_DIR / "coverage.json").write_text(
        json.dumps(
            {
                "successful_runs": len(df) // 3,
                "expected_main_runs": expected,
                "complete_summary_cells": int(summary.complete.sum()),
                "total_summary_cells": len(summary),
                "replication_unit": "one mask+model seed run on fixed split",
                "n_temporal_datasets": len(cfg["datasets"]),
                "cities": 1,
                "temporal_datasets_are_not_independent_cities": True,
            },
            indent=2,
        )
    )
    print(
        "Analyzed",
        len(df) // 3,
        "runs;",
        int(summary.complete.sum()),
        "complete metric cells",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(OUTPUT_DIR))
    a = ap.parse_args()
    OUTPUT_DIR = Path(a.output_root).resolve()
    main()
