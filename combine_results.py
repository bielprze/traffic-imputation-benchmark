"""
Combine GRIN and the other eleven methods into one verified result grid.
"""

import os
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
# Keep imports after required runtime/environment initialization.
import matplotlib.pyplot as plt  # noqa: E402

import analyze_results as analysis  # noqa: E402
from grinmodel import atomic_json  # noqa: E402
from grin_experiments import sha, progress  # noqa: E402

HERE = Path(__file__).resolve().parent
SHARED = (
    Path(os.environ.get("TRAFFIC_RUN_ROOT", HERE / "runs/reproduction"))
    / "benchmark"
)


def pair_contrasts(raw):
    rows = []
    for (ds, scenario, metric), group in raw.groupby(
        ["dataset", "scenario", "metric"]
    ):
        reference = group[group.method == "linear"][
            ["seed", "value", "mask_sha256"]
        ]
        for method, other in group.groupby("method"):
            if method == "linear":
                continue
            paired = other.merge(
                reference,
                on="seed",
                suffixes=("", "_ref"),
                validate="one_to_one",
            )
            assert set(paired.seed) == set(range(5))
            assert (paired.mask_sha256 == paired.mask_sha256_ref).all()
            delta = paired.value - paired.value_ref
            sd = delta.std(ddof=1)
            ci = stats.t.ppf(0.975, 4) * sd / np.sqrt(5)
            p = (
                stats.ttest_1samp(delta, 0).pvalue
                if sd
                else float(delta.mean() == 0)
            )
            rows.append(
                dict(
                    dataset=ds,
                    scenario=scenario,
                    metric=metric,
                    method=method,
                    reference="linear",
                    mean_difference=delta.mean(),
                    ci_lower=delta.mean() - ci,
                    ci_upper=delta.mean() + ci,
                    p_raw=p,
                    n_pairs=5,
                )
            )
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, group in df.groupby(["dataset", "scenario", "metric"]):
        order = group.sort_values("p_raw").index
        df.loc[order, "p_holm"] = np.minimum(
            1,
            np.maximum.accumulate(
                df.loc[order, "p_raw"].to_numpy()
                * np.arange(len(order), 0, -1)
            ),
        )
    return df


def merge_csv(benchmark_path, grin_path, destination, manifest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = True
    total = 0
    columns = None
    for source, is_grin in [(benchmark_path, False), (grin_path, True)]:
        for chunk in pd.read_csv(source, chunksize=100000):
            if is_grin:
                assert set(chunk.method) == {"GRIN"}
            else:
                chunk = chunk[chunk.method != "GRIN"].copy()
            chunk["source_namespace"] = "grin" if is_grin else "benchmark"
            if len(chunk):
                if columns is None:
                    columns = list(chunk.columns)
                if set(chunk.columns) != set(columns):
                    raise ValueError(f"Incompatible CSV columns: {source}")
                chunk = chunk[columns]
                chunk.to_csv(
                    destination,
                    mode="w" if first else "a",
                    index=False,
                    header=first,
                )
                total += len(chunk)
                first = False
    manifest.append(
        {
            "destination": str(destination),
            "rows": total,
            "benchmark_source": str(benchmark_path),
            "benchmark_sha256": sha(benchmark_path),
            "grin_source": str(grin_path),
            "grin_sha256": sha(grin_path),
            "sha256": sha(destination),
        }
    )


def tables_and_figures(delivery, out):
    table_dir, figure_dir = delivery / "tables", delivery / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    summary = pd.read_csv(delivery / "results_summary.csv")
    for (ds, scenario, metric), group in summary.groupby(
        ["dataset", "scenario", "metric"]
    ):
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\caption{"
            + f"{ds} {scenario} {metric}".replace("_", r"\_")
            + r": mean [95\% CI], five runs; source-aligned GRIN.}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Method & Mean [95\% CI] \\",
            r"\midrule",
        ]
        for _, row in group.sort_values("method").iterrows():
            cell = (
                f'{row["mean"]:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]'
                if row.complete
                else "--"
            )
            lines.append(
                row["method"].replace("_", r"\_") + " & " + cell + r" \\"
            )
        (table_dir / f"primary_{ds}_{scenario}_{metric}.tex").write_text(
            "\n".join(
                lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            ),
            encoding="utf-8",
        )
    for ds, group in summary[
        (summary.metric == "MAE") & (summary.masking == "block")
    ].groupby("dataset"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, (method, g) in enumerate(group.groupby("method")):
            g = g.sort_values("gap_length")
            ax.errorbar(
                g.gap_length,
                g["mean"],
                yerr=g.ci_upper - g["mean"],
                color=plt.get_cmap("tab20")(j),
                linestyle="--" if j % 2 else "-",
                marker=".",
                capsize=2,
                label=method,
            )
        ax.set_xscale("log", base=2)
        ax.set(
            xlabel="Gap length (10-minute samples)",
            ylabel="Test MAE (vehicles)",
            title=f"{ds} — source-aligned GRIN",
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"primary_{ds}_gap_mae.png", dpi=150)
        plt.close(fig)
    per = pd.read_csv(delivery / "per_sensor_summary.csv")
    selected = per[
        per.sensor.isin(["A023", "A027", "A142"]) & (per.masking == "block")
    ]
    for (ds, sensor, metric), group in selected.groupby(
        ["dataset", "sensor", "metric"]
    ):
        lengths = sorted(group.gap_length.unique())
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{"
            + f"{ds} {sensor} {metric}".replace("_", r"\_")
            + r": mean [95\% CI], five runs; source-aligned GRIN.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "r" * len(lengths) + "}",
            r"\toprule",
            "Method & " + " & ".join(str(int(x)) for x in lengths) + r" \\",
            r"\midrule",
        ]
        for method, g in group.groupby("method"):
            cells = []
            for length in lengths:
                r = g[g.gap_length == length]
                if len(r) != 1 or not r.iloc[0].complete:
                    cells.append("--")
                else:
                    row = r.iloc[0]
                    cells.append(
                        f'{row["mean"]:.2f} [{row.ci_lower:.2f},'
                        f" {row.ci_upper:.2f}]"
                    )
            lines.append(
                method.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
            )
        (table_dir / f"primary_{ds}_{sensor}_{metric}.tex").write_text(
            "\n".join(
                lines + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
            ),
            encoding="utf-8",
        )
    for ds in summary.dataset.unique():
        meta = json.loads((out / "data" / ds / "dataset.json").read_text())
        with np.load(out / "data" / ds / "data.npz") as z:
            b = meta["splits"]["test"][0]
            truth = z["values"][b:]
            dates = pd.to_datetime(z["timestamps"][b:])
        for sensor in ["A023", "A027", "A142"]:
            if sensor not in meta["sensors"]:
                continue
            c = meta["sensors"].index(sensor)
            with np.load(
                SHARED / "runs" / ds / "block_144/linear_s0/predictions.npz"
            ) as z:
                rows = z["rows"][z["cols"] == c]
            lo, hi = max(0, rows.min() - 36), min(len(truth), rows.max() + 37)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                dates[lo:hi],
                truth[lo:hi, c],
                color="black",
                label="Ground truth",
            )
            for method in [
                "linear",
                "knn_unscaled",
                "historical",
                "GAIN",
                "GRIN",
                "sage_voronoi",
            ]:
                source = out if method == "GRIN" else SHARED
                with np.load(
                    source
                    / "runs"
                    / ds
                    / "block_144"
                    / f"{method}_s0/predictions.npz"
                ) as z:
                    hit = z["cols"] == c
                    ax.plot(
                        dates[z["rows"][hit]],
                        z["prediction"][hit],
                        label=method,
                        alpha=0.8,
                    )
            ax.set(
                title=f"{ds}, {sensor}, fixed seed 0; source-aligned GRIN",
                ylabel="Vehicles per 10 minutes",
            )
            ax.legend(ncol=3, fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(figure_dir / f"primary_{ds}_{sensor}_144.png", dpi=150)
            plt.close(fig)


def write_results_report(delivery, training):
    (delivery / "results_report.md").write_text(
        "# Recomputed final results\n\nGRIN uses the corrected"
        " author-reference implementation. All other methods use the final"
        " NaN-only protocol. See repository README and full CSV files. Five"
        " seeds are the replication unit; confidence intervals are conditional"
        " on fixed splits. Mean ranks are descriptive; paired comparisons are"
        " versus linear interpolation only.\n",
        encoding="utf-8",
    )


def analyze_all(out):
    progress(out, "analysis", {}, major=True)
    for namespace in [out, out / "availability"]:
        (namespace / "logs").mkdir(exist_ok=True)
        analysis.OUTPUT_DIR = namespace
        analysis.main()
    delivery = out.parent / "results"
    delivery.mkdir(exist_ok=True)
    manifest = []
    for kind in ["raw", "summary"]:
        merge_csv(
            SHARED / f"results_{kind}.csv",
            out / f"results_{kind}.csv",
            delivery / f"results_{kind}.csv",
            manifest,
        )
        for name in ["per_sensor", "strata", "downstream", "rounding"]:
            merge_csv(
                SHARED / f"{name}_{kind}.csv",
                out / f"{name}_{kind}.csv",
                delivery / f"{name}_{kind}.csv",
                manifest,
            )
        merge_csv(
            SHARED / "availability" / f"results_{kind}.csv",
            out / "availability" / f"results_{kind}.csv",
            delivery / "availability" / f"results_{kind}.csv",
            manifest,
        )
    raw = pd.read_csv(delivery / "results_raw.csv")
    summary = pd.read_csv(delivery / "results_summary.csv")
    assert len(raw) == 4680 and len(summary) == 936 and summary.complete.all()
    assert not raw.duplicated(
        ["dataset", "method", "scenario", "seed", "metric"]
    ).any()
    pairs = pair_contrasts(raw)
    assert len(pairs) == 858
    pairs.to_csv(delivery / "paired_comparisons.csv", index=False)
    av = pd.read_csv(delivery / "availability/results_raw.csv")
    assert len(av) == 720
    pair_contrasts(av).to_csv(
        delivery / "availability/paired_comparisons.csv", index=False
    )
    tables_and_figures(delivery, out)
    training = []
    for p in sorted((out / "models").rglob("fit.json")):
        f = json.loads(p.read_text())
        history = json.loads((p.parent / "history.json").read_text())
        training.append(
            {
                "dataset": p.parent.parent.name,
                **{
                    k: f[k]
                    for k in [
                        "method",
                        "seed",
                        "best_epoch",
                        "epochs_run",
                        "validation_MAE",
                        "optimizer_updates",
                        "sampled_windows",
                        "training_seconds",
                        "budget_limited",
                        "parameter_count",
                    ]
                },
            }
        )
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_MAE"] for r in history],
        )
        ax.axvline(
            f["best_epoch"],
            linestyle="--",
            color="black",
            label="Selected checkpoint",
        )
        ax.set(
            xlabel="Epoch (5120 sampled windows)",
            ylabel="Validation MAE",
            title=p.parent.parent.name + f', seed {f["seed"]}',
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            delivery
            / "figures"
            / f'learning_{p.parent.parent.name}_GRIN_s{f["seed"]}.png',
            dpi=140,
        )
        plt.close(fig)
    pd.DataFrame(training).to_csv(
        delivery / "grin_training_audit.csv", index=False
    )
    write_results_report(delivery, training)
    # Forecast and supplementary studies did not use GRIN; keep exact
    # artifacts.
    for name in [
        "forecast_raw.csv",
        "forecast_summary.csv",
        "supplements_raw.csv",
        "supplements_summary.csv",
    ]:
        shutil.copyfile(SHARED / name, delivery / name)
        manifest.append(
            {
                "unchanged_source": str(SHARED / name),
                "destination": str(delivery / name),
                "sha256": sha(SHARED / name),
            }
        )
    shutil.copyfile(
        SHARED / "tables/forecast_sage.tex",
        delivery / "tables/forecast_sage.tex",
    )
    atomic_json(delivery / "assembly_manifest.json", manifest)
    methodology = HERE / "docs/GRIN.md"
    if methodology.is_file():
        shutil.copyfile(methodology, delivery / "grin_methodology.md")
    (delivery / "README.md").write_text(
        "# Recomputed benchmark results\n\nAll twelve methods use paired masks"
        " and five seeds on fixed chronological splits. Confidence intervals"
        " use run-level replication. See the repository README.\n",
        encoding="utf-8",
    )
    progress(
        out,
        "analysis_completed",
        {
            "primary_metric_rows": len(raw),
            "summary_cells": len(summary),
            "paired_contrasts": len(pairs),
            "next": "artifact_QA_and_checkpoint_restore",
        },
        major=True,
    )
