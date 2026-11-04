"""
Validation-only training audit and separate plain-forecast test
summaries.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib

matplotlib.use("Agg")
# Keep imports after required runtime/environment initialization.
import matplotlib.pyplot as plt  # noqa: E402
from helper import metrics  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    cfg = json.loads((OUTPUT_DIR / "configs/resolved_main.json").read_text())
    rows = []
    forecasts = []
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    for path in (OUTPUT_DIR / "models").rglob("fit.json"):
        fit = json.loads(path.read_text())
        if "best_epoch" not in fit:
            continue
        history = json.loads((path.parent / "history.json").read_text())
        ds = path.parent.parent.name
        method, seed = path.parent.name.rsplit("_s", 1)
        score = np.array(
            [
                x.get("validation_MAE", x.get("validation_masked_mae"))
                for x in history
            ]
        )
        best = int(np.argmin(score))
        assert best + 1 == fit["best_epoch"]
        old = float(score[:20].min())
        new = float(score.min())
        limit = fit["config"]["epochs"]
        rows.append(
            dict(
                dataset=ds,
                method=method,
                seed=int(seed),
                epochs_run=len(history),
                epoch_cap=limit,
                best_epoch=best + 1,
                best_validation_MAE=new,
                best_through_epoch20=old,
                validation_improvement_after20=old - new,
                cap_reached=len(history) == limit,
                best_in_last20=best + 1 > limit - 20,
                training_seconds=fit["training_seconds"],
            )
        )
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(
            np.arange(1, len(score) + 1), score, label="Masked validation MAE"
        )
        ax.axvline(20, color="gray", linestyle=":", label="Previous epoch cap")
        ax.axvline(
            best + 1,
            color="black",
            linestyle="--",
            label="Selected checkpoint",
        )
        ax.set(
            xlabel="Epoch",
            ylabel="Vehicles / 10 minutes",
            title=f"{ds}, {method}, seed {seed}",
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "figures" / f"learning_{ds}_{method}_s{seed}.png",
            dpi=140,
        )
        plt.close(fig)
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "training_budget_audit.csv", index=False
    )
    for p in (OUTPUT_DIR / "forecast").rglob("result.json"):
        r = json.loads(p.read_text())
        z = np.load(p.parent / "predictions.npz")
        for metric, value in metrics(z["truth"], z["prediction"]).items():
            assert np.isclose(value, r["metrics"][metric], equal_nan=True)
            forecasts.append(
                dict(
                    dataset=r["dataset"],
                    method=r["method"],
                    seed=r["seed"],
                    metric=metric,
                    value=value,
                    n_points=r["n_points"],
                    seconds=r["seconds"],
                )
            )
    pd.DataFrame(forecasts).to_csv(
        OUTPUT_DIR / "forecast_raw.csv", index=False
    )
    sums = []
    if forecasts:
        frame = pd.DataFrame(forecasts)
        for key, g in frame.groupby(["dataset", "method", "metric"]):
            assert not g.seed.duplicated().any()
            v = g.value.to_numpy()
            n = len(v)
            sd = np.std(v, ddof=1) if n > 1 else np.nan
            delta = t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            sums.append(
                dict(
                    zip(["dataset", "method", "metric"], key),
                    mean=v.mean(),
                    std=sd,
                    ci_lower=v.mean() - delta,
                    ci_upper=v.mean() + delta,
                    n_runs=n,
                    complete=set(g.seed) == set(cfg["seeds"]),
                )
            )
    pd.DataFrame(sums).to_csv(OUTPUT_DIR / "forecast_summary.csv", index=False)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Sage-Voronoi plain one-step forecasting, mean [95\%"
            r" run-level CI], five training seeds.}"
        ),
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Dataset & Metric & Mean [95\% CI] \\",
        r"\midrule",
    ]
    for r in sums:
        if r["complete"]:
            lines.append(
                r["dataset"].replace("_", r"\_")
                + " & "
                + r["metric"]
                + f" & {r['mean']:.3f} [{r['ci_lower']:.3f},"
                f" {r['ci_upper']:.3f}] "
                + chr(92) * 2
            )
    (OUTPUT_DIR / "tables/forecast_sage.tex").write_text(
        "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    )
    print(
        "Training fits audited",
        len(rows),
        "forecast runs",
        len(forecasts) // 3,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(OUTPUT_DIR))
    a = ap.parse_args()
    OUTPUT_DIR = Path(a.output_root).resolve()
    main()
