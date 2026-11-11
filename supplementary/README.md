# Supplementary material

Supplementary tables and machine-readable results for **Imputation of Missing Traffic Sensor Data: A Comparative Study of Classical, Graph-Based, and Deep Spatio-Temporal Methods**.

This package accompanies the revised manuscript and uses the final graph results integrated on 6 November 2026. The two periods are January 2023 and October-December 2022 in Darmstadt. The qualified road graphs have 254 and 252 edges, respectively, and no isolated sensors. No experiments were rerun when packaging this supplement.

## Contents

| File or directory | Contents |
| --- | --- |
| `supplementary_material.pdf` | Standalone, 81-page supplement containing all 127 tables. |
| `supplementary_material.tex` | Standalone LaTeX entry point. |
| `supplementary_results_p01.tex` | Tables S1-S96. |
| `supplementary_analyses_p03.tex` | Tables A1-A31. |
| `data/` | 17 CSV files with summaries and individual-run results. |

The table numbering matches the manuscript. All text and table contents are black; revision colours are omitted. The LaTeX files do not depend on the main manuscript's class or bibliography.

## Tables and data

| Tables / analysis | Summary CSV | Individual-run CSV |
| --- | --- | --- |
| S1-S24: primary results, all 13 scenarios and 12 methods | `results_summary.csv`; also included in `supplementary_results_p01.csv` | `results_raw.csv` |
| S25-S96: sensors A023, A027 and A142 | `supplementary_results_p01.csv`, rows with a nonempty `sensor` field | `per_sensor_selected_raw.csv` |
| A1-A3: persistent loss of 25% or 50% of sensor streams | `availability_results_summary.csv` | `availability_results_raw.csv` |
| A4-A9: Graph Temporal graph/scaling controls | `additional_analyses_p03.csv` | `supplements_raw.csv` |
| A10-A12: KNN scaling control at fixed k=10 | `additional_analyses_p03.csv` | `supplements_raw.csv` |
| A13-A24: traffic-count and graph-degree strata | `additional_analyses_p03.csv` | `strata_raw.csv` |
| A25-A30: reconstructed daily sensor-count sums | `additional_analyses_p03.csv` | `downstream_raw.csv` |
| A31: separate one-step Sage-Voronoi forecasting | `additional_analyses_p03.csv` | `forecast_raw.csv` |
| Rounding sensitivity | `additional_rounding_differences_p03.csv` | `rounding_raw.csv` |
| Training effort and checkpoint selection | `additional_training_summary_p03.csv` | `additional_training_runs_p03.csv` |

All filenames in this table are relative to `data/`. `additional_analyses_p03.csv` also includes primary results; use its `analysis`, `variant` and `stratum` columns to distinguish analyses. Some control tables use primary results as reference rows, so consult `results_raw.csv` alongside `supplements_raw.csv`.

`paired_comparisons.csv` contains 858 primary paired contrasts against linear interpolation; `availability_paired_comparisons.csv` contains 132 availability contrasts. These files include the statistical comparisons described in the manuscript.

The primary raw results contain 4,680 metric rows from 1,560 evaluations; the availability results contain 720 metric rows from 240 evaluations. Each evaluation contributes MAE, RMSE and MRE. Per-sensor raw results are restricted to the three sensors displayed in S25-S96 (14,040 metric rows), avoiding duplication of the large all-sensor file. These are metric records, not full prediction arrays or input datasets.

Summary files retain unrounded values; the PDF formats them for readability. Means, sample standard deviations and 95% Student-t intervals use five complete seed runs (seeds 0-4), not individual imputed points. Intervals are conditional on the fixed chronological splits. Both periods come from one city, and the missingness scenarios have not been quantitatively validated against natural missingness.

## Rebuild the PDF

Run the following twice from this directory with a LaTeX distribution such as TeX Live or MiKTeX:

```bash
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

The second pass resolves table links. Required standard packages are `fontenc`, `lmodern`, `inputenc`, `geometry`, `amsmath`, `amssymb`, `booktabs`, `makecell`, `xcolor`, `graphicx`, `textcomp` and `hyperref`. No BibTeX pass is required. The ready-made PDF can be read without installing LaTeX.

## Repository placement

Copy this entire directory as `supplementary/` at the root of the project repository, alongside its main README and experiment scripts. Keep the PDF, all three LaTeX files, this README and `data/` together. Compilation by-products (`.aux`, `.log`, `.out`) need not be uploaded.

Preparing this local package does not publish it to GitHub. The package contains supplementary results; the repository's experiment scripts and configurations remain the entry point for rerunning experiments.
