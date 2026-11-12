# Imputation of Missing Traffic Sensor Data

This repository contains code and results for:

**Imputation of Missing Traffic Sensor Data: A Comparative Study of Classical,
Graph-Based, and Deep Spatio-Temporal Methods**

> The recorded experimental runtime uses **Python 3.10.5**. Dependency versions
> are pinned in `requirements.txt`. The default GRIN configuration requires a
> CUDA-capable GPU; Sage-Voronoi and GAIN use CPU execution.

## 📌 Overview

The benchmark compares twelve methods on January 2023 and October–December 2022
traffic counts from Darmstadt. It uses chronological 50/20/30 train/validation/test
splits, five seeds, nine asynchronous gap lengths, three random-point missing
fractions and one synchronous gap scenario. Two additional scenarios hide entire
test streams of 25% or 50% of sensors. Test scores never select configurations.

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/bielprze/traffic-imputation-benchmark.git
cd traffic-imputation-benchmark
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows: `venv\Scripts\activate`.
Linux/macOS: `source venv/bin/activate`.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

For GRIN, install the CUDA-enabled PyTorch build appropriate to your CUDA/driver
environment, retaining the pinned PyTorch version. Verify GPU availability with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The experimental scripts, model implementations, helpers, graph inputs and
vendor code are local to this repository. No sibling project folders are needed.

## 📂 Dataset

### Traffic data

- `combined_sensors_2023-01-01_2023-01-31.csv`
- `combined_sensors_2022-10-01_2022-12-31.csv`

Each CSV has a timestamp column followed by sensor columns. Only NaN denotes
missingness. Zero is an observed count. Training-only qualification retains
sensors with at least 90% observed training entries and at least one nonzero
observed training value: 158 January sensors and 159 October–December sensors.

### Graph structure

`edges.csv` and `edges_octdec.csv` contain the same completed source road graph,
with columns `from_id,to_id`. There are 265 rows, representing 259 unique
undirected edges; repeated/reversed pairs are collapsed and self-edges excluded.

`graph_reduction.py` preserves connections through excluded sensors. Two retained
sensors are connected when all internal vertices of a source-graph path are
excluded sensors. Traversal stops at retained sensors. It creates binary edges,
records witness paths and rejects unknown IDs. The resulting graphs contain
254 and 252 edges, respectively, with no isolated nodes. Their edge lists and
added-edge paths are saved as `graph_reduction.json` in each prepared dataset.

`sensors_location.geojson` supplies the separate geometry-based Sage-Voronoi
graph. Changing the road edge files does not change Sage-Voronoi's graph.

## ⚙️ Data Preparation

`helper.py` performs qualification, splits, train-only statistics and missingness
profiling. `prepare_masked_data.py` prepares shared validation/test masks through
the entry point:

```bash
python run_experiments.py --stage prepare --output-root runs/reproduction
```

The saved artifacts include sensor IDs, timestamps, split indices, scalers,
adjacency, mask arrays and dataset metadata. Artificial masks hide observed
values only. The natural-gap profile uses training timestamps only; the qualified
training panels contain no NaN, so synthetic scenarios are not empirical replay.

## 🧠 Imputation Methods

### 1. Baseline methods

`baseline_imputation.py` implements `linear`, `median`, `ffill_bfill`,
`rolling_mean`, `knn`, `knn_unscaled`, `historical`, `graph_laplacian` and
`graph_temporal`. Validation MAE selects configurations from the explicit grids.
Graph solvers reject failed convergence and excessive relative residuals.

### 2. Deep spatio-temporal methods

- `gcnnmodel.py`: Sage-Voronoi training and causal recursive imputation.
- `gainmodel.py`: GAIN training and contemporaneous multivariate imputation.
- `grinmodel.py`: GRIN reference architecture, training and checkpoint handling.
- `vendor/grin/`: author GRIN architecture and accompanying license/source files.
- `grin_experiments.py`: GRIN train/validation selection and test evaluation.
- `data_imputation.py`, `train_GNN.py`: benchmark execution and orchestration.

GRIN uses the author architecture at commit
`4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e`. GRIN is bidirectional; Sage-Voronoi is
causal; GAIN uses the visible sensors at the same timestamp. Consequently this
is a retrospective imputation comparison, not an equal-context real-time ranking.

## ⚙️ Configuration

`config.yaml` is written as JSON (a YAML subset). It contains all datasets,
methods, seeds, masking scenarios, validation grids and training settings.
Seeds are 0–4. Validation masks hide 20% of observed entries.

GRIN: 300-epoch cap, patience 40, 24-sample windows, hidden size 64, batch size
32 and 5120 sampled windows per epoch. GAIN and Sage-Voronoi have 200-epoch caps
and patience 20. Selection and early stopping use validation only. Final scores
use artificially hidden test values, with primary predictions left unrounded.

## 🚀 Workflow

### 1. Reproduce the full benchmark from scratch

```bash
python run_experiments.py --output-root runs/reproduction
```

This prepares data, trains/selects methods, evaluates primary and availability
scenarios, runs controls, aggregates results and checks statistics. Use a new
output directory after changing code, data or configuration. Epoch-boundary
GRIN state and completed fits support resuming with unchanged inputs.

### 2. Run individual stages

```bash
python run_experiments.py --stage train --output-root runs/reproduction
python run_experiments.py --stage evaluate --output-root runs/reproduction
python run_experiments.py --stage supplements --output-root runs/reproduction
python run_experiments.py --stage analyze --output-root runs/reproduction
python run_experiments.py --stage qa --output-root runs/reproduction
```

Stages use the same configuration and prepared output directory. Run them in the
listed order if not using the full command. Selected checkpoint files are not
bundled in this source package: use fresh training, not `--from-checkpoints`.

### 3. Check preparation and stored final statistics

```bash
python test_package.py
python test_graph_and_metrics.py
python verify_results.py --results results
```

`reference/preparation_checks.json` contains compact expected preparation checks.
The optional `test_package.py --checkpoints` requires separately supplied model
weights and prediction checks and is not part of this source-only package.

## 📊 Outputs

`results/` contains the final combined manuscript results, including raw and
summary CSVs, paired comparisons, per-sensor and degree/count strata, graph/scaling
controls, rounding, daily aggregation and availability results. Historical
old/new differences are retained separately within that directory.

New runs write to the requested output directory, with `benchmark/` and `grin/`
subdirectories for models, masks, predictions, logs and local scores, and
`results/` for combined results. GRIN writes epoch progress to `grin/status.json`.
Means, sample SDs and 95% Student-t CIs use five complete seed runs as replications.
Paired tests compare against linear interpolation with within-family Holm correction.

## 🧩 Notes

- Observed entries are preserved; masked ground truth never becomes model input.
- Sensor qualification, scaling and profiling use training data only.
- Both periods are from one city; the CIs do not measure between-city uncertainty.
- Stream-loss experiments vary information availability, not physical density.
- Natural-missingness realism has not been quantitatively validated.
- Forecasting and daily aggregation are separate checks, not forecasting after
  imputation.
