# Imputation of Missing Traffic Sensor Data

This repository contains code and experiments for the paper:

**"Imputation of Missing Traffic Sensor Data: A Comparative Study of Classical, Graph-Based, and Deep Spatio-Temporal Methods"**
> **Note**
> This project was developed and tested with **Python 3.11**.  
> Using newer Python versions (e.g., 3.12+) may lead to dependency installation issues.
---

## 📌 Overview

This project investigates different approaches to imputing missing values in traffic sensor data, including:

* Classical statistical methods
* Graph-based methods
* Deep spatio-temporal models (SAGE-Voronoi / ST-GCN)

All experiments are conducted on real-world traffic data from Darmstadt.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/traffic-imputation-benchmark.git
cd traffic-imputation-benchmark
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

### Traffic Data

* `combined_sensors_2023-01-01_2023-01-31.csv`
  Traffic sensor data for Darmstadt (January 2023).
  Used for all experiments.

### Graph Structure

* `edges.csv`
  Graph edge list describing relationships between sensors.

Format:

```
sensor_id, neighbor_id
```

---

## ⚙️ Data Preparation

### Masking Missing Values

Artificial missing values are introduced using:

```
prepare_masked_data.py
```

This script:

* Randomly removes a percentage of observed values
* Preserves the timestamp column
* Ensures at least one value per sensor (by default)

### Example:

```bash
python prepare_masked_data.py \
    --input combined_sensors_2023-01-01_2023-01-31.csv \
    --output data_masked.csv \
    --percent 10 \
    --maskmap mask_map.csv \
    --removed removed_values.csv \
    --removed_long removed_values_long.csv \
    --seed 123
```

---

## 🧠 Imputation Methods

### 1. Baseline Methods

Implemented in:

```
baseline_imputation.py
```

Supported methods:

* `ffill_bfill`
* `median`
* `rolling_mean`
* `spline`
* `knn_columns`

Graph-based:

* `graph_laplacian`
* `graph_temporal`

---

### 2. Deep Spatio-Temporal Models

#### Core files:

**`gcnnmodel.py`**

* Graph convolutional model definitions
* ST-GCN and Voronoi-based extensions

**`data_imputation.py`**

* Performs imputation using trained models
* Reads wide CSV (timestamp + sensors)
* Imputes only missing values (NaNs)
* Contains optional masking (not used in experiments)

**`helper.py`**

* Data loading and preprocessing utilities
* GeoJSON processing

---

## ⚙️ Configuration

Experiments are controlled via a YAML configuration file.

### Key parameters:

#### Data

```yaml
data_csv: "combined_sensors_2023-01-01_2023-01-31.csv"
masked_csv: "data_masked.csv"
output_csv: "data_filled.csv"
```

#### Graph

```yaml
edges_csv: "edges.csv"
sigma2: 0.1
epsilon: 0.95
```

#### Model

```yaml
in_feat: 1
out_feat: 10
lstm_units: 64
```

#### Training

```yaml
epochs: 40
batch_size: 64
input_seq_len: 12
forecast_horizons: [1, 2, 3]
```

#### Baselines

```yaml
baseline_method: "rolling_mean"
```

---

## 🚀 Workflow

### 1. Prepare masked data

```bash
python prepare_masked_data.py ...
```

### 2. Run baseline imputation

```bash
python baseline_imputation.py ...
```

### 3. Train / evaluate deep models

```bash
python train.py
python evaluate.py
```

### 4. Run deep imputation

```bash
python data_imputation.py
```

---

## 📊 Outputs

* `data_filled.csv` — imputed dataset
* `metrics.json` — evaluation metrics
* `imputation_summary.csv` — aggregated results

---

## 🧩 Notes

* Only missing values (NaNs) are imputed — existing values are never overwritten
* External masking (`prepare_masked_data.py`) is used in experiments
* Internal masking in `data_imputation.py` is not used
