# AI-driven predictive thermal management for EV batteries

Leave-one-battery-out temperature estimation for 18650 lithium-ion cells, using lightweight models that take only electrical measurements (no temperature history). This repository holds the ICCAQI 2026 double-blind draft, the scripts that produce the tables and figures, and the reported results.

The paper compares a random forest, XGBoost, and a small multilayer perceptron (865 parameters) on discharge cycles from NASA PCoE cells B0005, B0006, B0007, and B0018 (185,721 samples).

## Key results

Under leave-one-battery-out (LOBO) evaluation, **XGBoost** reached **MAE 0.97 °C, RMSE 1.16 °C, and R² 0.906**. Per-fold metrics, the training-mean baseline, model size and latency, and the feature ablation are in [`results/`](results/) and in Section 5 of [`paper/manuscript.md`](paper/manuscript.md). Those files are the experimental record; this README does not add further metrics.

## Repository layout

| Path | Contents |
|---|---|
| `code/run_experiments.py` | LOBO comparison (random forest, XGBoost, MLP), baseline, size, latency, and figures |
| `code/ablation_features.py` | XGBoost feature-set ablation under the same LOBO protocol |
| `code/README.md` | Script notes (outputs, seeds, what the `complexity` column means) |
| `results/` | Metric CSVs, `hardware.json`, dataset notes, and figures 1–3 |
| `paper/manuscript.md` | Double-blind paper draft (Markdown). Figure links point at `../results/` |
| `paper/manuscript.docx` | Same draft in Word, with the three figures embedded |

The draft is anonymous. Do not add author names.

## Reproduce

### 1. Install dependencies

The reported run used Python 3.13 on an 8-vCPU Intel Xeon virtual machine (see `results/hardware.json`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`.

### 2. Download the NASA battery data

The experiments use the NASA Prognostics Center of Excellence **Battery Data Set** (Saha & Goebel, 2007):

> B. Saha and K. Goebel, "Battery Data Set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.  
> https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Download the archive linked from that page (retrieved 6 November 2026):

https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

Cells **B0005**, **B0006**, **B0007**, and **B0018** are in the inner archive `1. BatteryAgingARC-FY08Q4.zip`. From the repository root:

```bash
mkdir -p data && cd data
curl -L -o b.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip b.zip && unzip "5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip" -d raw
cd ..
```

The loaders read `data/raw/B0005.mat`, `data/raw/B0006.mat`, `data/raw/B0007.mat`, and `data/raw/B0018.mat`. If the inner archive extracts into a subdirectory, move those four files up into `data/raw/`. Raw `.mat` and `.zip` files are gitignored and are not part of this repository.

Only discharge cycles are used (2 A constant-current discharge). See `results/processed_dataset_notes.md`.

### 3. Run

From the repository root:

```bash
cd code
python3 run_experiments.py       # LOBO for RF / XGBoost / MLP -> ../results/
python3 ablation_features.py     # feature-set ablation -> ../results/ablation_features_xgb.csv
```

`ablation_features.py` executes the data loader in `run_experiments.py` and must be started from `code/`.

Runtime is about 2 minutes on 8 vCPU. Seeds are fixed (`random_state=0`); latency will vary by machine. Package versions are not pinned, so a new run is not guaranteed to match every stored digit. The tables and figures in `results/` are the numbers and plots reported in the paper. Re-running overwrites those files. The `complexity` column is random-forest node count, MLP parameter count, or (for XGBoost) booster dump lines — use `size_kB` for model size. Details are in [`code/README.md`](code/README.md).

## Data citation

Saha, B. and Goebel, K. (2007). *Battery Data Set*. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
