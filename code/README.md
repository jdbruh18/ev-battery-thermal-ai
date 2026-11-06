# Reproducing the experiments

Run the commands below from the **repository root** after installing `requirements.txt`.

Data: NASA PCoE "Battery Data Set" (Saha & Goebel, 2007), downloaded 6 Nov 2026 from
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
(link given on the NASA PCoE Data Set Repository page,
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
B0005, B0006, B0007, and B0018 are in the inner archive `1. BatteryAgingARC-FY08Q4.zip`.

The scripts expect `data/raw/B0005.mat`, `data/raw/B0006.mat`, `data/raw/B0007.mat`, and `data/raw/B0018.mat`. They do not download the dataset. Raw `.mat` and `.zip` files stay out of git.

```bash
pip install -r requirements.txt
mkdir -p data && cd data
curl -L -o b.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip b.zip && unzip "5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip" -d raw
cd ../code
python3 run_experiments.py      # LOBO for RF / XGBoost / MLP -> ../results/
python3 ablation_features.py    # XGBoost feature-set ablation -> ../results/ablation_features_xgb.csv
```

`ablation_features.py` loads data by executing the loader at the top of `run_experiments.py` via a relative path, so both commands must be run with `code/` as the working directory.

Outputs: `dataset_summary.csv`, `metrics_per_fold.csv`, `metrics_summary.csv`, `baseline_mean_predictor.csv`, `ablation_features_xgb.csv`, `hardware.json`, and `fig1`–`fig3` PNGs, all under `results/`. Runtime is about 2 min on 8 vCPU. Seeds are fixed (`random_state=0`); latency numbers will vary by machine.

Note: the `complexity` column is random-forest total tree nodes, MLP parameter count, and for XGBoost a count of dump lines (not a clean parameter count; use `size_kB`).
