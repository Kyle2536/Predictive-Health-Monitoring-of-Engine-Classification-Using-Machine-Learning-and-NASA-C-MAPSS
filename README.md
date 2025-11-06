# Predictive Health Monitoring of Engines (NASA C‑MAPSS)

Classifies aircraft engine health (**Normal = 0** vs **At‑Risk = 1**) from NASA C‑MAPSS turbofan data.  
Focus: safety‑first (high recall) with a tuned Random Forest.

---

## 1) Environment

```bash
# clone + env
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create & activate venv
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt
```
> Requires Python 3.9+

---

## 2) Repo layout

```
.
├── README.md
├── requirements.txt
├── train_FD001.txt                 # example raw C‑MAPSS file (optional demo)
├── notebooks/
│   └── CS4375FinalProject_KyleKeshmeshian.ipynb
├── reports/
│   ├── PredictiveHealthMonitoringOfEngineClassificationUsingMachineLearningAndNASAC-MAPSS-KyleKeshmeshian.pdf
│   └── slides/CS4375FinalProject-KyleKeshmeshian.pptx
└── src/
    ├── features/feature_engineering.py
    └── models/train.py
```
> If you still have `feature_engineering.py` and `train.py` at the repo root, you can either:
> 1) move them into `src/features/` and `src/models/`, or  
> 2) run them from the root as shown below.

---

## 3) Data → features

You need a CSV with a **binary** column named `status` (1 = At‑Risk, 0 = Normal).  
Here’s a minimal script to convert a standard **FD001** training file into that CSV:

<details>
<summary><strong>make_processed_from_fd001.py</strong> (click to expand)</summary>

```python
import pandas as pd

RAW = "train_FD001.txt"         # path to your FD001 raw file
OUT = "data/processed.csv"      # output path

# FD001 has: unit, cycle, 3 settings + 21 sensors (26 cols after unit/cycle)
COLS = (["engine_id","cycle","op1","op2","op3"] +
        [f"sensor_{i}" for i in range(1,22)])

df = pd.read_csv(RAW, sep=r"\s+", header=None, names=COLS)
df = df.sort_values(["engine_id","cycle"]).reset_index(drop=True)

# Remaining Useful Life (RUL) per engine_id
rul_max = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
df = df.merge(rul_max, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Binary label: At‑Risk if RUL <= 30
df["status"] = (df["RUL"] <= 30).astype(int)

# Optional: simple trend feature
for k in [2,3,4,7,11,14]:  # a few sensors
    col = f"sensor_{k}"
    df[f"{col}_avg_of5"] = (df
        .groupby("engine_id")[col]
        .transform(lambda s: s.rolling(5, min_periods=1).mean()))

# Normalize cycle position (0..1)
df["normalized_cycles"] = df["cycle"] / df["max_cycle"]

# Keep a compact set of features for the demo
keep = ["engine_id","cycle","normalized_cycles","status"]        + [f"sensor_{k}" for k in [2,3,4,7,11,14]]        + [f"sensor_{k}_avg_of5" for k in [2,3,4,7,11,14]]

# Ensure output dir exists
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df[keep].to_csv(OUT, index=False)
print(f"Wrote {OUT} with shape={df[keep].shape}")
```
</details>

Run it:

```bash
# ensure data/ exists, then
python make_processed_from_fd001.py
```
You should now have `data/processed.csv` with a `status` column.

---

## 4) Train the model

### If your training script is at the repo <em>root</em> (common when first uploading)
```bash
python train.py --input-csv data/processed.csv --output-json models/rf_results.json
```

### If your code lives under <em>src/</em>
```bash
python src/models/train.py --input-csv data/processed.csv --output-json models/rf_results.json
```

What you get:
- Console printout of metrics  
- `models/rf_results.json` with:
  - `best_params`
  - `best_score_cv_recall`
  - `report` (precision/recall/F1 per class)
  - `confusion_matrix`
  - `roc_auc`

---

## 5) Repro tips

- `RANDOM_STATE=42` in training for repeatability.
- Stratified 80/20 split; 5‑fold CV; `class_weight="balanced"`.
- Optimize **recall** to minimize false negatives (safety‑critical).

---

## 6) Notebook & report

- Notebook: `notebooks/CS4375FinalProject_KyleKeshmeshian.ipynb`  
- Report & slides: `reports/`

---

## 7) Publish

```bash
git add .
git commit -m "Docs: runnable README and data prep"
git push
```

---

## Citation

If you use this project in academic work, please cite the accompanying report and the NASA C‑MAPSS dataset.
