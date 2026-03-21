# Designing a Privacy-Preserving Cybersecurity Threat-Detection Framework Using Differential Privacy

**Dissertation Prototype** — A Python and Streamlit application implementing a
Differential Privacy-enabled machine learning pipeline for network intrusion
detection, evaluated on the CIC-IDS2018 and UNSW-NB15 benchmark datasets.

> This artifact is a research prototype developed under a Design Science Research
> (DSR) methodology. It is intended for academic demonstration and evaluation,
> not production deployment.

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Artifact Overview](#2-artifact-overview)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Datasets](#5-datasets)
6. [Privacy Mechanism](#6-privacy-mechanism)
7. [Setup](#7-setup)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Streamlit Application](#9-streamlit-application)
10. [Experiment Design](#10-experiment-design)
11. [Configuration Reference](#11-configuration-reference)
12. [Module Reference](#12-module-reference)
13. [CLI Reference](#13-cli-reference)
14. [Reproducibility](#14-reproducibility)
15. [Limitations](#15-limitations)
16. [References](#16-references)

---

## 1. Research Context

### Dissertation Title

*Designing a Privacy-Preserving Cybersecurity Threat-Detection Framework Using
Differential Privacy*

### Research Question

> How can Differential Privacy be effectively integrated into machine-learning-based
> cybersecurity threat-detection systems to achieve an optimal balance between
> data privacy and analytic utility?

### Theoretical Framework

This research is grounded in two complementary theoretical foundations:

**Differential Privacy (DP)** formalises privacy guarantees by injecting
statistically calibrated noise into computations, ensuring that the presence or
absence of any single individual's record has a bounded and measurable effect on
analytical outputs (Dwork et al., 2017). The privacy–utility balance is
controlled by the privacy budget parameters ε and δ. Smaller ε represents a
stronger privacy guarantee and, typically, a greater reduction in model utility.

**Design Science Research (DSR)** provides the methodological framework for
creating and evaluating the artifact. DSR emphasises iterative design, empirical
evaluation, and contribution through artifacts that address real-world problems
(Hevner et al., 2004). The prototype developed here constitutes the DSR artifact,
evaluated through quantitative experiments comparing private and non-private
classifiers on benchmark cybersecurity datasets (Chatterjee & Bhattacharjee, 2024).

### Significance

Cybersecurity systems increasingly rely on machine learning to detect threats in
network traffic that often contains confidential or personally identifiable
information. Traditional anonymisation strategies are insufficient against
model-inversion and membership-inference attacks (Shokri et al., 2022). Regulations
such as GDPR and HIPAA mandate stronger data protections, creating an operational
imperative for privacy-aware detection systems (Binns & Veale, 2023). This artifact
demonstrates how Differential Privacy can be embedded in a machine learning
classification pipeline while preserving operationally acceptable detection rates.

---

## 2. Artifact Overview

The prototype implements the full DSR artifact cycle:

| DSR Phase | Implementation |
|---|---|
| Problem identification | Lack of empirically validated DP integration in ML-based IDS |
| Artifact design | Modular Python pipeline: data layer → model layer → evaluation layer |
| Artifact instantiation | DP-enabled classifier trained across a range of ε values |
| Evaluation | Quantitative comparison of non-private baselines vs DP models on held-out test data |
| Reflection | Privacy–utility tradeoff curve; statistical comparison across ε values |

**Core experimental design:** Hold all conditions fixed — dataset, features, train/test
split, random seed — and vary only ε. Measure the resulting change in F1 score,
ROC-AUC, and Detection Rate. The privacy–utility tradeoff curve is the primary
empirical finding.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│       Home · Data Overview · Train Models · Compare         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Evaluation Layer                            │
│           metrics.py            results_store.py             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Model Layer                                │
│    baseline.py (LR · RF · XGBoost)    dp_model.py (DP-LR)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Data Layer                                │
│    loader · preprocessor · sampler · label_mapping          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Storage Layer                               │
│   data/raw (CSV) → data/processed (Parquet) → data/subsets  │
│   results/ (JSON · joblib · CSV)                             │
└─────────────────────────────────────────────────────────────┘
```

The pipeline executes in two discrete stages separated by a disk checkpoint,
ensuring that the computationally expensive preprocessing step runs only once:

- **Stage A — Raw → Processed Parquet:** Chunked CSV reading, cleaning,
  label normalisation, dtype optimisation, and Parquet serialisation.
- **Stage B — Processed → Working Subset:** Stratified sampling, fixed
  train/validation/test split, StandardScaler fitting on the training split only.

All model training and evaluation operates exclusively on the fixed Stage B subset.

---

## 4. Project Structure

```
privacy-preserving-ids-prototype/
│
├── app.py                          # Root-level launch helper
├── config.yaml                     # All paths, hyperparameters, ε values
├── requirements.txt                # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Streamlit theme settings
│
├── app/                            # Streamlit application
│   ├── main.py                     # Entry point — pipeline status home page
│   ├── _path_setup.py              # sys.path bootstrap for page imports
│   └── pages/
│       ├── 1_Data_Overview.py      # Dataset status + distribution charts
│       ├── 2_Train_Models.py       # Baseline + DP training controls
│       └── 3_Compare_Results.py    # Tradeoff curve, tables, confusion matrix
│
├── scripts/                        # Terminal pipeline scripts
│   ├── train_baseline.py           # Train LR, RF, XGBoost
│   ├── train_private.py            # Run DP-LR epsilon sweep
│   └── evaluate.py                 # Generate reports and CSV exports
│
├── src/                            # Core Python library
│   ├── data/
│   │   ├── loader.py               # Raw CSV discovery + chunked reading
│   │   ├── preprocessor.py         # Cleaning, normalisation, Stage A pipeline
│   │   ├── sampler.py              # Stratified subset + train/val/test split
│   │   └── label_mapping.py        # Shared label taxonomy for both datasets
│   ├── models/
│   │   ├── baseline.py             # LR, RF, XGBoost — training + persistence
│   │   ├── dp_model.py             # DP-LR — training + epsilon sweep
│   │   └── model_utils.py          # Pipeline status, model registry
│   ├── evaluation/
│   │   ├── metrics.py              # Model-agnostic metric computation
│   │   └── results_store.py        # JSON persistence + comparison table builders
│   └── utils/
│       ├── config.py               # config.yaml loader + path resolver
│       └── logger.py               # Centralised logging
│
├── data/
│   ├── raw/                        # Place dataset CSVs here — git-ignored
│   │   ├── cic_ids2018/            # One CSV per capture day
│   │   └── unsw_nb15/              # UNSW-NB15_1.csv … UNSW-NB15_4.csv
│   ├── processed/                  # Stage A Parquet output — git-ignored
│   └── subsets/                    # Stage B working subset — git-ignored
│
├── results/                        # All experiment outputs
│   ├── models/                     # Saved .joblib model and scaler files
│   ├── *.json                      # Experiment result records
│   └── *.csv                       # Exported tables for LaTeX / dissertation
│
└── notebooks/                      # Exploratory analysis
```

---

## 5. Datasets

| Dataset | Role | Raw Size | Source |
|---|---|---|---|
| **CIC-IDS2018** | Primary | ~16 GB | Canadian Institute for Cybersecurity |
| **UNSW-NB15** | Secondary (replication) | ~4 GB | UNSW Canberra Cyber |

### Rationale for dataset selection

**CIC-IDS2018** was generated in a purpose-built testbed by the Canadian Institute
for Cybersecurity using CICFlowMeter to extract approximately 80 bidirectional
flow features (Sharafaldin et al., 2018). It encompasses seven attack families —
DoS, DDoS, brute force, botnet, web attacks, and infiltration — recorded over
five days of realistic traffic. It is widely cited in the IDS literature, enabling
comparison with published results.

**UNSW-NB15** was generated at UNSW Canberra using Argus and Bro network monitors,
yielding ~49 features and nine attack categories (Moustafa & Slay, 2015). Its
distinct toolchain and network topology make it a rigorous secondary dataset: if the
privacy–utility tradeoff observed on CIC-IDS2018 replicates on UNSW-NB15, the
finding is dataset-independent.

### Directory layout

```
data/raw/cic_ids2018/
    Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
    Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
    ...

data/raw/unsw_nb15/
    UNSW-NB15_1.csv
    UNSW-NB15_2.csv
    UNSW-NB15_3.csv
    UNSW-NB15_4.csv
```

### Known data quality issues and mitigations

| Issue | Dataset | Mitigation |
|---|---|---|
| Latin-1 encoding | CIC-IDS2018 | `encoding="latin-1"` in loader |
| Leading/trailing whitespace in column names | CIC-IDS2018 | `normalize_column_names()` |
| ±Inf values from ratio-based features | CIC-IDS2018 | Replaced with NaN before imputation |
| Duplicate rows | Both | Intra-chunk + global deduplication |
| Empty `attack_cat` for normal traffic | UNSW-NB15 | Mapped to `BENIGN` |
| Label spelling inconsistencies | CIC-IDS2018 | Explicit mapping in `label_mapping.py` |

### Shared label taxonomy

Both datasets are mapped to a unified three-column schema:

| Column | Type | Description |
|---|---|---|
| `label_raw` | string | Preserved original label — for audit and reproducibility |
| `label_binary` | int8 | 0 = benign, 1 = attack |
| `label_multiclass` | category | Normalised attack category |

Normalised categories: `BENIGN`, `DoS`, `DDoS`, `BruteForce`, `Botnet`,
`Infiltration`, `WebAttack`, `Fuzzing`, `Reconnaissance`, `Exploits`,
`Malware`, `Other`

---

## 6. Privacy Mechanism

### Differential Privacy

This artifact implements (ε, δ)-Differential Privacy applied to the model training
process. The formal guarantee is: for any two datasets D and D′ differing by one
record, and for any output S of a randomised mechanism M:

```
Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D′) ∈ S] + δ
```

Smaller ε provides a stronger privacy guarantee at the cost of greater noise
magnitude and, typically, lower model utility (Dwork et al., 2017).

### Implementation

The prototype uses **IBM diffprivlib** (`diffprivlib.models.LogisticRegression`),
which provides a scikit-learn-compatible Differentially Private Logistic Regression
via output perturbation of the optimisation objective (Holohan et al., 2019).

> **Note on DP-SGD:** The dissertation prospectus proposes DP-SGD via PyTorch Opacus
> for application to deep neural networks. For this prototype, diffprivlib's DP-LR
> was selected because it is sklearn-compatible, requires no additional training
> infrastructure, and provides the same formal (ε, δ)-DP guarantee with a single
> interpretable parameter. A production implementation would extend this to DP-SGD
> applied to a neural network classifier using Opacus, as originally proposed.

### Privacy budget sweep

| ε | Privacy interpretation |
|---|---|
| 0.1 | Very strong — heavy noise; model utility substantially reduced |
| 0.5 | Strong — significant noise |
| 1.0 | Standard DP convention; widely cited threshold in the literature |
| 2.0 | Moderate |
| 5.0 | Weak — approaching non-private performance |
| 10.0 | Near-negligible noise; near-baseline utility expected |

The core experimental result is the **privacy–utility tradeoff curve**: F1 score
plotted against ε, with non-private baseline performance as horizontal reference
lines. This curve directly answers the research question by showing at what privacy
budget the DP model achieves acceptable detection performance.

### Class balancing under DP

`diffprivlib` does not natively support `class_weight='balanced'`. This is handled
by computing inverse-frequency sample weights and passing them via `sample_weight`
to `.fit()`, reproducing sklearn's balanced weighting behaviour.

---

## 7. Setup

### Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| RAM | 16 GB (for full CIC-IDS2018 preprocessing) |
| Disk | 30 GB free (raw data + Parquet + models) |
| OS | Linux or macOS |

### Step 1 — Clone

```bash
git clone <your-repo-url>
cd privacy-preserving-ids-prototype
```

### Step 2 — Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt. Run `source .venv/bin/activate`
at the start of every session.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Place raw datasets

```bash
mkdir -p data/raw/cic_ids2018
mkdir -p data/raw/unsw_nb15

# Copy CIC-IDS2018 day-level CSV files
cp /path/to/cic_ids2018_files/*.csv data/raw/cic_ids2018/

# Copy UNSW-NB15 files
cp /path/to/UNSW-NB15_*.csv data/raw/unsw_nb15/
```

### Step 5 — Verify the installation

```bash
python -c "
from src.utils.config import PATHS, DATASET_CFG
from src.data.label_mapping import map_cic_label
print('Config OK. Paths:', list(PATHS.keys()))
print('Label test:', map_cic_label('Benign'))   # Expected: BENIGN
print('Installation verified.')
"
```

---

## 8. Running the Pipeline

> Always activate the virtual environment first: `source .venv/bin/activate`

### Step 1 — Preprocessing (Stage A)

Reads raw CSVs in chunks, cleans, normalises, and writes processed Parquet files.
Run once per dataset. Safe to re-run — skips if output already exists.

```bash
python -c "
from src.data.preprocessor import run_preprocessing_pipeline
run_preprocessing_pipeline('cic_ids2018')
run_preprocessing_pipeline('unsw_nb15')
"
```

Runtime: 10–40 minutes per dataset depending on hardware.

### Step 2 — Sampling (Stage B)

Draws a stratified 300K-row subset and creates the fixed train/val/test split.

```bash
python -c "
from src.data.sampler import run_sampling_pipeline
run_sampling_pipeline('cic_ids2018')
run_sampling_pipeline('unsw_nb15')
"
```

### Step 3 — Train baseline models

```bash
python scripts/train_baseline.py --dataset cic_ids2018 --label binary
python scripts/train_baseline.py --dataset unsw_nb15   --label binary
```

### Step 4 — Run the DP epsilon sweep

```bash
python scripts/train_private.py --dataset cic_ids2018 --label binary
python scripts/train_private.py --dataset unsw_nb15   --label binary
```

To run specific ε values only:
```bash
python scripts/train_private.py --dataset cic_ids2018 --epsilon 0.1 1.0 10.0
```

### Step 5 — Generate evaluation report

```bash
python scripts/evaluate.py --dataset cic_ids2018 --export-csv --report
python scripts/evaluate.py --dataset unsw_nb15   --export-csv
```

Exports Tables 1 and 2 as CSV to `results/` and prints a key findings summary
including the convergence epsilon (where DP-LR F1 comes within 5% of the LR baseline).

---

## 9. Streamlit Application

### Launch

```bash
streamlit run app/main.py
# Opens at http://localhost:8501

# Alternative:
python app.py
```

### Page guide

**Home** — Pipeline status grid showing ✅/❌ for every stage across both datasets.
Guides the user to the next required action.

**Data Overview** — Shows detected CSV files, runs preprocessing and sampling
pipelines via UI buttons, and displays class distribution charts and split
statistics for the working subset.

**Train Models** — Controls for training all three baselines with a single button,
and for running the DP epsilon sweep with a configurable multiselect. Shows a
live-updating results table as each model completes.

**Compare Results** — The central dissertation demonstration page. Displays:
- Table 1: Baseline model comparison (F1, AUC, MCC, FPR, Detection Rate)
- Privacy–utility tradeoff curve (interactive Plotly, with baseline reference lines)
- Table 2: Tradeoff table (colour-coded: green ≤5% loss, yellow 5–15%, red >15%)
- Confusion matrix viewer (select any trained model)
- Feature importance chart (Random Forest)
- CSV export buttons for dissertation tables

---

## 10. Experiment Design

### Experiment matrix

| Run | Model | ε | Dataset | Label |
|---|---|---|---|---|
| B1 | Logistic Regression | — | CIC-IDS2018 | Binary |
| B2 | Random Forest | — | CIC-IDS2018 | Binary |
| B3 | XGBoost | — | CIC-IDS2018 | Binary |
| P1 | DP-LR | 0.1 | CIC-IDS2018 | Binary |
| P2 | DP-LR | 0.5 | CIC-IDS2018 | Binary |
| P3 | DP-LR | 1.0 | CIC-IDS2018 | Binary |
| P4 | DP-LR | 2.0 | CIC-IDS2018 | Binary |
| P5 | DP-LR | 5.0 | CIC-IDS2018 | Binary |
| P6 | DP-LR | 10.0 | CIC-IDS2018 | Binary |
| B4–B6 | LR, RF, XGB | — | UNSW-NB15 | Binary |
| P7–P12 | DP-LR | 0.1→10.0 | UNSW-NB15 | Binary |

**Fixed across all runs:** dataset subset, split assignments, random seed (42),
StandardScaler fitted on training split only, `class_weight='balanced'`.

**Variable:** model type; ε value (DP runs only).

### Evaluation metrics

| Metric | Symbol | Rationale |
|---|---|---|
| F1 Score (macro) | F1 | Primary metric. Treats all classes equally; robust to class imbalance |
| ROC-AUC | AUC | Threshold-independent; measures overall discrimination ability |
| Matthews Correlation Coefficient | MCC | Optimal single metric for binary imbalanced classification |
| False Positive Rate | FPR | False alarm rate — operationally critical for IDS deployment |
| Detection Rate (TPR) | DR | Fraction of attacks successfully detected |
| Accuracy | Acc | Reported for completeness; not the primary metric on imbalanced data |

> Statistical analysis: paired comparisons between baseline and DP model
> performance across ε values support the quantitative evaluation component
> of the DSR assessment, consistent with Hevner & Gregor (2013).

---

## 11. Configuration Reference

All configuration is managed in `config.yaml` at the project root.
No paths or hyperparameters are hardcoded in source files.

```yaml
paths:
  raw_data:       data/raw/
  processed_data: data/processed/
  subsets:        data/subsets/
  results:        results/
  models:         results/models/

datasets:
  cic_ids2018:
    name: "CIC-IDS2018"
    label_column: "Label"
    encoding: "latin-1"
    separator: ","
  unsw_nb15:
    name: "UNSW-NB15"
    label_column: "label"
    encoding: "utf-8"
    separator: ","

sampling:
  subset_size: 300000       # rows drawn per dataset (stratified)
  random_seed: 42           # fixed for reproducibility
  test_split: 0.2           # 20% test, 20% val, 60% train

privacy:
  epsilon_values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  default_epsilon: 1.0

models:
  n_estimators: 100
  max_depth: 10
  n_jobs: -1                # use all CPU cores
```

**Common adjustments:**

| Goal | Change |
|---|---|
| Faster experimentation | `subset_size: 50000` |
| Add an ε value | Append to `epsilon_values` |
| Stronger baseline | `n_estimators: 200` |
| Reproduce a specific result | Keep `random_seed: 42` unchanged |

---

## 12. Module Reference

### `src/utils/config.py`
Loads `config.yaml` at import time. Exposes `PATHS` (absolute Path objects),
`DATASET_CFG`, `SAMPLING_CFG`, `PRIVACY_CFG`, `MODEL_CFG`. Creates all output
directories on import. Import this in any module that needs a path or parameter.

### `src/utils/logger.py`
`get_logger(__name__)` returns a consistently formatted logger.
Format: `timestamp | level | module | message`.

### `src/data/label_mapping.py`
Single source of truth for the threat taxonomy. Contains `CIC_IDS2018_LABEL_MAP`
and `UNSW_NB15_LABEL_MAP` with vectorised mapping functions
`apply_cic_label_mapping()` and `apply_unsw_label_mapping()`. Also provides
`encode_multiclass_labels()` for integer encoding via sklearn `LabelEncoder`.

### `src/data/loader.py`
`list_raw_files(dataset_name)` discovers CSV files in `data/raw/{dataset}/`.
`read_csv_chunked(filepath, dataset_name, chunk_size)` yields chunks with
dataset-specific encoding and separator from config. `load_processed()` and
`load_subset()` load Stage A and Stage B Parquet files respectively.

### `src/data/preprocessor.py`
Exposes individual cleaning functions (each `DataFrame → DataFrame`) and the
top-level orchestrator `run_preprocessing_pipeline(dataset_name, force=False)`.
Outputs Stage A Parquet and a JSON file recording dropped column names.

### `src/data/sampler.py`
`run_sampling_pipeline(dataset_name)` is the one-call orchestrator for Stage B.
`get_split_arrays(dataset_name, label, scale)` returns model-ready numpy arrays
with StandardScaler fitted on the training split and persisted to disk.

### `src/models/baseline.py`
`train_all_baselines(dataset_name, label)` trains all three baselines.
`get_feature_importances(dataset_name, label)` returns sorted Random Forest
importances for dissertation analysis.

### `src/models/dp_model.py`
`run_epsilon_sweep(dataset_name, label)` sweeps all ε values from config.yaml.
`estimate_data_norm(X_train)` computes the L2 norm bound required by diffprivlib.
`train_dp_model(dataset_name, epsilon, label, force)` trains at a single ε.

### `src/evaluation/metrics.py`
`evaluate_model(model, X, y, label_type)` is model-agnostic and called
identically for baseline and DP models. Returns all metrics as a plain Python
dict (JSON-serialisable). `compute_privacy_utility_tradeoff()` augments DP results
with absolute and percentage F1 loss relative to the LR baseline.

### `src/evaluation/results_store.py`
`save_results()` uses upsert logic — re-running one ε updates only that entry.
`build_comparison_table()` produces dissertation Table 1.
`build_tradeoff_table()` produces dissertation Table 2 with loss columns.
`export_to_csv()` writes any table to `results/` for LaTeX import.

### `src/models/model_utils.py`
`get_pipeline_status(dataset_name, label)` returns a full boolean dict covering
every pipeline stage. Used by the Streamlit home page to drive the status display.

---

## 13. CLI Reference

All scripts are run from the project root with the virtual environment active.

### `scripts/train_baseline.py`

```
python scripts/train_baseline.py --dataset {cic_ids2018|unsw_nb15}
                                  [--label {binary|multiclass}]
                                  [--model {logistic_regression|random_forest|xgboost}]
                                  [--force]
```

Trains one or all baseline models. Results saved to
`results/{dataset}_baselines_{label}.json`. Skips already-trained models unless
`--force` is set.

### `scripts/train_private.py`

```
python scripts/train_private.py --dataset {cic_ids2018|unsw_nb15}
                                 [--label {binary|multiclass}]
                                 [--epsilon ε [ε ...]]
                                 [--force]
```

Runs the DP-LR epsilon sweep. Defaults to all values from `config.yaml`.
Results saved to `results/{dataset}_dp_sweep_{label}.json`.

### `scripts/evaluate.py`

```
python scripts/evaluate.py --dataset {cic_ids2018|unsw_nb15}
                            [--label {binary|multiclass}]
                            [--split {val|test}]
                            [--export-csv]
                            [--report]
```

Generates Tables 1 and 2, prints a key findings summary, and optionally exports
CSVs and per-model sklearn classification reports.

---

## 14. Reproducibility

The following conditions ensure that results can be reproduced exactly:

- **Fixed random seed** (default: 42) used by all sampling, splitting, and
  model training operations, configured in `config.yaml`.
- **Fixed subset:** The working subset Parquet includes a `split` column.
  All experiments use identical row assignments once the subset is created.
- **Scaler and encoder** fitted exclusively on the training split and saved to
  `results/models/`. The same artefacts are used for all evaluations.
- **Idempotent pipeline:** Each stage skips silently if its output exists
  (unless `force=True` is passed), preventing accidental re-processing.
- **Config-driven:** All hyperparameters and ε values live in `config.yaml`.

To reproduce from scratch:

```bash
rm -rf data/processed/ data/subsets/ results/models/ results/*.json results/*.csv

python -c "from src.data.preprocessor import run_preprocessing_pipeline; \
           run_preprocessing_pipeline('cic_ids2018')"
python -c "from src.data.sampler import run_sampling_pipeline; \
           run_sampling_pipeline('cic_ids2018')"
python scripts/train_baseline.py --dataset cic_ids2018
python scripts/train_private.py  --dataset cic_ids2018
python scripts/evaluate.py       --dataset cic_ids2018 --export-csv
```

---

## 15. Limitations

1. **Centralised deployment only.** DP is applied to model training on a
   centralised dataset. A federated architecture — where DP protects distributed
   training across multiple clients — is not implemented (Naseri et al., 2020).

2. **Prototype privacy mechanism.** The implementation uses diffprivlib's output
   perturbation DP-LR. The proposed full implementation would use DP-SGD via
   Opacus applied to neural network classifiers, which may exhibit different
   privacy–utility characteristics (Mironov & Talwar, 2023; Tian et al., 2023).

3. **Convergence at very low ε.** At ε ≤ 0.5, the DP optimiser may not converge
   within the iteration limit, producing near-random outputs. This is consistent
   with published findings on the limits of DP at strong privacy budgets
   (Wang & Zhang, 2024) and is interpreted as empirical evidence of the
   privacy–utility tension rather than an implementation failure.

4. **Subset-based evaluation.** All experiments use a 300K-row stratified subset.
   Results may differ at full dataset scale, though stratified sampling preserves
   class proportions and the subset size is consistent with published IDS literature.

5. **Dataset quality.** CIC-IDS2018 has documented quality issues including
   duplicate records and potential feature leakage. Preprocessing mitigates these
   but cannot eliminate all artefacts (Engelen et al., 2021).

6. **Adversarial resilience not evaluated.** The artifact measures classification
   performance degradation under DP noise but does not directly measure
   susceptibility to membership-inference or model-inversion attacks before and
   after applying DP (Shokri et al., 2022).

---

## 16. References

Binns, R., & Veale, M. (2023). Governing machine learning risk: Privacy,
accountability, and design implications. *AI & Society, 38*(3), 879–894.
https://doi.org/10.1007/s00146-023-01612-9

Chatterjee, S., & Bhattacharjee, A. (2024). Applying design science principles
to adaptive cybersecurity analytics. *Computers & Security, 143*, 103095.
https://doi.org/10.1016/j.cose.2024.103095

Dwork, C., Roth, A., & Vadhan, S. (2017). The algorithmic foundations of
differential privacy. *Foundations and Trends in Theoretical Computer Science,
9*(3–4), 211–407. https://doi.org/10.1561/0400000042

Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion
detection dataset: The CICIDS2017 case study. *IEEE Security and Privacy
Workshops*, 7–13. https://doi.org/10.1109/SPW53761.2021.00009

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in
information systems research. *MIS Quarterly, 28*(1), 75–105.
https://doi.org/10.2307/25148625

Hevner, A. R., & Gregor, S. (2013). Positioning and presenting design science
research for maximum impact. *MIS Quarterly, 37*(2), 337–355.
https://doi.org/10.25300/MISQ/2013/37.2.01

Holohan, N., Braghin, S., Mac Aonghusa, P., & Levacher, K. (2019). Diffprivlib:
The IBM differential privacy library. *arXiv preprint arXiv:1907.02444*.

Mironov, I., & Talwar, K. (2023). Calibrating privacy budgets for deep learning
systems. *Proceedings of Privacy Enhancing Technologies, 2023*(4), 54–72.
https://doi.org/10.56553/popets-2023-0110

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network
intrusion detection systems. *Military Communications and Information Systems
Conference (MilCIS)*, 1–6. https://doi.org/10.1109/MilCIS.2015.7348942

Naseri, M., Hayes, J., & De Cristofaro, E. (2020). Local and central differential
privacy for federated learning in practice. *USENIX Security Symposium 2020*, 1–16.

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a
new intrusion detection dataset and intrusion traffic characterization. *ICISSP
2018*, 108–116. https://doi.org/10.5220/0006639801080116

Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2022). Membership inference
attacks against machine learning models: A survey. *ACM Computing Surveys, 54*(7),
1–40. https://doi.org/10.1145/3453400

Tian, Y., Liu, Z., & Chen, X. (2023). Differential privacy for deep learning in
cybersecurity applications: A survey and future directions. *IEEE Access, 11*,
13645–13663. https://doi.org/10.1109/ACCESS.2023.3240178

Wang, K., & Zhang, Y. (2024). Balancing privacy and utility in machine-learning-based
intrusion detection with differential privacy. *Journal of Cybersecurity Research,
10*(2), 22–38. https://doi.org/10.1016/j.jcsr.2024.100189

---

*Prototype developed as part of a doctoral dissertation. Not licensed for commercial use.*
