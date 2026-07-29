# Startup Success Prediction Using Machine Learning
### A VC-Grade Deal Screening & Scoring Engine

---

## 1. Project Overview

This project builds an end-to-end machine learning system that estimates the **probability that a startup will succeed** (defined as reaching a priced follow-on round, acquisition, or IPO, versus shutting down / going dormant) using company-level features such as funding history, team composition, market category, geography, and growth signals.

The system is designed to mimic a **VC associate's early-stage deal screening workflow**: ingest raw company/deal data → engineer investor-relevant features → train a calibrated classifier → explain *why* the model scored a company the way it did (via SHAP) → surface the result in a lightweight scoring dashboard.

It is built to run entirely in **Google Colab**, using publicly available startup datasets (Crunchbase exports / Kaggle startup datasets), with a modular codebase that can later be pointed at a live Crunchbase/AngelList API feed.

---

## 2. Real-World Finance Use Case

Venture capital firms receive far more inbound deal flow than partners can manually diligence. Analysts spend significant time on **top-of-funnel triage**: is this company worth a first call?

ML-assisted sourcing tools (used in some form by firms like EQT Ventures ["Motherbrain"], SignalFire, and Correlation Ventures) do the following:

- **Score inbound deal flow** so analysts prioritize high-probability opportunities first
- **Flag "hidden gem" companies** that don't yet have hype/press coverage but show strong fundamentals
- **Standardize screening criteria** across analysts to reduce inconsistent, gut-feel decisions
- **Provide explainability** (SHAP-style reasoning) so partners can sanity-check a model's recommendation before an investment committee meeting

This project reproduces that workflow at a portfolio-project scale: a reproducible, explainable classifier plus a screening dashboard — the kind of artifact a quant/data role at a VC, growth-equity shop, or fintech underwriting team would recognize immediately.

**Important honesty note (include this in your README/resume context):** predicting "startup success" is inherently noisy — survivorship bias, missing private data, and small sample sizes mean this is a *decision-support/screening* tool, not a definitive predictor. The project should always be framed as "ML-assisted screening," never as a guarantee.

---

## 3. System Architecture

```
                      ┌───────────────────────────┐
                      │   Data Sources             │
                      │  Kaggle CSV / Crunchbase   │
                      │  export / AngelList export │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  1. Data Ingestion Layer    │
                      │  data_loader.py             │
                      │  - schema validation         │
                      │  - dedup / merge sources     │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  2. Cleaning & Feature Eng  │
                      │  feature_engineering.py     │
                      │  - missing value handling   │
                      │  - categorical encoding     │
                      │  - derived VC features      │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  3. Modeling Layer          │
                      │  model_training.py          │
                      │  - train/test split         │
                      │  - XGBoost / LightGBM       │
                      │  - hyperparameter search    │
                      │  - calibration               │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  4. Explainability Layer    │
                      │  explainability.py          │
                      │  - SHAP global + local       │
                      │  - feature importance        │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  5. Evaluation & Dashboard  │
                      │  evaluation.py / dashboard   │
                      │  - ROC / PR curves           │
                      │  - confusion matrix           │
                      │  - scorecard for new startups │
                      └───────────────────────────┘
```

**Data flow summary:** raw CSVs → validated & merged DataFrame → engineered feature matrix `X` + target `y` → train/validation/test split (time-aware where possible) → gradient-boosted classifier (XGBoost primary, LightGBM benchmark) → probability calibration → SHAP explainability → metrics + visual dashboard → exportable "startup scorecard" function usable on new/unseen companies.

---

## 4. Required APIs and Data Sources

| Source | Purpose | Access Notes |
|---|---|---|
| **Kaggle "Startup Success Prediction" dataset** (and similar Crunchbase-derived Kaggle sets) | Primary training data (offline CSV) | Free via `kaggle` API or manual download; no auth needed beyond Kaggle account |
| **Crunchbase Basic/Pro API** | Optional live enrichment (funding rounds, categories, founders) | Requires paid API key; rate-limited; used only if `USE_LIVE_API=True` |
| **AngelList (Wellfound) data exports** | Optional supplementary team/hiring signals | AngelList API access is restricted; typically used via bulk export/partner data |
| **Public company registries (OpenCorporates)** | Optional entity resolution / incorporation date verification | Free tier available |

For reproducibility, the code defaults to **offline CSV mode** using a Kaggle-style schema, with a clean abstraction (`CrunchbaseAPIClient`) that can be switched on with a real API key without touching the modeling code.

---

## 5. Required Python Libraries

```
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
matplotlib
seaborn
plotly
requests
joblib
imbalanced-learn
```

---

## 6. Folder/File Structure

Even though the build runs in Colab, structure the notebook/Drive folder like a real repo so it ports cleanly to GitHub:

```
startup-success-prediction/
│
├── README.md
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
│
├── data/
│   ├── raw/                     # original Kaggle/Crunchbase CSVs
│   ├── processed/                # cleaned + feature-engineered data
│   └── external/                 # optional API pulls
│
├── notebooks/
│   └── startup_success_prediction.ipynb   # main Colab notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # ingestion + Crunchbase API client
│   ├── feature_engineering.py     # cleaning + derived features
│   ├── model_training.py          # XGBoost/LightGBM training + tuning
│   ├── explainability.py          # SHAP utilities
│   ├── evaluation.py              # metrics + plots
│   └── scorecard.py               # inference function for new startups
│
├── models/
│   └── startup_success_xgb.joblib
│
├── reports/
│   └── figures/                   # exported PNG/HTML charts
│
└── tests/
    └── test_pipeline.py
```

---

## 7. Step-by-Step Build Guide

1. **Setup** — install libraries, set random seeds, mount Drive (optional) or load local CSV.
2. **Ingest** — load Kaggle startup dataset(s); validate schema; merge Crunchbase-style fields if multiple sources.
3. **Explore (EDA)** — class balance, missingness, correlation heatmap, category/geography distribution.
4. **Clean** — handle missing values, fix dtypes, remove duplicate/leaked rows, cap outliers.
5. **Engineer features** — funding velocity, round-to-round timing, founder/team signals, category one-hot/target encoding, geographic tiering.
6. **Split** — stratified train/val/test split (time-aware split preferred to avoid look-ahead leakage).
7. **Baseline model** — logistic regression for a sanity-check benchmark.
8. **Train primary models** — XGBoost + LightGBM with cross-validated hyperparameter search.
9. **Calibrate probabilities** — isotonic/Platt calibration so output scores are usable as true probabilities.
10. **Evaluate** — ROC-AUC, PR-AUC, F1, confusion matrix, calibration curve.
11. **Explain** — SHAP summary plot (global), SHAP waterfall (local, per-startup).
12. **Build scorecard function** — takes a new startup's raw features → cleaned → scored → explained.
13. **Package outputs** — save model, encoders, and feature list with `joblib` for reuse.
14. **Document & publish** — README, resume bullet, GitHub push.

---

## 8. Data Collection Pipeline

- **Primary path (default):** load a Kaggle-style CSV (`startup_data.csv`) with columns such as `funding_total_usd`, `funding_rounds`, `category`, `country`, `founded_at`, `first_funding_at`, `last_funding_at`, `status` (`operating`/`acquired`/`ipo`/`closed`).
- **Optional live path:** `CrunchbaseAPIClient` class wraps authenticated REST calls to pull organization + funding-round records, paginate results, respect rate limits (exponential backoff), and normalize the response into the same schema as the offline CSV so downstream code is source-agnostic.
- **Data source abstraction:** a single `load_startup_data(source="csv"|"api")` function returns a unified DataFrame regardless of origin, which is the key architectural decision that keeps the rest of the pipeline decoupled from *where* the data came from.

---

## 9. Data Cleaning & Feature Engineering

**Cleaning**
- Parse date columns; drop rows with impossible dates (e.g., `first_funding_at` before `founded_at`).
- Impute missing `funding_total_usd` conservatively (median by category), flag imputed rows with an indicator column.
- Standardize categorical text (lowercase, strip whitespace, collapse rare categories into `"other"`).
- Remove obvious duplicates (same company name + country + founding year).

**Engineered features (the VC-relevant signal set)**
- `company_age_years` — from `founded_at` to reference date (or last known status date).
- `time_to_first_funding_days` — speed of first raise (fast-raising = market conviction signal).
- `funding_velocity` — `funding_total_usd / max(company_age_years, 0.5)`.
- `avg_days_between_rounds` — momentum indicator.
- `funding_rounds` and `rounds_per_year`.
- `category_success_rate` — historical success rate within the company's category (target-encoded, computed only on training folds to avoid leakage).
- `is_top_tier_country` — binary flag for US/UK/major startup hubs.
- `log_funding_total` — log-transform to tame skew.
- `has_multiple_investors` (if investor-count field available).
- One-hot or frequency encoding for `category`/`country` (with rare-category bucketing).

**Leakage safeguards**
- Any encoding that uses the target (e.g., category success rate) is computed **inside the cross-validation fold**, never on the full dataset before splitting.
- Features that are only known *after* the outcome (e.g., "acquired_by") are explicitly excluded from `X`.

---

## 10. Core Models/Algorithms

| Model | Role |
|---|---|
| **Logistic Regression** | Interpretable baseline / sanity check |
| **XGBoost Classifier** | Primary production model — handles nonlinearity, missing values natively, strong tabular performance |
| **LightGBM Classifier** | Benchmark/ensemble candidate — faster training, good for categorical-heavy data |
| **CalibratedClassifierCV** (sklearn) | Wraps the best model to output well-calibrated probabilities |

Class imbalance (successes are usually the minority class) is handled with `scale_pos_weight` (XGBoost) / `is_unbalance` (LightGBM) and optionally SMOTE from `imbalanced-learn` for the baseline model.

Hyperparameter tuning uses stratified k-fold cross-validation with `RandomizedSearchCV` (fast, sufficient for a portfolio project; grid search noted as an upgrade).

---

## 11. Visualizations & Dashboard Components

1. **Class balance bar chart** — success vs. failure counts.
2. **Correlation heatmap** — numeric feature relationships.
3. **Funding distribution plots** — histogram/boxplot of `log_funding_total` by outcome.
4. **ROC curve** — with AUC annotation, model comparison overlay (LogReg vs XGB vs LightGBM).
5. **Precision-Recall curve** — critical given class imbalance; PR-AUC often more informative than ROC-AUC here.
6. **Confusion matrix heatmap** — at a chosen decision threshold.
7. **Calibration curve** — predicted probability vs. observed frequency.
8. **SHAP summary (beeswarm) plot** — global feature importance + directionality.
9. **SHAP waterfall plot** — local explanation for a single startup's score.
10. **Interactive Plotly scorecard** — bar of top contributing features for a given company, rendered as a lightweight "dashboard" cell in Colab.

---

## 12. Performance Metrics

- **ROC-AUC** — overall ranking ability.
- **PR-AUC (Average Precision)** — primary metric given imbalance; more VC-relevant since false negatives (missed unicorns) and false positives (wasted diligence time) both matter but asymmetrically.
- **F1-score at chosen operating threshold.**
- **Precision@K** — of the top-K scored companies, what fraction actually succeeded (mirrors "if we only diligence the top 20 deals this month, how good is our hit rate").
- **Brier score** — calibration quality of predicted probabilities.
- **Confusion matrix** at both default (0.5) and business-optimal thresholds.

---

## 13. Final Deliverables

- Cleaned, documented, reproducible Colab notebook (or `src/` modules) implementing the full pipeline.
- Trained, serialized model (`.joblib`) with a `predict_startup_success(new_company_dict)` inference function.
- Full suite of evaluation and SHAP explainability visualizations exported to `reports/figures/`.
- `README.md` with setup instructions, ethical/limitations disclaimer, and sample scorecard output.
- Optional: a simple Plotly/Streamlit-style scorecard cell that scores a hypothetical new startup end-to-end.

---

## 14. Resume Description

> **Startup Success Prediction Engine (Python, XGBoost, LightGBM, SHAP)** — Built an end-to-end ML pipeline that predicts startup success probability from Crunchbase/Kaggle funding data for VC-style deal screening; engineered 15+ investor-relevant features (funding velocity, category success rates, founding momentum), trained and calibrated gradient-boosted classifiers achieving [X]% ROC-AUC / [Y]% PR-AUC, and built SHAP-based explainability and ROC/PR/calibration dashboards to make model decisions auditable for investment-committee use.

*(Fill in [X]/[Y] with your actual results once trained on real data.)*

---

## 15. Potential Upgrades

- Swap the static CSV for a **live Crunchbase Pro API feed** with a scheduled ingestion job (Airflow/Prefect).
- Add **NLP features** from startup descriptions/pitch decks (embeddings via sentence-transformers) to capture qualitative signal.
- Add **founder-level features** (prior exits, LinkedIn tenure, team size growth) via AngelList/LinkedIn enrichment.
- Model **time-to-exit** as a survival analysis problem (Cox proportional hazards) in addition to binary classification.
- Deploy as a **FastAPI microservice** with a scorecard endpoint, wrapped in a lightweight Streamlit front end.
- Add **model monitoring** (population stability index / drift detection) if used on a rolling deal-flow feed.
- Ensemble XGBoost + LightGBM + a neural net (TabTransformer) via stacking for a small accuracy lift.
- Backtest the model against **actual historical VC decisions** to quantify screening lift versus human-only triage.
