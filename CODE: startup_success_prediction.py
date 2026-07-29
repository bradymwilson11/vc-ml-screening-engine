"""
==============================================================================
STARTUP SUCCESS PREDICTION USING MACHINE LEARNING
A VC-Grade Deal Screening & Scoring Engine
==============================================================================

Each "# %% [CELL N] ..." marker below corresponds to ONE Google Colab cell.
Copy each block into its own cell, in order, and run top to bottom.

Author  : Senior Quant/ML Engineering Template
Purpose : End-to-end, production-quality pipeline for predicting startup
          success probability from funding/company data, with explainable
          gradient-boosted models suitable for VC deal-screening workflows.

Disclaimer: This is a decision-SUPPORT tool. Startup outcomes are noisy,
survivorship-biased, and shaped by unobservable factors (founder quality,
market timing, luck). Treat model output as one input among many, never
as a standalone investment decision.
==============================================================================
"""

# %% [CELL 1] ENVIRONMENT SETUP & DEPENDENCY INSTALLATION
# -----------------------------------------------------------------------------
# Run this once at the start of the Colab session.
# -----------------------------------------------------------------------------
"""
!pip install -q xgboost lightgbm shap imbalanced-learn plotly scikit-learn --upgrade
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix, classification_report, brier_score_loss
)

import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import joblib

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("startup_ml")

# Folder layout (created locally; mirrors the GitHub repo structure)
BASE_DIR = "startup-success-prediction"
DIRS = {
    "raw": f"{BASE_DIR}/data/raw",
    "processed": f"{BASE_DIR}/data/processed",
    "models": f"{BASE_DIR}/models",
    "figures": f"{BASE_DIR}/reports/figures",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

logger.info(f"Environment ready. Directories initialized under '{BASE_DIR}/'")


# %% [CELL 2] DATA SOURCE ABSTRACTION — CRUNCHBASE API CLIENT + CSV LOADER
# -----------------------------------------------------------------------------
# A single entry point (load_startup_data) returns a unified schema whether
# data comes from an offline Kaggle-style CSV or a live Crunchbase API pull.
# This keeps the rest of the pipeline completely source-agnostic.
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "name", "category", "country", "founded_at",
    "first_funding_at", "last_funding_at",
    "funding_total_usd", "funding_rounds", "status",
]


class CrunchbaseAPIClient:
    """
    Thin wrapper around the Crunchbase REST API.

    Not required for the default (offline CSV) workflow — included so the
    pipeline can be pointed at live data by flipping `USE_LIVE_API=True`
    and supplying a valid API key, without touching any downstream code.
    """

    BASE_URL = "https://api.crunchbase.com/api/v4"

    def __init__(self, api_key: str, timeout: int = 15, max_retries: int = 3):
        if not api_key:
            raise ValueError("A Crunchbase API key is required for live mode.")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import requests
        params = {**params, "user_key": self.api_key}
        backoff = 1.5
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/{endpoint}", params=params, timeout=self.timeout
                )
                if resp.status_code == 429:  # rate limited
                    wait = backoff ** attempt
                    logger.warning("Rate limited by Crunchbase API. Waiting %.1fs...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.error(f"Crunchbase API request failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff ** attempt)
        raise RuntimeError("Crunchbase API request exhausted retries.")

    def fetch_organizations(self, limit: int = 500) -> pd.DataFrame:
        """Fetch organization + funding records and normalize to REQUIRED_COLUMNS."""
        raw = self._get("searches/organizations", {"limit": limit})
        records = raw.get("entities", [])
        rows = []
        for r in records:
            props = r.get("properties", {})
            rows.append({
                "name": props.get("name"),
                "category": props.get("category_groups_list", ["other"])[0]
                            if props.get("category_groups_list") else "other",
                "country": props.get("country_code", "unknown"),
                "founded_at": props.get("founded_on"),
                "first_funding_at": props.get("first_funding_on"),
                "last_funding_at": props.get("last_funding_on"),
                "funding_total_usd": props.get("funding_total", {}).get("value_usd", np.nan),
                "funding_rounds": props.get("num_funding_rounds", np.nan),
                "status": props.get("status", "unknown"),
            })
        return pd.DataFrame(rows)


def _generate_synthetic_dataset(n: int = 4000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a realistic, schema-correct synthetic dataset so the ENTIRE
    pipeline is runnable end-to-end with zero external downloads.

    Replace this with a real Kaggle/Crunchbase CSV for production use:
        df = pd.read_csv(f"{DIRS['raw']}/startup_data.csv")

    The synthetic generator deliberately encodes realistic, non-trivial
    relationships (category effects, funding-speed effects, noise) so that
    the downstream feature engineering / modeling code has real signal
    to find, mirroring how a genuine Crunchbase export behaves.
    """
    rng = np.random.default_rng(seed)

    categories = ["fintech", "healthtech", "saas", "biotech", "consumer",
                  "ecommerce", "ai_ml", "cleantech", "edtech", "other"]
    category_base_rate = {
        "fintech": 0.34, "healthtech": 0.30, "saas": 0.38, "biotech": 0.22,
        "consumer": 0.24, "ecommerce": 0.26, "ai_ml": 0.40, "cleantech": 0.27,
        "edtech": 0.25, "other": 0.20,
    }
    countries = ["USA", "GBR", "DEU", "IND", "CHN", "FRA", "ISR", "CAN", "SGP", "other"]
    top_tier = {"USA", "GBR", "ISR", "SGP"}

    rows = []
    start_year_pool = rng.integers(2005, 2021, size=n)

    for i in range(n):
        cat = rng.choice(categories)
        country = rng.choice(countries, p=[0.35, 0.10, 0.08, 0.10, 0.07,
                                            0.07, 0.05, 0.06, 0.07, 0.05])
        founded_year = int(start_year_pool[i])
        founded_at = pd.Timestamp(f"{founded_year}-{rng.integers(1,13):02d}-01")

        days_to_first_funding = int(rng.gamma(shape=2.0, scale=180))
        first_funding_at = founded_at + pd.Timedelta(days=days_to_first_funding)

        funding_rounds = int(rng.poisson(2.2) + 1)
        avg_gap_days = int(rng.gamma(shape=2.5, scale=200))
        last_funding_at = first_funding_at + pd.Timedelta(
            days=avg_gap_days * max(funding_rounds - 1, 0)
        )

        base_funding = rng.lognormal(mean=14.5, sigma=1.3)  # ~ USD hundreds of thousands to tens of millions
        funding_total_usd = float(base_funding * (1 + 0.15 * funding_rounds))

        # --- Latent success probability (ground truth generative process) ---
        p = category_base_rate[cat]
        p += 0.08 if country in top_tier else -0.03
        p += 0.05 if days_to_first_funding < 200 else -0.03
        p += 0.04 * min(funding_rounds, 5) / 5
        p += 0.06 if funding_total_usd > np.exp(15.5) else 0.0
        p -= 0.05 if (2020 <= founded_year <= 2021) else 0.0  # COVID cohort headwind
        # Nonlinear interaction: fast-funded + top-tier country compounds strongly
        # (deliberately nonlinear so tree-based models have a real edge over
        # linear baselines, mirroring realistic VC signal interactions).
        if days_to_first_funding < 150 and country in top_tier and funding_rounds >= 3:
            p += 0.12
        p = float(np.clip(p + rng.normal(0, 0.07), 0.02, 0.95))

        outcome = rng.random() < p
        if outcome:
            status = rng.choice(["acquired", "ipo"], p=[0.85, 0.15])
        else:
            status = rng.choice(["closed", "operating"], p=[0.55, 0.45])
            # "operating" but not yet successful is treated as censored/negative
            # for this simplified binary framing (documented assumption).

        rows.append({
            "name": f"Startup_{i:05d}",
            "category": cat,
            "country": country,
            "founded_at": founded_at,
            "first_funding_at": first_funding_at,
            "last_funding_at": last_funding_at,
            "funding_total_usd": round(funding_total_usd, 2),
            "funding_rounds": funding_rounds,
            "status": status,
        })

    df = pd.DataFrame(rows)

    # Inject realistic missingness
    for col in ["funding_total_usd", "last_funding_at", "country"]:
        mask = rng.random(len(df)) < 0.04
        df.loc[mask, col] = np.nan

    return df


def load_startup_data(source: str = "csv", csv_path: Optional[str] = None,
                       api_client: Optional[CrunchbaseAPIClient] = None) -> pd.DataFrame:
    """
    Unified data loader. `source` in {"csv", "api", "synthetic"}.

    Returns a DataFrame with (at minimum) REQUIRED_COLUMNS, regardless of
    origin, so downstream code never needs to know where the data came from.
    """
    try:
        if source == "synthetic":
            logger.info("Loading SYNTHETIC demo dataset (no external download required).")
            df = _generate_synthetic_dataset()

        elif source == "csv":
            path = csv_path or f"{DIRS['raw']}/startup_data.csv"
            if not os.path.exists(path):
                logger.warning(
                    f"No CSV found at '{path}'. Falling back to synthetic demo data. "
                    f"Place a Kaggle 'Startup Success Prediction' CSV there for real results."
                )
                df = _generate_synthetic_dataset()
            else:
                df = pd.read_csv(path)
                logger.info(f"Loaded CSV with shape {df.shape} from '{path}'.")

        elif source == "api":
            if api_client is None:
                raise ValueError("api_client must be provided when source='api'.")
            df = api_client.fetch_organizations()
            logger.info(f"Fetched {len(df)} organizations from Crunchbase API.")

        else:
            raise ValueError(f"Unknown source '{source}'. Use 'csv', 'api', or 'synthetic'.")

        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Loaded data is missing required columns: {missing_cols}")

        return df

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise


# Load data (defaults to synthetic so this notebook runs standalone;
# swap source="csv" once you've placed a real Kaggle CSV in data/raw/)
raw_df = load_startup_data(source="csv")
raw_df.to_csv(f"{DIRS['raw']}/startup_data_loaded_snapshot.csv", index=False)
print(f"Raw data shape: {raw_df.shape}")
raw_df.head()


# %% [CELL 3] EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Prints quick-look diagnostics: shape, dtypes, missingness, class balance."""
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print("\nMissing values per column (%):")
    print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

    print("\nStatus / outcome distribution:")
    print(df["status"].value_counts(dropna=False))


run_eda(raw_df)

# --- Class balance visualization ---
success_statuses = {"acquired", "ipo"}
temp_target = raw_df["status"].isin(success_statuses).astype(int)

fig, ax = plt.subplots(figsize=(7, 5))
counts = temp_target.value_counts().sort_index()
bars = ax.bar(["Not Successful (0)", "Successful (1)"], counts.values,
              color=["#e63946", "#2a9d8f"])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val:,}",
            ha="center", fontweight="bold")
ax.set_title("Class Balance: Startup Success vs. Non-Success")
ax.set_ylabel("Number of Companies")
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/class_balance.png", dpi=150)
plt.show()

print(f"\nBase success rate: {temp_target.mean():.2%}  "
      f"(informs class-imbalance handling in Cell 6)")


# %% [CELL 4] DATA CLEANING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
TOP_TIER_COUNTRIES = {"USA", "GBR", "ISR", "SGP", "CAN", "DEU"}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Type coercion, date sanity checks, missing-value handling, dedup."""
    df = df.copy()

    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows with impossible date ordering (data-quality guard)
    bad_dates = df["first_funding_at"] < df["founded_at"]
    if bad_dates.any():
        logger.warning(f"Dropping {bad_dates.sum()} rows with first_funding_at < founded_at.")
        df = df.loc[~bad_dates].copy()

    # Deduplicate on name + country + founding year (proxy for entity match)
    before = len(df)
    df = df.drop_duplicates(subset=["name", "country", "founded_at"])
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} duplicate rows.")

    # Standardize categorical text
    for col in ["category", "country", "status"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["country"] = df["country"].replace("nan", "unknown")

    # Median-impute funding_total_usd by category, with a missingness flag
    df["funding_total_usd_missing"] = df["funding_total_usd"].isna().astype(int)
    df["funding_total_usd"] = df.groupby("category")["funding_total_usd"] \
        .transform(lambda s: s.fillna(s.median()))
    df["funding_total_usd"] = df["funding_total_usd"].fillna(df["funding_total_usd"].median())

    # Fill missing last_funding_at with first_funding_at (conservative: no further rounds)
    df["last_funding_at"] = df["last_funding_at"].fillna(df["first_funding_at"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Builds the VC-relevant feature set described in the design doc.

    IMPORTANT: `category_success_rate` (target encoding) is intentionally
    NOT computed here — it is fit only on the TRAINING fold inside the
    modeling cell to prevent target leakage.
    """
    df = df.copy()
    reference_date = reference_date or df["last_funding_at"].max()

    df["company_age_years"] = (
        (reference_date - df["founded_at"]).dt.days / 365.25
    ).clip(lower=0.01)

    df["time_to_first_funding_days"] = (
        (df["first_funding_at"] - df["founded_at"]).dt.days
    ).clip(lower=0)

    df["funding_velocity"] = df["funding_total_usd"] / df["company_age_years"]

    span_days = (df["last_funding_at"] - df["first_funding_at"]).dt.days.clip(lower=1)
    df["avg_days_between_rounds"] = span_days / df["funding_rounds"].clip(lower=1)

    df["rounds_per_year"] = df["funding_rounds"] / df["company_age_years"]

    df["is_top_tier_country"] = df["country"].str.upper().isin(TOP_TIER_COUNTRIES).astype(int)

    df["log_funding_total"] = np.log1p(df["funding_total_usd"])

    # Binary target: 1 = acquired/ipo, 0 = closed/operating (documented simplification)
    df["target"] = df["status"].isin(["acquired", "ipo"]).astype(int)

    return df


cleaned_df = clean_data(raw_df)
featured_df = engineer_features(cleaned_df)
featured_df.to_csv(f"{DIRS['processed']}/startup_data_featured.csv", index=False)

print(f"Cleaned + engineered dataset shape: {featured_df.shape}")
featured_df[[
    "company_age_years", "time_to_first_funding_days", "funding_velocity",
    "avg_days_between_rounds", "rounds_per_year", "is_top_tier_country",
    "log_funding_total", "target",
]].describe().round(2)


# %% [CELL 5] CORRELATION HEATMAP & FUNDING DISTRIBUTION VISUALS
# -----------------------------------------------------------------------------
numeric_cols = [
    "company_age_years", "time_to_first_funding_days", "funding_velocity",
    "avg_days_between_rounds", "rounds_per_year", "is_top_tier_country",
    "log_funding_total", "funding_rounds", "target",
]

corr = featured_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/correlation_heatmap.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=featured_df, x="target", y="log_funding_total", ax=axes[0],
            palette=["#e63946", "#2a9d8f"])
axes[0].set_title("Log(Total Funding) by Outcome")
axes[0].set_xticklabels(["Not Successful", "Successful"])

sns.histplot(data=featured_df, x="funding_velocity", hue="target",
             bins=40, log_scale=(True, False), ax=axes[1],
             palette=["#e63946", "#2a9d8f"], element="step")
axes[1].set_title("Funding Velocity Distribution by Outcome")
axes[1].set_xlabel("Funding Velocity (USD / year, log scale)")
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/funding_distributions.png", dpi=150)
plt.show()


# %% [CELL 6] TRAIN / VALIDATION / TEST SPLIT + LEAK-SAFE CATEGORY ENCODING
# -----------------------------------------------------------------------------
FEATURE_COLUMNS_NUMERIC = [
    "company_age_years", "time_to_first_funding_days", "funding_velocity",
    "avg_days_between_rounds", "rounds_per_year", "is_top_tier_country",
    "log_funding_total", "funding_rounds", "funding_total_usd_missing",
]
CATEGORY_COL = "category"
TARGET_COL = "target"


def add_leak_safe_category_encoding(
    train_df: pd.DataFrame, other_df: pd.DataFrame, smoothing: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target-encodes `category` using ONLY the training split, with additive
    smoothing toward the global mean to avoid overfitting on rare categories.
    Applied to `other_df` (val/test) via a learned mapping — never refit on it.
    """
    global_mean = train_df[TARGET_COL].mean()
    stats = train_df.groupby(CATEGORY_COL)[TARGET_COL].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    mapping = smoothed.to_dict()

    train_df = train_df.copy()
    other_df = other_df.copy()
    train_df["category_success_rate"] = train_df[CATEGORY_COL].map(mapping).fillna(global_mean)
    other_df["category_success_rate"] = other_df[CATEGORY_COL].map(mapping).fillna(global_mean)
    return train_df, other_df


try:
    X_full = featured_df.copy()
    train_val_df, test_df = train_test_split(
        X_full, test_size=0.15, stratify=X_full[TARGET_COL], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1765,  # ~15% of full dataset
        stratify=train_val_df[TARGET_COL], random_state=RANDOM_SEED
    )

    train_df, val_df = add_leak_safe_category_encoding(train_df, val_df)
    train_df, test_df = add_leak_safe_category_encoding(train_df, test_df)

    FEATURE_COLUMNS = FEATURE_COLUMNS_NUMERIC + ["category_success_rate"]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL]

    logger.info(
        f"Split sizes -> train: {len(X_train)} | val: {len(X_val)} | "
        f"test: {len(X_test)} | success rate (train): {y_train.mean()*100:.2f}%"
    )
except Exception as e:
    logger.error(f"Data splitting failed: {e}")
    raise


# %% [CELL 7] BASELINE MODEL — LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

baseline_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
)
baseline_model.fit(X_train_scaled, y_train)

baseline_val_probs = baseline_model.predict_proba(X_val_scaled)[:, 1]
baseline_auc = roc_auc_score(y_val, baseline_val_probs)
baseline_pr_auc = average_precision_score(y_val, baseline_val_probs)

print(f"[Baseline: Logistic Regression]  Val ROC-AUC: {baseline_auc:.4f} | "
      f"Val PR-AUC: {baseline_pr_auc:.4f}")


# %% [CELL 8] PRIMARY MODEL TRAINING — XGBOOST WITH RANDOMIZED HYPERPARAMETER SEARCH
# -----------------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_param_dist = {
    "n_estimators": [200, 300, 400, 600],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 2, 5],
}

xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

try:
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=25,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    logger.info(f"Best XGBoost params: {xgb_search.best_params_}")
except Exception as e:
    logger.error(f"XGBoost hyperparameter search failed, falling back to defaults: {e}")
    xgb_model = xgb_base.fit(X_train, y_train)

xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
print(f"[XGBoost]  Val ROC-AUC: {roc_auc_score(y_val, xgb_val_probs):.4f} | "
      f"Val PR-AUC: {average_precision_score(y_val, xgb_val_probs):.4f}")


# %% [CELL 9] BENCHMARK MODEL — LIGHTGBM
# -----------------------------------------------------------------------------
lgb_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    is_unbalance=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=-1,
)
lgb_model.fit(X_train, y_train)
lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]

print(f"[LightGBM] Val ROC-AUC: {roc_auc_score(y_val, lgb_val_probs):.4f} | "
      f"Val PR-AUC: {average_precision_score(y_val, lgb_val_probs):.4f}")

# --- Select champion model based on validation PR-AUC (imbalance-aware metric) ---
model_scores = {
    "logistic_regression": average_precision_score(y_val, baseline_val_probs),
    "xgboost": average_precision_score(y_val, xgb_val_probs),
    "lightgbm": average_precision_score(y_val, lgb_val_probs),
}
champion_name = max(model_scores, key=model_scores.get)
champion_raw_model = {"logistic_regression": baseline_model,
                      "xgboost": xgb_model, "lightgbm": lgb_model}[champion_name]
print(f"\nChampion model (by val PR-AUC): {champion_name} -> {model_scores}")


# %% [CELL 10] PROBABILITY CALIBRATION
# -----------------------------------------------------------------------------
# Wrap the champion model so predicted probabilities are trustworthy enough
# to present directly to an investment committee (e.g., "62% predicted
# success probability" should actually mean ~62% historically).
# -----------------------------------------------------------------------------
if champion_name == "logistic_regression":
    X_train_for_calibration, X_val_for_calibration = X_train_scaled, X_val_scaled
else:
    X_train_for_calibration, X_val_for_calibration = X_train, X_val

try:
    # scikit-learn >= 1.6 replaced cv="prefit" with the FrozenEstimator wrapper.
    from sklearn.frozen import FrozenEstimator
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(champion_raw_model), method="isotonic"
    )
except ImportError:
    # scikit-learn < 1.6 fallback
    calibrated_model = CalibratedClassifierCV(
        champion_raw_model, method="isotonic", cv="prefit"
    )
calibrated_model.fit(X_val_for_calibration, y_val)

if champion_name == "logistic_regression":
    X_test_for_eval = scaler.transform(X_test)
else:
    X_test_for_eval = X_test

test_probs = calibrated_model.predict_proba(X_test_for_eval)[:, 1]
print("Calibration complete. Ready for final evaluation on held-out test set.")


# %% [CELL 11] FINAL EVALUATION — ROC, PRECISION-RECALL, CONFUSION MATRIX, CALIBRATION
# -----------------------------------------------------------------------------
def evaluate_and_plot(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes the full VC-relevant metric suite and renders diagnostic plots."""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0, 0].plot(fpr, tpr, color="#2a9d8f", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[0, 1].plot(recall, precision, color="#e76f51", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[0, 1].axhline(y_true.mean(), ls="--", color="gray", lw=1, label="Baseline (prevalence)")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend(loc="upper right")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
    axes[1, 0].set_title(f"Confusion Matrix (threshold={threshold})")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    axes[1, 1].plot(mean_pred, frac_pos, "o-", color="#264653", label="Model")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1, 1].set_xlabel("Mean Predicted Probability")
    axes[1, 1].set_ylabel("Observed Success Rate")
    axes[1, 1].set_title(f"Calibration Curve (Brier = {brier:.4f})")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{DIRS['figures']}/final_evaluation_dashboard.png", dpi=150)
    plt.show()

    print("\n" + classification_report(y_true, y_pred, target_names=["Not Successful", "Successful"]))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "brier": brier}


test_metrics = evaluate_and_plot(y_test.values, test_probs)
print(f"\nFINAL TEST METRICS: {json.dumps(test_metrics, indent=2)}")

# --- Precision@K: hit rate if only the top-K scored deals get diligenced ---
def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].mean())

for k in [10, 25, 50]:
    if k <= len(y_test):
        print(f"Precision@{k}: {precision_at_k(y_test.values, test_probs, k):.2%}")


# %% [CELL 12] EXPLAINABILITY — SHAP GLOBAL & LOCAL PLOTS
# -----------------------------------------------------------------------------
if SHAP_AVAILABLE and champion_name in {"xgboost", "lightgbm"}:
    try:
        explainer = shap.TreeExplainer(champion_raw_model)
        shap_values = explainer.shap_values(X_test)

        # Global feature importance (beeswarm)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Summary — Global Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{DIRS['figures']}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Local explanation for a single company (highest-scored test example)
        top_idx = int(np.argmax(test_probs))
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[top_idx],
                base_values=explainer.expected_value,
                data=X_test.iloc[top_idx],
                feature_names=X_test.columns.tolist(),
            ),
            show=False,
        )
        plt.title(f"SHAP Waterfall — Top-Scored Test Company (idx={top_idx})")
        plt.tight_layout()
        plt.savefig(f"{DIRS['figures']}/shap_waterfall_example.png", dpi=150, bbox_inches="tight")
        plt.show()

    except Exception as e:
        logger.error(f"SHAP explainability failed: {e}")
else:
    logger.warning("SHAP not available or champion model is not tree-based; skipping SHAP plots.")
    # Fallback: native feature importance
    if hasattr(champion_raw_model, "feature_importances_"):
        importances = pd.Series(
            champion_raw_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        importances.plot(kind="barh", figsize=(8, 6), color="#2a9d8f", title="Feature Importance")
        plt.tight_layout()
        plt.show()


# %% [CELL 13] INTERACTIVE PLOTLY SCORECARD (LIGHTWEIGHT DASHBOARD)
# -----------------------------------------------------------------------------
def build_scorecard_figure(feature_names: List[str], shap_row: np.ndarray,
                            company_name: str, predicted_prob: float) -> go.Figure:
    """Renders an interactive bar chart of the top SHAP contributors for one company."""
    order = np.argsort(np.abs(shap_row))[::-1][:10]
    fig = go.Figure(go.Bar(
        x=shap_row[order],
        y=[feature_names[i] for i in order],
        orientation="h",
        marker_color=["#2a9d8f" if v > 0 else "#e63946" for v in shap_row[order]],
    ))
    fig.update_layout(
        title=f"Score Drivers — {company_name} (Predicted Success Prob: {predicted_prob:.1%})",
        xaxis_title="SHAP Contribution (impact on predicted success)",
        yaxis_title="Feature",
        template="plotly_white",
        height=450,
    )
    return fig


if SHAP_AVAILABLE and champion_name in {"xgboost", "lightgbm"}:
    example_idx = int(np.argmax(test_probs))
    example_company = test_df.iloc[example_idx]["name"]
    fig = build_scorecard_figure(
        FEATURE_COLUMNS, shap_values[example_idx], example_company, test_probs[example_idx]
    )
    fig.show()


# %% [CELL 14] PRODUCTION INFERENCE FUNCTION — SCORE A NEW STARTUP
# -----------------------------------------------------------------------------
def predict_startup_success(
    new_company: Dict[str, Any],
    trained_model=calibrated_model,
    category_success_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Scores a single new/unseen startup end-to-end: raw dict -> cleaned ->
    feature-engineered -> scored -> (optionally) SHAP-explained.

    Parameters
    ----------
    new_company : dict with keys matching REQUIRED_COLUMNS (dates as strings
        'YYYY-MM-DD' or pd.Timestamp; funding_total_usd as float).

    Returns
    -------
    dict with `predicted_success_probability` and a human-readable label.
    """
    try:
        df_new = pd.DataFrame([new_company])
        df_new = clean_data(pd.concat([raw_df.iloc[:0], df_new], ignore_index=True))
        df_new = engineer_features(df_new, reference_date=pd.Timestamp.today())

        cat_map = category_success_map or (
            train_df.groupby(CATEGORY_COL)["category_success_rate"].mean().to_dict()
        )
        global_mean = train_df[TARGET_COL].mean()
        df_new["category_success_rate"] = df_new[CATEGORY_COL].map(cat_map).fillna(global_mean)

        X_new = df_new[FEATURE_COLUMNS]
        if champion_name == "logistic_regression":
            X_new = scaler.transform(X_new)

        prob = float(trained_model.predict_proba(X_new)[:, 1][0])
        label = "High Potential" if prob >= 0.6 else ("Moderate Potential" if prob >= 0.35 else "Low Potential")

        return {
            "predicted_success_probability": round(prob, 4),
            "screening_label": label,
            "note": "Decision-support estimate only — not a substitute for full diligence.",
        }
    except Exception as e:
        logger.error(f"Inference failed for new company: {e}")
        return {"error": str(e)}


# --- Example usage ---
example_new_startup = {
    "name": "Novel AI Labs",
    "category": "ai_ml",
    "country": "USA",
    "founded_at": "2023-01-15",
    "first_funding_at": "2023-05-01",
    "last_funding_at": "2024-11-01",
    "funding_total_usd": 8_500_000,
    "funding_rounds": 2,
    "status": "operating",  # unknown at prediction time; ignored by feature engineering
}
result = predict_startup_success(example_new_startup)
print(json.dumps(result, indent=2))


# %% [CELL 15] PERSIST MODEL ARTIFACTS FOR DEPLOYMENT / GITHUB
# -----------------------------------------------------------------------------
artifact_bundle = {
    "model": calibrated_model,
    "champion_model_name": champion_name,
    "feature_columns": FEATURE_COLUMNS,
    "scaler": scaler if champion_name == "logistic_regression" else None,
    "category_success_map": train_df.groupby(CATEGORY_COL)["category_success_rate"].mean().to_dict(),
    "test_metrics": test_metrics,
    "trained_at": pd.Timestamp.today().isoformat(),
}

model_path = f"{DIRS['models']}/startup_success_model.joblib"
joblib.dump(artifact_bundle, model_path)
logger.info(f"Model artifact bundle saved to '{model_path}'.")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Champion model      : {champion_name}")
print(f"Test ROC-AUC         : {test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC          : {test_metrics['pr_auc']:.4f}")
print(f"Test F1              : {test_metrics['f1']:.4f}")
print(f"Test Brier Score     : {test_metrics['brier']:.4f}")
print(f"Artifacts saved under: {BASE_DIR}/")