"""
Streamlit app for the Startup Success Prediction project.

This is a self-contained app — it does NOT depend on the Colab notebook.
It loads the saved model bundle (startup_success_model.joblib) and
re-implements the same lightweight cleaning/feature-engineering logic
used during training, so a user can enter a hypothetical startup's
details and get a live prediction.

To run locally:   streamlit run app.py
To deploy:         push this file + requirements.txt + the .joblib model
                    to GitHub, then deploy via share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="📈",
    layout="centered",
)

TOP_TIER_COUNTRIES = {"USA", "GBR", "ISR", "SGP", "CAN", "DEU"}

REQUIRED_COLUMNS = [
    "name", "category", "country", "founded_at",
    "first_funding_at", "last_funding_at",
    "funding_total_usd", "funding_rounds", "status",
]


# -----------------------------------------------------------------------------
# Load the trained model bundle (cached so it only loads once per session)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    model_path = "startup_success_model.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


bundle = load_model_bundle()


# -----------------------------------------------------------------------------
# Feature engineering — mirrors the training pipeline exactly
# -----------------------------------------------------------------------------
def engineer_features_single(company: dict, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame([company])

    df["founded_at"] = pd.to_datetime(df["founded_at"])
    df["first_funding_at"] = pd.to_datetime(df["first_funding_at"])
    df["last_funding_at"] = pd.to_datetime(df["last_funding_at"])

    df["company_age_years"] = ((reference_date - df["founded_at"]).dt.days / 365.25).clip(lower=0.01)
    df["time_to_first_funding_days"] = ((df["first_funding_at"] - df["founded_at"]).dt.days).clip(lower=0)
    df["funding_velocity"] = df["funding_total_usd"] / df["company_age_years"]

    span_days = (df["last_funding_at"] - df["first_funding_at"]).dt.days.clip(lower=1)
    df["avg_days_between_rounds"] = span_days / df["funding_rounds"].clip(lower=1)
    df["rounds_per_year"] = df["funding_rounds"] / df["company_age_years"]
    df["is_top_tier_country"] = df["country"].str.upper().isin(TOP_TIER_COUNTRIES).astype(int)
    df["log_funding_total"] = np.log1p(df["funding_total_usd"])
    df["funding_total_usd_missing"] = 0

    return df


def predict(company: dict, bundle: dict) -> dict:
    try:
        reference_date = pd.Timestamp.today()
        df = engineer_features_single(company, reference_date)

        cat_map = bundle["category_success_map"]
        global_mean = float(np.mean(list(cat_map.values()))) if cat_map else 0.3
        df["category_success_rate"] = df["category"].map(cat_map).fillna(global_mean)

        feature_columns = bundle["feature_columns"]
        X_new = df[feature_columns]

        if bundle["champion_model_name"] == "logistic_regression" and bundle["scaler"] is not None:
            X_new = bundle["scaler"].transform(X_new)

        prob = float(bundle["model"].predict_proba(X_new)[:, 1][0])
        label = "High Potential" if prob >= 0.6 else ("Moderate Potential" if prob >= 0.35 else "Low Potential")

        return {"probability": prob, "label": label, "error": None}
    except Exception as e:
        return {"probability": None, "label": None, "error": str(e)}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Startup Success Predictor")
st.markdown(
    "A VC-style deal screening tool. Enter a startup's details below to get a "
    "predicted success probability, based on a model trained on real "
    "Crunchbase-derived funding data."
)
st.info(
    "⚠️ **Decision-support estimate only** — not a substitute for full "
    "investment diligence. Startup outcomes are noisy and shaped by many "
    "unobservable factors.",
    icon="⚠️",
)

if bundle is None:
    st.error(
        "Model file 'startup_success_model.joblib' not found in the app folder. "
        "Make sure it's uploaded alongside this app.py file."
    )
else:
    with st.form("startup_form"):
        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox(
                "Category",
                ["software", "web", "mobile", "enterprise", "advertising",
                 "gamesvideo", "ecommerce", "biotech", "consulting", "othercategory"],
            )
            country = st.selectbox("Country", ["USA", "GBR", "DEU", "CAN", "ISR", "SGP", "other"])
            founded_year = st.slider("Year Founded", 2005, 2026, 2022)

        with col2:
            funding_total_usd = st.number_input(
                "Total Funding Raised (USD)", min_value=0, value=2_000_000, step=100_000
            )
            funding_rounds = st.slider("Number of Funding Rounds", 1, 10, 2)
            months_to_first_funding = st.slider("Months to First Funding", 0, 36, 6)

        submitted = st.form_submit_button("Predict Success Probability", type="primary")

    if submitted:
        founded_at = pd.Timestamp(f"{founded_year}-01-01")
        first_funding_at = founded_at + pd.DateOffset(months=int(months_to_first_funding))
        last_funding_at = first_funding_at + pd.DateOffset(months=6)

        company = {
            "name": "User Input Company",
            "category": category,
            "country": country,
            "founded_at": founded_at.strftime("%Y-%m-%d"),
            "first_funding_at": first_funding_at.strftime("%Y-%m-%d"),
            "last_funding_at": last_funding_at.strftime("%Y-%m-%d"),
            "funding_total_usd": funding_total_usd,
            "funding_rounds": funding_rounds,
            "status": "operating",
        }

        result = predict(company, bundle)

        if result["error"]:
            st.error(f"Prediction failed: {result['error']}")
        else:
            prob = result["probability"]
            label = result["label"]

            st.markdown("---")
            st.metric("Predicted Success Probability", f"{prob:.1%}")

            color = "green" if label == "High Potential" else ("orange" if label == "Moderate Potential" else "red")
            st.markdown(f"**Screening Label:** :{color}[{label}]")

            st.progress(min(max(prob, 0.0), 1.0))

st.markdown("---")
st.caption(
    "Built as a portfolio project demonstrating an end-to-end ML pipeline for "
    "VC-style startup deal screening — see the full writeup and code on GitHub."
)
