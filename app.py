"""
Streamlit app for the Startup Success Prediction project — v2.

Adds, on top of the basic prediction:
  1. A live SHAP explanation (interactive bar chart) showing WHY the model
     produced this specific score for THIS specific company.
  2. An interactive comparison chart showing how this company's inputs
     compare to the average successful vs. unsuccessful company in the
     real training data.

This is fully self-contained — it does not depend on the Colab notebook.
It loads the saved model bundle (startup_success_model.joblib).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Startup Success Predictor", page_icon="📈", layout="centered")

TOP_TIER_COUNTRIES = {"USA", "GBR", "ISR", "SGP", "CAN", "DEU"}

FEATURE_DISPLAY_NAMES = {
    "company_age_years": "Company Age (years)",
    "time_to_first_funding_days": "Days to First Funding",
    "funding_velocity": "Funding Velocity ($/year)",
    "avg_days_between_rounds": "Avg. Days Between Rounds",
    "rounds_per_year": "Funding Rounds per Year",
    "is_top_tier_country": "Top-Tier Country",
    "log_funding_total": "Log(Total Funding)",
    "funding_rounds": "Number of Funding Rounds",
    "funding_total_usd_missing": "Funding Data Was Missing",
    "category_success_rate": "Category's Historical Success Rate",
}


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


def build_feature_row(company: dict, bundle: dict) -> pd.DataFrame:
    """Returns the fully engineered, correctly-ordered feature row (unscaled)."""
    reference_date = pd.Timestamp.today()
    df = engineer_features_single(company, reference_date)

    cat_map = bundle["category_success_map"]
    global_mean = float(np.mean(list(cat_map.values()))) if cat_map else 0.3
    df["category_success_rate"] = df["category"].map(cat_map).fillna(global_mean)

    return df[bundle["feature_columns"]]


def predict(company: dict, bundle: dict) -> dict:
    try:
        X_new = build_feature_row(company, bundle)

        if bundle["champion_model_name"] == "logistic_regression" and bundle["scaler"] is not None:
            X_for_model = bundle["scaler"].transform(X_new)
        else:
            X_for_model = X_new

        prob = float(bundle["model"].predict_proba(X_for_model)[:, 1][0])
        label = "High Potential" if prob >= 0.6 else ("Moderate Potential" if prob >= 0.35 else "Low Potential")
        return {"probability": prob, "label": label, "X_new": X_new, "error": None}
    except Exception as e:
        return {"probability": None, "label": None, "X_new": None, "error": str(e)}


# -----------------------------------------------------------------------------
# Live SHAP explanation for a single prediction
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shap_explainer(_bundle):
    """
    Builds a SHAP explainer once and caches it (expensive to rebuild every click).
    Tree models (XGBoost/LightGBM) use TreeExplainer with tree_path_dependent
    perturbation, which reads the tree structure directly and doesn't need a
    background dataset (and avoids a categorical-split compatibility error).
    Linear models (Logistic Regression) use LinearExplainer with the scaled
    background sample, since coefficients are on the scaled feature space.
    """
    try:
        import shap
        raw_model = _bundle["raw_champion_model"]

        if _bundle["champion_model_name"] == "logistic_regression":
            background = _bundle["scaler"].transform(_bundle["shap_background"])
            explainer = shap.LinearExplainer(raw_model, background)
        else:
            explainer = shap.TreeExplainer(raw_model, feature_perturbation="tree_path_dependent")
        return explainer
    except Exception:
        return None


def compute_shap_contributions(X_new: pd.DataFrame, bundle: dict):
    """Returns a Series of SHAP contribution values for this one company, or None on failure."""
    explainer = get_shap_explainer(bundle)
    if explainer is None:
        return None
    try:
        if bundle["champion_model_name"] == "logistic_regression":
            X_for_explainer = bundle["scaler"].transform(X_new)
            shap_values = explainer.shap_values(X_for_explainer)
        else:
            shap_values = explainer.shap_values(X_new)

        values = np.array(shap_values).reshape(-1)
        return pd.Series(values, index=bundle["feature_columns"])
    except Exception:
        return None


def render_shap_chart(shap_row: pd.Series, top_n: int = 8):
    ordered = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(top_n)
    ordered = ordered.iloc[::-1]  # largest at top when plotted horizontally
    labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in ordered.index]

    fig = go.Figure(go.Bar(
        x=ordered.values,
        y=labels,
        orientation="h",
        marker_color=["#2a9d8f" if v > 0 else "#e63946" for v in ordered.values],
        text=[f"{v:+.3f}" for v in ordered.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Why the model made this prediction",
        xaxis_title="Impact on predicted success (SHAP value)",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green bars pushed the prediction toward success. "
        "Red bars pushed it toward failure. Longer bars = bigger impact."
    )


# -----------------------------------------------------------------------------
# Interactive comparison chart: this company vs. real successful/unsuccessful startups
# -----------------------------------------------------------------------------
def render_comparison_chart(X_new: pd.DataFrame, bundle: dict):
    stats = bundle.get("feature_stats")
    if not stats:
        return

    compare_features = [
        "log_funding_total", "funding_velocity", "rounds_per_year",
        "category_success_rate", "is_top_tier_country",
    ]
    compare_features = [f for f in compare_features if f in bundle["feature_columns"]]

    this_company = X_new.iloc[0]
    successful_avg = stats["successful_mean"]
    unsuccessful_avg = stats["unsuccessful_mean"]

    labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in compare_features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="This Company",
        x=labels,
        y=[this_company[f] for f in compare_features],
        marker_color="#264653",
    ))
    fig.add_trace(go.Bar(
        name="Avg. Successful Startup",
        x=labels,
        y=[successful_avg[f] for f in compare_features],
        marker_color="#2a9d8f",
    ))
    fig.add_trace(go.Bar(
        name="Avg. Unsuccessful Startup",
        x=labels,
        y=[unsuccessful_avg[f] for f in compare_features],
        marker_color="#e63946",
    ))
    fig.update_layout(
        title="How this company compares to real startups in the training data",
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bars are on each feature's raw scale, so heights aren't directly comparable "
        "across features — focus on whether this company's bar sits closer to the "
        "successful or unsuccessful average for each one."
    )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Startup Success Predictor")
st.markdown(
    "A VC-style deal screening tool. Enter a startup's details below to get a "
    "predicted success probability — with an explanation of why — based on a "
    "model trained on real Crunchbase-derived funding data."
)
st.info(
    "⚠️ **Decision-support estimate only** — not a substitute for full "
    "investment diligence. In this dataset, \"success\" means the company was "
    "acquired; \"not successful\" means it shut down. Outcomes are noisy and "
    "shaped by many unobservable factors.",
    icon="⚠️",
)

if bundle is None:
    st.error(
        "Model file 'startup_success_model.joblib' not found in the app folder. "
        "Make sure it's uploaded alongside this app.py file."
    )
elif "raw_champion_model" not in bundle:
    st.warning(
        "This model file was saved by an older version of the pipeline and doesn't "
        "include the data needed for live explanations. Re-run Cell 15 in Colab with "
        "the updated code and re-upload the new .joblib file to unlock SHAP "
        "explanations and comparison charts."
    )

if bundle is not None:
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
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Predicted Success Probability", f"{prob:.1%}")
                color = "green" if label == "High Potential" else ("orange" if label == "Moderate Potential" else "red")
                st.markdown(f"**Screening Label:** :{color}[{label}]")
            with col_b:
                st.progress(min(max(prob, 0.0), 1.0))

            if "raw_champion_model" in bundle:
                st.markdown("### Explainability")
                shap_row = compute_shap_contributions(result["X_new"], bundle)
                if shap_row is not None:
                    render_shap_chart(shap_row)
                else:
                    st.caption("SHAP explanation unavailable for this model/input combination.")

                st.markdown("### How This Company Compares")
                render_comparison_chart(result["X_new"], bundle)

st.markdown("---")
st.caption(
    "Built as a portfolio project demonstrating an end-to-end ML pipeline for "
    "VC-style startup deal screening — see the full writeup and code on GitHub."
)
