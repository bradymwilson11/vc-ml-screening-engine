# Startup Success Prediction Using Machine Learning

An end-to-end, explainable ML pipeline that estimates a startup's probability
of success (acquisition/IPO vs. shutdown) from funding and company metadata —
built to mirror how VC firms do ML-assisted deal screening.
🔗 **[Try the live app](https://vc-ml-screening-engine.streamlit.app/)**
> **Disclaimer:** Startup outcomes are noisy and shaped by unobservable
> factors. This is a decision-**support** tool for screening/triage, not a
> substitute for full investment diligence.

## What's inside
- `PROJECT_DOCUMENTATION.md` — full project spec: architecture, data sources,
  feature engineering, models, metrics, and upgrade ideas.
- `startup_success_prediction.py` — complete pipeline, structured as
  Colab-ready cells (`# CELL N`). Runs standalone out of the box using a
  built-in synthetic dataset generator; swap in a real Kaggle/Crunchbase CSV
  for production results.
- `requirements.txt` — pinned dependency list.

## Quickstart
```bash
pip install -r requirements.txt
python startup_success_prediction.py
```
Or paste each `# CELL N` block into its own Google Colab cell and run top to
bottom.

To use real data, place a Kaggle "Startup Success Prediction" (or similar
Crunchbase-derived) CSV at `startup-success-prediction/data/raw/startup_data.csv`
with these columns: `name, category, country, founded_at, first_funding_at,
last_funding_at, funding_total_usd, funding_rounds, status`.

## Pipeline
Data ingestion → cleaning → leak-safe feature engineering → Logistic
Regression baseline → XGBoost / LightGBM (randomized hyperparameter search,
imbalance-aware) → probability calibration → ROC / PR / calibration
evaluation → SHAP global & local explainability → interactive scorecard →
serialized model artifact + `predict_startup_success()` inference function.

## Sample Results
*(trained on real Crunchbase-derived data, ~923 US startups, 2013 snapshot)*

| Metric | Score |
|---|---|
| ROC-AUC | 0.671 |
| PR-AUC | 0.753 |
| F1 | 0.703 |
| Brier Score | 0.213 |

### Evaluation Dashboard
![Evaluation Dashboard](reports/figures/final_evaluation_dashboard.png)

### Feature Correlations
![Correlation Heatmap](reports/figures/correlation_heatmap.png)

### Funding Distributions by Outcome
![Funding Distributions](reports/figures/funding_distributions.png)

### SHAP Global Feature Importance
![SHAP Summary](reports/figures/shap_summary_beeswarm.png)

### SHAP Local Explanation (single company)
![SHAP Waterfall](reports/figures/shap_waterfall_example.png)

## License
MIT — see `LICENSE`.
