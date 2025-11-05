# Streamlit Inference App

This app loads your trained models (CatBoost/XGBoost/LightGBM) and optional sklearn preprocessor, then serves batch CSV inference with an optional simple ensemble. No dataset is required at deploy time; users upload CSVs.

## Project Layout

- `streamlit_app.py` – main app
- `requirements.txt` – dependencies (installed on Streamlit Cloud)
- `.streamlit/config.toml` – dark theme
- `model_assets/` – place your artifacts here:
  - `CatBoost.joblib` / `XGBoost.joblib` / `LightGBM.joblib` (or any of the alternative names used in the notebook)
  - `preprocessor.joblib` (recommended)
  - optional: `feature_names.json`, `hybrid_metrics.csv`, `model_metrics_summary.csv`

## Local Preview (optional)

If you want to run locally without installing ML libs on your laptop, skip. Otherwise:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a public GitHub repo.
2. On Streamlit Cloud, create a new app pointing to that repo, file `streamlit_app.py`.
3. In the repo, put your artifacts inside `model_assets/`.

## Preparing Artifacts in Colab

Artifacts the app can use (any subset works):
- Models: `CatBoost.joblib`, `XGBoost.joblib`, `LightGBM.joblib` (supports common alt names)
- Preprocessor: `preprocessor.joblib`
- Optional: `feature_names.json`, `hybrid_metrics.csv`, `model_metrics_summary.csv`

The app auto-detects assets and displays metrics if the CSVs are present.

