"""
Streamlit App for Heart Attack Risk Prediction
Based on ensemble model (XGBoost + CatBoost + LightGBM)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Predicting Heart Attack Risk: An Ensemble Modeling Approach",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Modern Design System */
    :root {
        --primary: #3B82F6;
        --primary-dark: #2563EB;
        --secondary: #8B5CF6;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-card: rgba(30, 41, 59, 0.4);
        --bg-card-hover: rgba(30, 41, 59, 0.6);
        --border: rgba(148, 163, 184, 0.1);
        --border-strong: rgba(148, 163, 184, 0.2);
        --text-primary: #F1F5F9;
        --text-secondary: #CBD5E1;
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.2);
        --radius: 16px;
        --radius-sm: 12px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header with gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        line-height: 1.2;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Modern cards */
    .info-card {
        padding: 1.5rem;
        border-radius: var(--radius-sm);
        background: var(--bg-card);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .info-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-strong);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: var(--bg-card);
        padding: 1rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }
    
    div[data-testid="metric-container"]:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-strong);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: var(--radius-sm);
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stRadio > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        color: var(--text-primary);
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-strong);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.5rem 1.5rem;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error states */
    .risk-high {
        color: var(--danger);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .risk-low {
        color: var(--success);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    h2 {
        font-size: 1.875rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Info boxes */
    .stMarkdown p {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Selectbox */
    .stSelectbox > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "model_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def find_first_existing(names):
    for n in names:
        p = os.path.join(ASSETS_DIR, n)
        if os.path.exists(p):
            return p
    return None

def load_performance_metrics():
    """Load model and ensemble metrics from available CSVs.
    Returns:
        metrics_rows: list of dicts with keys: model, accuracy, recall, f1, roc_auc
        hybrid_rows: list of dicts with keys: version, accuracy, recall, f1, roc_auc
    """
    metrics_rows = []
    hybrid_rows = []

    # Candidate files in order of preference
    candidate_model_metrics = [
        os.path.join(BASE_DIR, "content", "models", "model_metrics_best.csv"),
        os.path.join(BASE_DIR, "model_assets", "model_metrics.csv"),
        os.path.join(BASE_DIR, "content", "models", "model_metrics.csv"),
    ]
    candidate_hybrid_metrics = [
        os.path.join(BASE_DIR, "content", "models", "hybrid_metrics_best.csv"),
        os.path.join(BASE_DIR, "model_assets", "hybrid_metrics.csv"),
        os.path.join(BASE_DIR, "content", "models", "hybrid_metrics.csv"),
    ]

    # Load model metrics - prioritize optimized metrics
    candidate_model_metrics_priority = [
        os.path.join(BASE_DIR, "content", "models", "model_metrics_optimized.csv"),
        os.path.join(BASE_DIR, "model_assets", "model_metrics_optimized.csv"),
        os.path.join(BASE_DIR, "content", "models", "model_metrics_best.csv"),
    ] + candidate_model_metrics
    
    for fp in candidate_model_metrics_priority:
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
            except Exception:
                try:
                    df = pd.read_csv(fp, index_col=0)
                except Exception:
                    continue
            cols = {c.lower(): c for c in df.columns}
            # Normalize rows
            for idx, row in df.iterrows():
                mr = {}
                mr["model"] = str(row.get(cols.get("model"), idx))
                for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    v = row.get(cols.get(k)) if cols.get(k) in row else None
                    try:
                        mr[k] = float(v)
                    except Exception:
                        mr[k] = None
                metrics_rows.append(mr)
            # Prefer first successful file then break
            if metrics_rows:
                break

    # Load hybrid/ensemble metrics - prioritize optimized metrics
    candidate_hybrid_metrics_priority = [
        os.path.join(BASE_DIR, "content", "models", "hybrid_metrics_best.csv"),
        os.path.join(BASE_DIR, "model_assets", "hybrid_metrics.csv"),
        os.path.join(BASE_DIR, "content", "models", "hybrid_metrics.csv"),
    ] + candidate_hybrid_metrics
    
    for fp in candidate_hybrid_metrics_priority:
        if os.path.exists(fp):
            try:
                dfh = pd.read_csv(fp)
            except Exception:
                try:
                    dfh = pd.read_csv(fp, index_col=0)
                except Exception:
                    continue
            cols = {c.lower(): c for c in dfh.columns}
            for idx, row in dfh.iterrows():
                hr = {}
                hr["version"] = str(row.get(cols.get("version", "version"), idx))
                for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    v = row.get(cols.get(k)) if cols.get(k) in row else None
                    try:
                        hr[k] = float(v)
                    except Exception:
                        hr[k] = None
                hybrid_rows.append(hr)
            if hybrid_rows:
                break

    return metrics_rows, hybrid_rows

def get_algo_metrics(metrics_rows, algo_name: str):
    """Pick metrics for a given algo ('XGBoost', 'CatBoost', 'LightGBM').
    Uses heuristics to match model names in CSV.
    Returns best (highest accuracy) matching row or None.
    """
    if not metrics_rows:
        return None
    name_hints = {
        "XGBoost": ["XGB", "XGBoost", "xgb"],
        "CatBoost": ["CAT", "CatBoost", "cat"],
        "LightGBM": ["LGBM", "LightGBM", "lgb"],
        "LogReg": ["LogReg", "logreg", "logistic"],
        "RandomForest": ["RF", "RandomForest", "random forest"],
    }
    hints = name_hints.get(algo_name, [algo_name])
    best = None
    for row in metrics_rows:
        label = str(row.get("model", "")).upper()
        if any(hint.upper() in label for hint in hints):
            if best is None:
                best = row
            else:
                acc_best = best.get("accuracy") or -1
                acc_new = row.get("accuracy") or -1
                if acc_new > acc_best:
                    best = row
    return best

def get_ensemble_metrics(hybrid_rows, metrics_rows=None):
    """Return the preferred ensemble metrics row.
    Preference: 'Ensemble_optimized' from model_metrics -> 'Ensemble_best@0.5' -> 'Ensemble@0.5' -> first Ensemble row.
    """
    # First, try to get Ensemble_optimized from model_metrics (most recent optimized)
    if metrics_rows:
        for row in metrics_rows:
            model_name = str(row.get("model", "")).upper()
            if "ENSEMBLE" in model_name and "OPTIMIZED" in model_name:
                return row
    
    # Then check hybrid_rows
    if not hybrid_rows:
        return None
    # Normalize
    rows = list(hybrid_rows)
    # First preference: Ensemble_best@0.5
    for r in rows:
        ver = str(r.get("version", ""))
        if ver.lower() == "ensemble_best@0.5" or ("ensemble_best" in ver.lower() and "@0.5" in ver.lower()):
            return r
    # Second preference: Ensemble@0.5
    for r in rows:
        ver = str(r.get("version", ""))
        if ver.lower() == "ensemble@0.5" or ("ensemble" in ver.lower() and "@0.5" in ver.lower()):
            return r
    # Any ensemble row
    for r in rows:
        ver = str(r.get("version", ""))
        if "ensemble" in ver.lower():
            return r
    return None

@st.cache_resource
def load_models():
    """Load models and preprocessor (cached for performance). Robust per-model loading."""
    preprocessor = None
    try:
        preproc_path = find_first_existing(["preprocessor.joblib"])
        if preproc_path:
            preprocessor = joblib.load(preproc_path)
    except Exception as e:
        st.warning(f"Preprocessor load skipped: {e}")

    models = {}
    # Resolve paths - prioritize optimized models
    xgb_path = find_first_existing([
        "XGBoost_optimized.joblib", "XGB_spw.joblib", "XGBoost.joblib", "xgb_model.joblib", "xgb_full.joblib", "XGBoost_best_5cv.joblib"
    ])
    cat_path = find_first_existing([
        "CatBoost_optimized.joblib", "CAT_cw.joblib", "CatBoost.joblib", "catboost.joblib", "cat_model.joblib", "cat_full.joblib", "CatBoost_best_5cv.joblib"
    ])
    lgb_path = find_first_existing([
        "LightGBM_optimized.joblib", "LGBM_cw.joblib", "LightGBM.joblib", "lgb_model.joblib", "LightGBM_best_5cv.joblib"
    ])

    # Load each model independently so one failure doesn't break others
    if xgb_path:
        try:
            models["XGBoost"] = joblib.load(xgb_path)
        except Exception as e:
            st.warning(f"XGBoost model failed to load from {os.path.basename(xgb_path)}: {e}")
    if cat_path:
        try:
            models["CatBoost"] = joblib.load(cat_path)
        except Exception as e:
            st.warning(f"CatBoost model failed to load from {os.path.basename(cat_path)}: {e}")
    if lgb_path:
        try:
            models["LightGBM"] = joblib.load(lgb_path)
        except Exception as e:
            st.warning(f"LightGBM model failed to load from {os.path.basename(lgb_path)}: {e}")

    # Do NOT restrict to CatBoost if preprocessor is missing; ensemble needs both.

    # Load metrics paths for display/selection (optional)
    metrics_paths = []
    for mp in ["hybrid_metrics.csv", "model_metrics_summary.csv", "model_metrics.csv"]:
        p = find_first_existing([mp])
        if p:
            metrics_paths.append(p)

    return preprocessor, models, metrics_paths

def pick_best_model(models: dict, metrics_paths: list):
    """Pick best model based on highest accuracy then recall from available metrics CSVs."""
    fallback_order = [
        ("CatBoost", ["CAT", "Cat", "cat"]),
        ("XGBoost", ["XGB", "XGBoost", "xgb"]),
        ("LightGBM", ["LGBM", "LightGBM", "lgbm"]),
    ]
    
    best_label = None
    best_acc = -1.0
    best_rec = -1.0
    
    for mp in metrics_paths:
        try:
            dfm = pd.read_csv(mp)
        except Exception:
            try:
                dfm = pd.read_csv(mp, index_col=0)
            except Exception:
                continue
        
        cols = {c.lower(): c for c in dfm.columns}
        if "accuracy" in cols and "recall" in cols:
            acc_col = cols["accuracy"]
            rec_col = cols["recall"]
            if "model" in {c.lower() for c in dfm.columns}:
                name_col = [c for c in dfm.columns if c.lower() == "model"][0]
                iter_rows = dfm[[name_col, acc_col, rec_col]].itertuples(index=False, name=None)
            else:
                iter_rows = zip(dfm.index.astype(str).tolist(), dfm[acc_col].tolist(), dfm[rec_col].tolist())
            
            for label, acc, rec in iter_rows:
                try:
                    acc_f = float(acc)
                    rec_f = float(rec)
                except Exception:
                    continue
                if (acc_f > best_acc) or (np.isclose(acc_f, best_acc) and rec_f > best_rec):
                    best_acc = acc_f
                    best_rec = rec_f
                    best_label = str(label)
    
    if best_label:
        label_u = best_label.upper()
        if "CAT" in label_u and "CatBoost" in models:
            return "CatBoost"
        if "XGB" in label_u and "XGBoost" in models:
            return "XGBoost"
        if ("LGBM" in label_u or "LGB" in label_u) and "LightGBM" in models:
            return "LightGBM"
    
    for key, hints in fallback_order:
        if key in models:
            return key
    return None

# Load models
preprocessor, models, metrics_paths = load_models()

if not models:
    st.error("⚠️ No models found in `model_assets/`. Please add your trained model files.")
    st.stop()

# Enforce Ensemble-only usage: require both XGBoost and CatBoost
if not ("XGBoost" in models and "CatBoost" in models):
    st.error("⚠️ Ensemble requires both XGBoost and CatBoost models. Please ensure both artifacts are present in `model_assets/`.")
    st.stop()

# Load ensemble configuration (weights and thresholds)
ensemble_config = None
ensemble_info_paths = [
    os.path.join(BASE_DIR, "model_assets", "ensemble_info_optimized.json"),
    os.path.join(BASE_DIR, "content", "models", "ensemble_info_optimized.json"),
]
for path in ensemble_info_paths:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                ensemble_config = json.load(f)
            break
        except Exception as e:
            continue

# Default ensemble weights if config not found
if ensemble_config:
    ensemble_weights_config = ensemble_config.get('weights', {})
    default_xgb_weight = ensemble_weights_config.get('XGBoost', 0.5)
    default_cat_weight = ensemble_weights_config.get('CatBoost', 0.5)
    default_lgb_weight = ensemble_weights_config.get('LightGBM', 0.0)
else:
    default_xgb_weight = 0.5
    default_cat_weight = 0.5
    default_lgb_weight = 0.0

# Main title
st.markdown('<h1 class="main-header">Predicting Heart Attack Risk: An Ensemble Modeling Approach</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced machine learning ensemble combining XGBoost, CatBoost, and LightGBM for accurate cardiovascular risk assessment</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Sidebar for model info
with st.sidebar:
    st.header("📊 Ensemble")
    # Display ensemble weights
    if ensemble_config:
        weights = ensemble_config.get('weights', {})
        xgb_w = weights.get('XGBoost', 0.5) * 100
        cat_w = weights.get('CatBoost', 0.5) * 100
        lgb_w = weights.get('LightGBM', 0.0) * 100
        if lgb_w > 0:
            st.success(f"✅ Using Optimized Ensemble\nXGBoost: {xgb_w:.1f}% | CatBoost: {cat_w:.1f}% | LightGBM: {lgb_w:.1f}%")
        else:
            st.success(f"✅ Using Optimized Ensemble\nXGBoost: {xgb_w:.1f}% | CatBoost: {cat_w:.1f}%")
    else:
        st.success("✅ Using Ensemble (50% XGBoost + 50% CatBoost)")
    _model_rows, _hybrid_rows = load_performance_metrics()
    ens_row = get_ensemble_metrics(_hybrid_rows, _model_rows)
    acc_text = f"{ens_row['accuracy']*100:.2f}%" if ens_row and ens_row.get('accuracy') is not None else "n/a"
    rec_text = f"{ens_row['recall']*100:.2f}%" if ens_row and ens_row.get('recall') is not None else "n/a"
    cols_side = st.columns(2)
    with cols_side[0]:
        st.metric("Accuracy", acc_text)
    with cols_side[1]:
        st.metric("Recall", rec_text)
    
    if metrics_paths:
        st.markdown("**Performance Metrics:**")
        for mp in metrics_paths:
            try:
                dfm = pd.read_csv(mp, index_col=0) if mp.endswith('.csv') else pd.read_csv(mp)
                st.dataframe(dfm.head(10), use_container_width=True)
            except Exception:
                pass
    
    st.markdown("---")
    st.info("""
    **Note:** This is a prediction tool, not a medical diagnosis.
    Always consult healthcare professionals for medical advice.
    """)

# Input form with all features
st.header("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    
    # Calculate BMI with category
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0
    if bmi < 18.5:
        bmi_status = "⚠️ Underweight"
        bmi_color = "inverse"
    elif bmi < 25:
        bmi_status = "✅ Normal"
        bmi_color = "normal"
    elif bmi < 30:
        bmi_status = "⚠️ Overweight"
        bmi_color = "normal"
    else:
        bmi_status = "🔴 Obese"
        bmi_color = "inverse"
    
    st.metric("BMI", f"{bmi:.2f}", delta=bmi_status, delta_color=bmi_color, 
              help="Body Mass Index - Healthy range: 18.5-24.9")

with col2:
    st.subheader("Blood Pressure")
    ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120, step=1)
    ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1)
    
    # Calculate BP_diff and category
    bp_diff = ap_hi - ap_lo
    
    # BP Status
    if ap_hi < 120 and ap_lo < 80:
        bp_status = "✅ Normal"
        bp_color = "normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_status = "⚠️ Elevated"
        bp_color = "normal"
    elif ap_hi < 140 or ap_lo < 90:
        bp_status = "🔴 Stage 1"
        bp_color = "inverse"
    else:
        bp_status = "🚨 Stage 2"
        bp_color = "inverse"
    
    st.metric("Pulse Pressure", f"{bp_diff} mmHg", delta=bp_status, delta_color=bp_color,
              help="Normal BP: <120/80 mmHg")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Medical History")
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                              format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x))
    gluc = st.selectbox("Glucose Level", options=[1, 2, 3],
                       format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x))
    smoke = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    alco = st.radio("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

with col4:
    st.subheader("Activity & Derived Features")
    active = st.radio("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    
    # Age in years (for display)
    age_years = st.number_input("Age (years)", min_value=20, max_value=100, value=50, step=1)
    age_days = age_years * 365  # Convert to days for model compatibility
    
    # Derived features
    systolic_pressure = ap_hi
    map_value = ap_lo + (bp_diff / 3)  # Mean Arterial Pressure approximation
    pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0

# Additional derived features
st.markdown("---")
st.subheader("Additional Health Metrics")

col5, col6, col7 = st.columns(3)

with col5:
    protein_level = st.number_input("Protein Level", min_value=0.0, max_value=200.0, value=14.0, step=0.1)

with col6:
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

with col7:
    # Calculate Lifestyle Score automatically
    lifestyle_score = 0
    risk_factors = []
    
    if smoke == 1:
        lifestyle_score += 1
        risk_factors.append("Smoking")
    if alco == 1:
        lifestyle_score += 1
        risk_factors.append("Alcohol")
    if active == 0:
        lifestyle_score += 1
        risk_factors.append("Physical inactivity")
    
    if lifestyle_score == 0:
        score_label = "✅ Low Risk"
        delta_color = "normal"
    elif lifestyle_score == 1:
        score_label = "⚠️ Moderate Risk"
        delta_color = "normal"
    elif lifestyle_score == 2:
        score_label = "🔴 High Risk"
        delta_color = "inverse"
    else:
        score_label = "🚨 Very High Risk"
        delta_color = "inverse"
    
    st.metric(
        "Lifestyle Risk Score", 
        f"{lifestyle_score}/3 - {score_label}",
        help=f"Auto-calculated from lifestyle factors. Risk factors: {', '.join(risk_factors) if risk_factors else 'None'}"
    )
    if risk_factors:
        st.caption(f"⚠️ Risk factors: {', '.join(risk_factors)}")

# Calculate additional derived features
obesity_flag = 1 if bmi >= 30 else 0
hypertension_flag = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
health_risk_score = lifestyle_score + obesity_flag + hypertension_flag
smoker_alcoholic = 1 if (smoke == 1 or alco == 1) else 0

# Age group and BMI category
if age_years < 30:
    age_group = "20-29"
elif age_years < 40:
    age_group = "30-39"
elif age_years < 50:
    age_group = "40-49"
elif age_years < 60:
    age_group = "50-59"
else:
    age_group = "60+"

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# BP Category
if ap_hi < 120 and ap_lo < 80:
    bp_category = "Normal"
elif ap_hi < 130 and ap_lo < 80:
    bp_category = "Elevated"
elif ap_hi < 140 or ap_lo < 90:
    bp_category = "Stage 1"
else:
    bp_category = "Stage 2"

# Risk Level (Note: data uses "Moderate" not "Medium")
if health_risk_score <= 2:
    risk_level = "Low"
elif health_risk_score <= 4:
    risk_level = "Moderate"  # Changed from "Medium" to match training data
else:
    risk_level = "High"

# Risk Age (derived)
risk_age = age_years + (health_risk_score * 5)

# Generate Reason based on risk factors
reasons = []
if obesity_flag == 1:
    reasons.append("High BMI (>30)")
if hypertension_flag == 1:
    reasons.append("High BP")
if cholesterol == 3:
    reasons.append("High cholesterol")
if gluc == 3:
    reasons.append("High glucose")
if lifestyle_score > 0:
    if smoke == 1:
        reasons.append("Smoking")
    if alco == 1:
        reasons.append("Alcohol consumption")
    if active == 0:
        reasons.append("Physical inactivity")
if not reasons:
    reasons.append("Healthy indicators")
reason = ", ".join(reasons)

# Create feature dictionary matching the dataset structure
feature_dict = {
    'age': age_days,
    'gender': gender,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': gluc,
    'smoke': smoke,
    'alco': alco,
    'active': active,
    'BMI': bmi,
    'BP_diff': bp_diff,
    'Systolic_Pressure': systolic_pressure,
    'age_years': age_years,
    'Age_Group': age_group,
    'Lifestyle_Score': lifestyle_score,
    'Obesity_Flag': obesity_flag,
    'Hypertension_Flag': hypertension_flag,
    'Health_Risk_Score': health_risk_score,
    'Reason': reason,
    'Pulse_Pressure_Ratio': pulse_pressure_ratio,
    'MAP': map_value,
    'BMI_Category': bmi_category,
    'Smoker_Alcoholic': smoker_alcoholic,
    'BP_Category': bp_category,
    'Risk_Age': risk_age,
    'Risk_Level': risk_level,
    'Protein_Level': protein_level,
    'Ejection_Fraction': ejection_fraction
}

# Health Summary Card (before prediction)
st.markdown("---")
st.subheader("📊 Health Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    if obesity_flag == 1:
        st.error("🔴 Obesity Risk")
    else:
        st.success("✅ Healthy Weight")

with summary_col2:
    if hypertension_flag == 1:
        st.error("🔴 Hypertension")
    else:
        st.success("✅ Normal BP")

with summary_col3:
    if lifestyle_score >= 2:
        st.error(f"🔴 High Lifestyle Risk ({lifestyle_score}/3)")
    elif lifestyle_score == 1:
        st.warning(f"⚠️ Moderate Risk ({lifestyle_score}/3)")
    else:
        st.success("✅ Low Risk (0/3)")

with summary_col4:
    if cholesterol == 3 or gluc == 3:
        st.error("🔴 Elevated Levels")
    elif cholesterol == 2 or gluc == 2:
        st.warning("⚠️ Above Normal")
    else:
        st.success("✅ Normal Levels")

# Prediction button
st.markdown("---")
predict_button = st.button("🔮 Predict Heart Attack Risk", type="primary", use_container_width=True)

if predict_button:
    try:
        # Create DataFrame matching EXACT training data structure (excluding id, cardio, Reason)
        feature_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
                       'smoke', 'alco', 'active', 'BMI', 'BP_diff', 'Systolic_Pressure', 'age_years', 
                       'Age_Group', 'Lifestyle_Score', 'Obesity_Flag', 'Hypertension_Flag', 'Health_Risk_Score',
                       'Pulse_Pressure_Ratio', 'MAP', 'BMI_Category', 'Smoker_Alcoholic', 'BP_Category',
                       'Risk_Age', 'Risk_Level', 'Protein_Level', 'Ejection_Fraction']
        
        # Build input row with exact feature order
        input_row = {
            'age': age_days,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'BMI': bmi,
            'BP_diff': bp_diff,
            'Systolic_Pressure': systolic_pressure,
            'age_years': age_years,
            'Age_Group': age_group,
            'Lifestyle_Score': lifestyle_score,
            'Obesity_Flag': obesity_flag,
            'Hypertension_Flag': hypertension_flag,
            'Health_Risk_Score': health_risk_score,
            'Pulse_Pressure_Ratio': pulse_pressure_ratio,
            'MAP': map_value,
            'BMI_Category': bmi_category,
            'Smoker_Alcoholic': smoker_alcoholic,
            'BP_Category': bp_category,
            'Risk_Age': risk_age,
            'Risk_Level': risk_level,
            'Protein_Level': protein_level,
            'Ejection_Fraction': ejection_fraction
        }
        
        # Create DataFrame with exact column order
        X_input = pd.DataFrame([input_row])[feature_cols]
        
        # The model expects numeric features - categorical columns were one-hot encoded during training
        # Load FULL dataset to get ALL possible categorical values (matching training)
        sample_csv = os.path.join(BASE_DIR, "content", "cardio_train_extended.csv")
        cat_cols = ['Age_Group', 'BMI_Category', 'BP_Category', 'Risk_Level']
        
        # Get all categorical values from FULL dataset (not just sample)
        if os.path.exists(sample_csv):
            # Load full dataset to get ALL unique values (matching training)
            full_df = pd.read_csv(sample_csv)
            cat_values = {}
            for col in cat_cols:
                if col in full_df.columns:
                    # Get all unique values and sort them (matching pandas get_dummies behavior)
                    cat_values[col] = sorted(full_df[col].unique().tolist())
        else:
            # Fallback to known values (matching actual data)
            cat_values = {
                'Age_Group': ['20-29', '30-39', '40-49', '50-59', '60+'],
                'BMI_Category': ['Normal', 'Obese', 'Overweight', 'Underweight'],  # Sorted order from data
                'BP_Category': ['Elevated', 'Normal', 'Stage 1', 'Stage 2'],  # Sorted order from data
                'Risk_Level': ['High', 'Low', 'Moderate']  # Note: "Moderate" not "Medium"
            }
        
        # Separate numeric and categorical columns
        numeric_cols = [col for col in X_input.columns if col not in cat_cols]
        X_numeric = X_input[numeric_cols].copy()
        
        # One-hot encode categorical columns with all possible categories in sorted order
        # This matches pandas get_dummies behavior during training
        X_cat_encoded_list = []
        for col in cat_cols:
            if col in X_input.columns:
                # Create one-hot columns for all possible values in sorted order
                for val in cat_values.get(col, []):
                    col_name = f"{col}_{val}"
                    X_cat_encoded_list.append(pd.Series([1 if X_input[col].iloc[0] == val else 0], name=col_name))
        
        if X_cat_encoded_list:
            X_cat_encoded = pd.concat(X_cat_encoded_list, axis=1)
            # Combine numeric and encoded categorical features
            X_processed = pd.concat([X_numeric, X_cat_encoded], axis=1)
        else:
            X_processed = X_numeric.copy()
        
        # Ensure all columns are numeric (float)
        X_processed = X_processed.astype(float)
        
        # Use ensemble model with optimized weights
        predictions = {}
        ensemble_probs = []
        ensemble_weights = []
        
        # Get ensemble weights from config or use defaults
        xgb_weight = default_xgb_weight if ensemble_config else 0.5
        cat_weight = default_cat_weight if ensemble_config else 0.5
        lgb_weight = default_lgb_weight if ensemble_config else 0.0
        
        # Normalize weights to sum to 1.0
        total_weight = xgb_weight + cat_weight + lgb_weight
        if total_weight > 0:
            xgb_weight = xgb_weight / total_weight
            cat_weight = cat_weight / total_weight
            lgb_weight = lgb_weight / total_weight
        
        # Try ensemble: XGBoost + CatBoost + LightGBM (if available)
        if "XGBoost" in models and "CatBoost" in models:
            try:
                # Predict with XGBoost
                xgb_model = models["XGBoost"]
                
                # Get expected features from XGBoost model
                if hasattr(xgb_model, 'feature_names_in_'):
                    expected_features = list(xgb_model.feature_names_in_)
                elif hasattr(xgb_model, 'get_booster'):
                    try:
                        booster = xgb_model.get_booster()
                        if hasattr(booster, 'feature_names') and booster.feature_names:
                            expected_features = list(booster.feature_names)
                        else:
                            # Check n_features_in_ to create placeholder columns
                            if hasattr(xgb_model, 'n_features_in_'):
                                n_features = xgb_model.n_features_in_
                                expected_features = [f"f{i}" for i in range(n_features)]
                            else:
                                expected_features = None
                    except:
                        expected_features = None
                else:
                    expected_features = None
                
                if expected_features:
                    # Align features exactly as XGBoost expects
                    X_aligned = pd.DataFrame(0.0, index=X_processed.index, columns=expected_features, dtype=float)
                    # Match columns by name
                    for col in X_processed.columns:
                        if col in X_aligned.columns:
                            X_aligned[col] = X_processed[col].values
                    X_xgb = X_aligned[expected_features]  # Ensure exact order
                else:
                    X_xgb = X_processed
                
                if hasattr(xgb_model, 'predict_proba'):
                    xgb_prob = float(xgb_model.predict_proba(X_xgb)[0, 1])
                    if xgb_weight > 0:
                        ensemble_probs.append(xgb_prob)
                        ensemble_weights.append(xgb_weight)
                    predictions["XGBoost"] = xgb_prob
            except Exception as e:
                st.warning(f"⚠️ XGBoost prediction failed (using CatBoost only): {str(e)}")
                # Don't add to predictions, but continue with CatBoost
        
        # Predict with CatBoost
        if "CatBoost" in models:
            try:
                cat_model = models["CatBoost"]
                # CatBoost is very strict about feature order and names
                if hasattr(cat_model, 'feature_names_'):
                    # CatBoost uses feature_names_ (with underscore)
                    expected_features = list(cat_model.feature_names_)
                elif hasattr(cat_model, 'feature_names_in_'):
                    expected_features = list(cat_model.feature_names_in_)
                else:
                    expected_features = None
                
                if expected_features:
                    # Create DataFrame with exact feature order and names expected by CatBoost
                    X_aligned = pd.DataFrame(0.0, index=X_processed.index, columns=expected_features, dtype=float)
                    # Match columns by name
                    for col in X_processed.columns:
                        if col in X_aligned.columns:
                            X_aligned[col] = X_processed[col].values
                    X_cat = X_aligned[expected_features]  # Ensure exact order
                else:
                    X_cat = X_processed
                
                if hasattr(cat_model, 'predict_proba'):
                    cat_prob = float(cat_model.predict_proba(X_cat)[0, 1])
                    if cat_weight > 0:
                        ensemble_probs.append(cat_prob)
                        ensemble_weights.append(cat_weight)
                    predictions["CatBoost"] = cat_prob
            except Exception as e:
                st.warning(f"CatBoost prediction failed: {e}")
        
        # Predict with LightGBM (if included in ensemble)
        if "LightGBM" in models and lgb_weight > 0:
            try:
                lgb_model = models["LightGBM"]
                # LightGBM is strict about feature order and names
                if hasattr(lgb_model, 'feature_name_'):
                    # LightGBM uses feature_name_ (with underscore, singular)
                    expected_features = list(lgb_model.feature_name_)
                elif hasattr(lgb_model, 'feature_names_in_'):
                    expected_features = list(lgb_model.feature_names_in_)
                else:
                    expected_features = None
                
                if expected_features:
                    # Create DataFrame with exact feature order and names expected by LightGBM
                    X_aligned = pd.DataFrame(0.0, index=X_processed.index, columns=expected_features, dtype=float)
                    # Match columns by name
                    for col in X_processed.columns:
                        if col in X_aligned.columns:
                            X_aligned[col] = X_processed[col].values
                    X_lgb = X_aligned[expected_features]  # Ensure exact order
                else:
                    X_lgb = X_processed
                
                if hasattr(lgb_model, 'predict_proba'):
                    lgb_prob = float(lgb_model.predict_proba(X_lgb)[0, 1])
                    ensemble_probs.append(lgb_prob)
                    ensemble_weights.append(lgb_weight)
                    predictions["LightGBM"] = lgb_prob
            except Exception as e:
                st.warning(f"LightGBM prediction failed: {e}")
        
        # Ensemble: require at least XGBoost and CatBoost probabilities
        if len(ensemble_probs) >= 2:
            # Normalize weights to sum to 1.0
            total_weight = sum(ensemble_weights)
            if total_weight > 0:
                ensemble_weights = [w / total_weight for w in ensemble_weights]
            # Ensemble prediction (weighted average)
            ensemble_prob = np.average(ensemble_probs, weights=ensemble_weights)
            predictions["Ensemble"] = ensemble_prob
        else:
            st.error("Ensemble prediction requires at least XGBoost and CatBoost probabilities.")
            with st.expander("Debug Info"):
                st.write("XGBoost available:", "XGBoost" in models)
                st.write("CatBoost available:", "CatBoost" in models)
                st.write("LightGBM available:", "LightGBM" in models)
                st.write("Ensemble probs count:", len(ensemble_probs))
                st.write("Ensemble weights:", ensemble_weights)
            st.stop()
        
        if not predictions:
            st.error("No models with predict_proba available.")
            st.stop()
        
        # Use ensemble prediction only
        if "Ensemble" in predictions:
            ensemble_prob = predictions["Ensemble"]
        else:
            st.error("Ensemble prediction missing.")
            st.stop()
        
        # Binary prediction
        prediction = 1 if ensemble_prob >= 0.5 else 0
        risk_percentage = ensemble_prob * 100
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Main result with visual indicator
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED** - {risk_percentage:.1f}% probability of heart disease")
        else:
            st.success(f"✅ **LOW RISK** - {risk_percentage:.1f}% probability of heart disease")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Risk Probability", f"{risk_percentage:.2f}%", 
                     delta=f"{'High' if risk_percentage >= 70 else 'Moderate' if risk_percentage >= 50 else 'Low'} Risk",
                     delta_color="inverse" if risk_percentage >= 70 else "normal")
        
        with col_result2:
            if risk_percentage >= 70:
                risk_level_display = "🚨 Very High"
            elif risk_percentage >= 50:
                risk_level_display = "🔴 High"
            elif risk_percentage >= 30:
                risk_level_display = "⚠️ Moderate"
            else:
                risk_level_display = "✅ Low"
            st.metric("Risk Level", risk_level_display)
        
        with col_result3:
            st.metric("Prediction", "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
                     delta="Consult Doctor" if prediction == 1 else "Continue Monitoring",
                     delta_color="inverse" if prediction == 1 else "normal")
        
        # Enhanced progress bar with color coding
        risk_bar_color = "#FF1744" if risk_percentage >= 70 else "#FF9800" if risk_percentage >= 50 else "#4CAF50"
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px; margin: 10px 0;">
            <div style="background-color: {risk_bar_color}; width: {risk_percentage}%; height: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {risk_percentage:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Reason with better formatting
        if reason and reason != "Healthy indicators":
            # Check if only "Physical inactivity" is the risk factor (less severe)
            if reason == "Physical inactivity":
                st.info(f"**ℹ️ Lifestyle Note:** {reason} - Consider adding regular physical activity to reduce risk.")
            else:
                st.warning(f"**⚠️ Key Risk Factors Identified:** {reason}")
        else:
            st.success(f"**✅ Health Status:** {reason}")
        
        # Detailed breakdown with visual bars
        with st.expander("📊 Model Details & Breakdown"):
            # Load accuracy/recall metrics for display under each model
            _model_rows_all, _hybrid_rows_all = load_performance_metrics()
            xgb_m_all = get_algo_metrics(_model_rows_all, "XGBoost")
            cat_m_all = get_algo_metrics(_model_rows_all, "CatBoost")
            lgb_m_all = get_algo_metrics(_model_rows_all, "LightGBM")
            
            # Get optimized ensemble metrics
            ens_opt_all = None
            for row in _model_rows_all or []:
                model_name = str(row.get("model", "")).upper()
                if "ENSEMBLE" in model_name and "OPTIMIZED" in model_name:
                    ens_opt_all = row
                    break
            
            # Explicit ensemble header with models and weights
            if ensemble_config:
                weights = ensemble_config.get('weights', {})
                xgb_w = weights.get('XGBoost', 0.5) * 100
                cat_w = weights.get('CatBoost', 0.5) * 100
                lgb_w = weights.get('LightGBM', 0.0) * 100
                if lgb_w > 0:
                    header_text = f"Ensemble uses: XGBoost ({xgb_w:.1f}%) + CatBoost ({cat_w:.1f}%) + LightGBM ({lgb_w:.1f}%)"
                else:
                    header_text = f"Ensemble uses: XGBoost ({xgb_w:.1f}%) + CatBoost ({cat_w:.1f}%)"
            else:
                header_text = "Ensemble uses: XGBoost + CatBoost"
            
            if ens_opt_all and ens_opt_all.get("accuracy") is not None:
                st.markdown(f"**{header_text}**  ·  Accuracy: {ens_opt_all['accuracy']*100:.2f}% | Recall: {ens_opt_all['recall']*100:.2f}%")
            else:
                st.markdown(f"**{header_text}**")
            
            # Helper function to create risk bar with percentage inside
            def create_risk_bar(risk_pct, model_name):
                # Use teal/green color for low risk, orange for moderate, red for high
                if risk_pct >= 50:
                    color = '#EF4444'  # Red
                elif risk_pct >= 30:
                    color = '#F59E0B'  # Orange
                else:
                    color = '#14B8A6'  # Teal/Green
                
                # Ensure bar width doesn't exceed 100%
                bar_width = min(risk_pct, 100)
                
                return f"""
                <div style="background: rgba(148, 163, 184, 0.15); border-radius: 8px; height: 36px; width: 100%; position: relative; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.3); margin: 8px 0;">
                    <div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: flex-start; padding-left: 8px; color: white; font-weight: 600; font-size: 0.85rem; transition: width 0.3s ease;">
                        {risk_pct:.2f}%
                    </div>
                </div>
                """
            
            # Display all models horizontally on the same line (4 columns)
            models_to_show = []
            
            # Collect all available models in order
            if "XGBoost" in predictions:
                models_to_show.append(("XGBoost Model", "XGBoost"))
            if "CatBoost" in predictions:
                models_to_show.append(("CatBoost Model", "CatBoost"))
            if "LightGBM" in predictions:
                models_to_show.append(("LightGBM Model", "LightGBM"))
            if "Ensemble" in predictions:
                models_to_show.append(("🎯 Ensemble (Final)", "Ensemble"))
            
            # Create columns for all models - equal width
            if models_to_show:
                num_cols = len(models_to_show)
                model_cols = st.columns(num_cols)
                
                for idx, (display_name, model_key) in enumerate(models_to_show):
                    with model_cols[idx]:
                        # Model title
                        st.markdown(f"**{display_name}**", unsafe_allow_html=True)
                        # Calculate risk percentage
                        risk_pct = float(predictions[model_key]) * 100
                        # Display progress bar
                        st.markdown(create_risk_bar(risk_pct, model_key), unsafe_allow_html=True)
                        # Risk percentage below bar
                        st.markdown(f"<div style='text-align: center; margin-top: -8px; font-size: 0.85rem; color: #666;'>{risk_pct:.2f}% risk</div>", unsafe_allow_html=True)
            
            # Show ensemble info
            if "Ensemble" in predictions:
                if ensemble_config:
                    weights = ensemble_config.get('weights', {})
                    xgb_w = weights.get('XGBoost', 0.5) * 100
                    cat_w = weights.get('CatBoost', 0.5) * 100
                    lgb_w = weights.get('LightGBM', 0.0) * 100
                    if lgb_w > 0:
                        st.info(f"💡 **Ensemble Method**: Weighted average (XGBoost: {xgb_w:.1f}%, CatBoost: {cat_w:.1f}%, LightGBM: {lgb_w:.1f}%). Final decision uses the Ensemble output.")
                    else:
                        st.info(f"💡 **Ensemble Method**: Weighted average (XGBoost: {xgb_w:.1f}%, CatBoost: {cat_w:.1f}%). Final decision uses the Ensemble output.")
                else:
                    st.info("💡 **Ensemble Method**: Weighted average (50% XGBoost + 50% CatBoost). Final decision uses the Ensemble output.")

            # Metrics breakdown: show per-model accuracy and averaged accuracy (concise)
            st.markdown("---")
            st.subheader("Ensemble Metrics")
            ens_row_bd = get_ensemble_metrics(_hybrid_rows_all, _model_rows_all)
            acc_bd = f"{ens_row_bd['accuracy']*100:.2f}%" if ens_row_bd and ens_row_bd.get('accuracy') is not None else "n/a"
            rec_bd = f"{ens_row_bd['recall']*100:.2f}%" if ens_row_bd and ens_row_bd.get('recall') is not None else "n/a"
            cols_acc = st.columns(2)
            with cols_acc[0]:
                st.metric("Accuracy", acc_bd)
            with cols_acc[1]:
                st.metric("Recall", rec_bd)
        
        # Recommendations
        st.markdown("---")
        if prediction == 1 or risk_percentage > 70:
            st.warning("⚠️ **High Risk Detected!** Please consult with a healthcare professional immediately.")
            st.info("""
            **Recommendations:**
            - Schedule an appointment with a cardiologist
            - Monitor blood pressure regularly
            - Maintain a healthy diet and exercise routine
            - Avoid smoking and limit alcohol consumption
            - Follow up with regular health checkups
            """)
        elif risk_percentage > 50:
            st.warning("⚠️ **Moderate Risk** - Consider consulting a healthcare professional.")
        else:
            st.success("✅ **Low Risk** - Continue maintaining a healthy lifestyle!")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)
