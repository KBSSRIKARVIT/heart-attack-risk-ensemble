import os
import json
import joblib
import pandas as pd
import numpy as np

# Paths inside the container image
APP_DIR = "/app"
ASSETS_DIR = os.path.join(APP_DIR, "model_assets")

# Resolve model paths with fallbacks
XGB_CANDIDATES = [
    "XGB_spw.joblib", "XGBoost_best_5cv.joblib", "XGBoost_best.joblib",
    "XGBoost.joblib", "xgb_model.joblib", "xgb_full.joblib"
]
CAT_CANDIDATES = [
    "CAT_cw.joblib", "CatBoost_best_5cv.joblib", "CatBoost_best.joblib",
    "CatBoost.joblib", "catboost.joblib", "cat_model.joblib", "cat_full.joblib"
]


def find_first(path_list):
    for name in path_list:
        p = os.path.join(ASSETS_DIR, name)
        if os.path.exists(p):
            return p
    return None


def build_sample_input():
    # Use values close to the UI defaults
    gender = 1
    height = 170
    weight = 70.0
    ap_hi = 120
    ap_lo = 80
    cholesterol = 1
    gluc = 1
    smoke = 0
    alco = 0
    active = 1
    age_years = 50
    age_days = age_years * 365

    # Derived features
    bmi = weight / ((height / 100) ** 2)
    bp_diff = ap_hi - ap_lo
    systolic_pressure = ap_hi
    map_value = ap_lo + (bp_diff / 3)
    pulse_ratio = bp_diff / ap_hi if ap_hi > 0 else 0

    obesity_flag = 1 if bmi >= 30 else 0
    hypertension_flag = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0
    lifestyle_score = (1 if smoke == 1 else 0) + (1 if alco == 1 else 0) + (1 if active == 0 else 0)
    health_risk_score = lifestyle_score + obesity_flag + hypertension_flag
    smoker_alcoholic = 1 if (smoke == 1 or alco == 1) else 0

    age_group = "50-59"
    bmi_category = (
        "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    )
    if ap_hi < 120 and ap_lo < 80:
        bp_category = "Normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_category = "Elevated"
    elif ap_hi < 140 or ap_lo < 90:
        bp_category = "Stage 1"
    else:
        bp_category = "Stage 2"

    risk_level = "Low" if health_risk_score <= 2 else "Medium" if health_risk_score <= 4 else "High"
    risk_age = age_years + (health_risk_score * 5)

    protein_level = 14.0
    ejection_fraction = 60.0

    feature_cols = [
        'age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','BMI','BP_diff',
        'Systolic_Pressure','age_years','Age_Group','Lifestyle_Score','Obesity_Flag','Hypertension_Flag','Health_Risk_Score',
        'Pulse_Pressure_Ratio','MAP','BMI_Category','Smoker_Alcoholic','BP_Category','Risk_Age','Risk_Level','Protein_Level','Ejection_Fraction'
    ]

    row = {
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
        'Pulse_Pressure_Ratio': pulse_ratio,
        'MAP': map_value,
        'BMI_Category': bmi_category,
        'Smoker_Alcoholic': smoker_alcoholic,
        'BP_Category': bp_category,
        'Risk_Age': risk_age,
        'Risk_Level': risk_level,
        'Protein_Level': protein_level,
        'Ejection_Fraction': ejection_fraction,
    }

    X = pd.DataFrame([row])[feature_cols]

    # One-hot encode categoricals using the same fallback values as app
    cat_cols = ['Age_Group', 'BMI_Category', 'BP_Category', 'Risk_Level']
    cat_values = {
        'Age_Group': ['20-29', '30-39', '40-49', '50-59', '60+'],
        'BMI_Category': ['Underweight', 'Normal', 'Overweight', 'Obese'],
        'BP_Category': ['Normal', 'Elevated', 'Stage 1', 'Stage 2'],
        'Risk_Level': ['Low', 'Medium', 'High'],
    }
    numeric_cols = [c for c in X.columns if c not in cat_cols]
    Xn = X[numeric_cols].copy()

    parts = []
    for col in cat_cols:
        if col in X.columns:
            for v in cat_values[col]:
                parts.append(pd.Series([1 if X[col].iloc[0] == v else 0], name=f"{col}_{v}"))
    Xe = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
    Xp = pd.concat([Xn, Xe], axis=1).astype(float)

    return Xp


def align_for_model(model, Xp):
    # Align dataframe columns to model expectations (by name when available)
    X_aligned = Xp
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        Xa = pd.DataFrame(0.0, index=Xp.index, columns=expected)
        for c in Xp.columns:
            if c in Xa.columns:
                Xa[c] = Xp[c].values
        X_aligned = Xa[expected]
    else:
        try:
            # xgboost booster feature names
            booster = getattr(model, 'get_booster', lambda: None)()
            if booster is not None and getattr(booster, 'feature_names', None):
                expected = list(booster.feature_names)
                Xa = pd.DataFrame(0.0, index=Xp.index, columns=expected)
                for c in Xp.columns:
                    if c in Xa.columns:
                        Xa[c] = Xp[c].values
                X_aligned = Xa[expected]
            elif hasattr(model, 'n_features_in_'):
                n = int(getattr(model, 'n_features_in_', Xp.shape[1]))
                # Fallback: trim or pad to match expected number of features
                if Xp.shape[1] >= n:
                    X_aligned = Xp.iloc[:, :n].copy()
                else:
                    # pad with zero columns
                    pad = pd.DataFrame(0.0, index=Xp.index, columns=[f"pad_{i}" for i in range(n - Xp.shape[1])])
                    X_aligned = pd.concat([Xp, pad], axis=1)
        except Exception:
            pass
    return X_aligned


def main():
    xgb_path = find_first(XGB_CANDIDATES)
    cat_path = find_first(CAT_CANDIDATES)

    assert xgb_path and os.path.exists(xgb_path), f"XGBoost artifact not found in {ASSETS_DIR}"
    assert cat_path and os.path.exists(cat_path), f"CatBoost artifact not found in {ASSETS_DIR}"

    xgb = joblib.load(xgb_path)
    cat = joblib.load(cat_path)

    Xp = build_sample_input()
    # Force shape match for XGBoost using n_features_in_
    n_xgb = int(getattr(xgb, 'n_features_in_', Xp.shape[1]))
    X_xgb = Xp.iloc[:, :n_xgb].values
    print(f"DBG: n_xgb={n_xgb}, Xp.shape={Xp.shape}, X_xgb.shape={X_xgb.shape}")
    # Align for CatBoost (by names if available), otherwise force shape
    if hasattr(cat, 'feature_names_in_'):
        X_cat = align_for_model(cat, Xp)
    else:
        # CatBoost models often don't expose names; pass full matrix
        X_cat = Xp.values
    print(f"DBG: X_cat.shape={X_cat.shape}")

    if hasattr(xgb, 'predict_proba'):
        px = float(xgb.predict_proba(X_xgb)[0, 1])
    else:
        px = float(xgb.predict(X_xgb)[0])

    if hasattr(cat, 'predict_proba'):
        pc = float(cat.predict_proba(X_cat)[0, 1])
    else:
        pc = float(cat.predict(X_cat)[0])

    pe = 0.5 * px + 0.5 * pc
    out = {
        'xgb_prob': px,
        'cat_prob': pc,
        'ensemble_prob': pe,
        'ensemble_risk_percent': pe * 100.0,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
