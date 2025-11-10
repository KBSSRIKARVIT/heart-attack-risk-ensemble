# 🚀 Running Streamlit App Locally

## ✅ What's Been Done

1. **Optimized models copied** to `model_assets/`:
   - ✅ XGBoost_optimized.joblib
   - ✅ CatBoost_optimized.joblib
   - ✅ LightGBM_optimized.joblib
   - ✅ ensemble_info_optimized.json
   - ✅ model_metrics_optimized.csv
   - ✅ hybrid_metrics.csv

2. **Streamlit app updated**:
   - ✅ Uses optimized models
   - ✅ Loads ensemble weights from config
   - ✅ Displays optimized ensemble weights in sidebar
   - ✅ All paths configured correctly

## 📋 To Run Locally

### Option 1: Using Docker (Recommended - Already Set Up)

```bash
# The Docker environment already has all dependencies
docker run --rm -p 8501:8501 \
  -v "$(pwd)/model_assets:/app/model_assets" \
  -v "$(pwd)/content:/app/content" \
  -v "$(pwd)/streamlit_app.py:/app/streamlit_app.py" \
  heart-optimization \
  streamlit run streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.port=8501
```

Then open: http://localhost:8501

### Option 2: Install Dependencies Locally

**Note:** Python 3.14.0 may have compatibility issues. Consider using Python 3.11 or 3.12.

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost catboost lightgbm joblib

# Run the app
streamlit run streamlit_app.py
```

### Option 3: Use Virtual Environment (Recommended)

```bash
# Create virtual environment with Python 3.11 or 3.12
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## 🎯 What to Test

1. **Model Loading**: Check sidebar shows "Using Optimized Ensemble" with correct weights
2. **Input Form**: Fill in patient information
3. **Prediction**: Click "Predict Heart Attack Risk" button
4. **Results**: Verify prediction and risk percentage display correctly
5. **Ensemble Info**: Check that ensemble weights match optimized config (XGB: 5%, CAT: 85%, LGB: 10%)

## 📊 Expected Results

- **Ensemble Weights**: XGBoost: 5.0% | CatBoost: 85.0% | LightGBM: 10.0%
- **Accuracy**: ~80.8% (from optimized metrics)
- **Recall**: ~93.3% (from optimized metrics)
- **ROC-AUC**: ~0.925

## 🐛 Troubleshooting

### If models don't load:
- Check `model_assets/` folder has all `.joblib` files
- Verify file permissions are readable

### If dependencies fail:
- Use Docker (Option 1) - already configured
- Or use Python 3.11/3.12 instead of 3.14

### If app doesn't start:
- Check port 8501 is not in use: `lsof -i :8501`
- Try different port: `streamlit run streamlit_app.py --server.port=8502`

## ✅ Once Working Locally

After confirming the app works locally, we'll proceed with Hugging Face deployment!

