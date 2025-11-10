# Quick Start Guide: Model Improvement

## Overview

This guide helps you improve your heart attack risk prediction models using advanced optimization techniques.

## 🐳 Docker Option (Recommended)

If you have Docker installed, this is the easiest way to run optimization:

```bash
# Simple one-command execution
./run_optimization_docker.sh

# Or with custom settings
./run_optimization_docker.sh --trials 50

# Run feature analysis
./run_optimization_docker.sh --script feature_importance_analysis.py
```

See [DOCKER_OPTIMIZATION.md](DOCKER_OPTIMIZATION.md) for detailed Docker instructions.

---

## Local Installation Option

## Current Performance

Your current models achieve:
- **Accuracy:** ~85.1%
- **Recall:** ~84.3%
- **ROC-AUC:** ~92.5%

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install Optuna and other required packages.

### Step 2: Run Model Optimization

```bash
python improve_models.py
```

**What this does:**
- Optimizes hyperparameters for XGBoost, CatBoost, and LightGBM using Optuna
- Finds optimal prediction thresholds for each model
- Optimizes ensemble weights
- Saves improved models to `content/models/`

**Time:** ~1-2 hours (100 trials per model)

**Output:**
- `XGBoost_optimized.joblib`
- `CatBoost_optimized.joblib`
- `LightGBM_optimized.joblib`
- `model_metrics_optimized.csv`
- `ensemble_info_optimized.json`
- `best_params_optimized.json`

### Step 3: Analyze Feature Importance (Optional)

```bash
python feature_importance_analysis.py
```

**What this does:**
- Analyzes feature importance across all models
- Performs statistical feature selection
- Generates visualizations
- Provides feature selection recommendations

**Time:** ~5-10 minutes

**Output:**
- `feature_selection_recommendations.json`
- `feature_importance_top30.png`
- `feature_correlation_top30.png`

### Step 4: Compare Results

```bash
python compare_models.py
```

**What this does:**
- Compares baseline vs optimized models
- Shows improvement metrics
- Displays optimal ensemble configuration

## Expected Improvements

After running the optimization:

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| Accuracy | 85.1% | 86-87% | +1-2% |
| Recall | 84.3% | 86-87.5% | +2-4% |
| F1 Score | 85.0% | 86-87% | +1-2% |

## Key Improvements Implemented

1. ✅ **Optuna Hyperparameter Optimization**
   - Tree-structured Parzen Estimator (TPE)
   - 100+ trials per model
   - Expanded parameter search spaces

2. ✅ **Multi-Objective Optimization**
   - Combined accuracy + recall scoring
   - Threshold optimization per model

3. ✅ **Enhanced Ensemble**
   - Three-model ensemble (XGBoost + CatBoost + LightGBM)
   - Optimized weights
   - Optimized threshold

4. ✅ **Feature Analysis**
   - Importance extraction
   - Statistical selection methods
   - Recommendations for feature engineering

## Faster Alternative

If you want faster results (less optimal but quicker):

Edit `improve_models.py` and change:
```python
n_trials = 100  # Change to 30-50 for faster results
```

## Troubleshooting

**Problem:** Script takes too long
- **Solution:** Reduce `n_trials` to 30-50

**Problem:** Memory errors
- **Solution:** Reduce `n_jobs` or use smaller data sample

**Problem:** No improvement
- **Solution:** Check data preprocessing matches training data

## Next Steps

1. Run optimization scripts
2. Compare results with baseline
3. Test optimized models on validation set
4. Deploy best performing model
5. Monitor performance

## Files Created

- `improve_models.py` - Main optimization script
- `feature_importance_analysis.py` - Feature analysis
- `compare_models.py` - Comparison tool
- `IMPROVEMENTS.md` - Detailed improvement analysis
- `QUICK_START.md` - This guide

## Questions?

See `IMPROVEMENTS.md` for detailed explanations of all improvements.

