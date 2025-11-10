# Google Colab Time Estimate & Setup Guide

## ⏱️ Time Comparison

### Current Local Setup (Docker)
- **CPUs:** 2 cores
- **Memory:** 4 GB
- **Total Time:** ~24.4 hours
  - XGBoost: ~2.9 hours
  - CatBoost: ~12.5 hours
  - LightGBM: ~9.0 hours

---

## 🆓 Google Colab Free Tier (CPU Only)

### Specifications
- **CPUs:** 1-2 cores (variable, shared resources)
- **Memory:** ~12.7 GB RAM
- **GPU:** None
- **Session Timeout:** 12 hours (disconnects after inactivity)

### Estimated Time
- **Total:** ~30.5 hours (20% slower than local)
  - XGBoost: ~3.7 hours
  - CatBoost: ~15.6 hours
  - LightGBM: ~11.3 hours

### ⚠️ Limitations
- **May timeout before completion** (12-hour limit)
- Slower due to shared resources
- May need to restart and resume from checkpoints

---

## 🎮 Google Colab Free Tier + GPU (T4)

### Specifications
- **CPUs:** 1-2 cores
- **Memory:** ~12.7 GB RAM
- **GPU:** NVIDIA T4 (16 GB)
- **Session Timeout:** 12 hours

### Estimated Time
- **Total:** ~18.0 hours (26% faster than local)
  - XGBoost: ~1.9 hours (50% faster with GPU)
  - CatBoost: ~9.6 hours (30% faster with GPU)
  - LightGBM: ~6.4 hours (40% faster with GPU)

### ⚠️ Limitations
- **May timeout before completion** (12-hour limit)
- GPU availability not guaranteed (may need to wait)
- Requires code modifications for GPU support

---

## 💎 Google Colab Pro ($10/month)

### Specifications
- **CPUs:** 2-4 cores (better allocation)
- **Memory:** ~32 GB RAM
- **GPU:** Better GPU access (T4/V100)
- **Session Timeout:** 24 hours
- **Background Execution:** Yes

### Estimated Time (CPU)
- **Total:** ~20.4 hours (17% faster than local)
  - XGBoost: ~2.4 hours
  - CatBoost: ~10.4 hours
  - LightGBM: ~7.5 hours

### Estimated Time (with GPU)
- **Total:** ~15.0 hours (39% faster than local)
  - XGBoost: ~1.6 hours
  - CatBoost: ~8.0 hours
  - LightGBM: ~5.4 hours

### ✅ Advantages
- Longer session time (24 hours)
- Background execution (can close browser)
- Better resource allocation
- More reliable GPU access

---

## 📊 Summary Table

| Platform | CPUs | GPU | Total Time | Cost | Session Limit |
|----------|------|-----|------------|------|---------------|
| **Local (Docker)** | 2 | No | ~24.4 hrs | Free | None |
| **Colab Free (CPU)** | 1-2 | No | ~30.5 hrs | Free | 12 hrs ⚠️ |
| **Colab Free (GPU)** | 1-2 | T4 | ~18.0 hrs | Free | 12 hrs ⚠️ |
| **Colab Pro (CPU)** | 2-4 | No | ~20.4 hrs | $10/mo | 24 hrs |
| **Colab Pro (GPU)** | 2-4 | T4/V100 | ~15.0 hrs | $10/mo | 24 hrs |

---

## 🚀 Setting Up for Google Colab

### 1. Enable GPU (if using)
```python
# In Colab, go to: Runtime → Change runtime type → Hardware accelerator → GPU
```

### 2. Install Dependencies
```python
!pip install xgboost catboost lightgbm optuna pandas numpy scikit-learn joblib
```

### 3. Upload Data
```python
from google.colab import files
# Upload cardio_train_extended.csv
uploaded = files.upload()
```

### 4. Modify Code for GPU Support

You'll need to modify `improve_models.py` to enable GPU:

**For XGBoost:**
```python
# Change tree_method to use GPU
xgb_params = {
    'tree_method': 'gpu_hist',  # Enable GPU
    'device': 'cuda',  # Use CUDA
    # ... other parameters
}
```

**For CatBoost:**
```python
cat_params = {
    'task_type': 'GPU',  # Enable GPU
    'devices': '0',  # Use first GPU
    # ... other parameters
}
```

**For LightGBM:**
```python
lgb_params = {
    'device': 'gpu',  # Enable GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    # ... other parameters
}
```

### 5. Handle Session Timeouts

For long-running training, save checkpoints:

```python
import pickle

# Save study state periodically
def save_checkpoint(study, trial):
    if trial.number % 50 == 0:
        with open('study_checkpoint.pkl', 'wb') as f:
            pickle.dump(study, f)

# Load checkpoint if resuming
try:
    with open('study_checkpoint.pkl', 'rb') as f:
        study = pickle.load(f)
except FileNotFoundError:
    study = optuna.create_study(...)
```

---

## 💡 Recommendations

### Best Option: **Colab Pro + GPU**
- ✅ Fastest completion (~15 hours)
- ✅ 24-hour session limit (enough time)
- ✅ Background execution
- ✅ Most reliable

### Budget Option: **Colab Free + GPU**
- ✅ Free
- ✅ Faster than local (~18 hours)
- ⚠️ May timeout (12-hour limit)
- ⚠️ Need to implement checkpointing

### Local Option: **Keep Current Setup**
- ✅ No cost
- ✅ No timeouts
- ✅ Full control
- ⚠️ Slower (~24 hours)

---

## 📝 Important Notes

1. **GPU Acceleration:** Requires code modifications to enable GPU support in XGBoost, CatBoost, and LightGBM
2. **Session Limits:** Free tier has 12-hour limits - may need to restart
3. **Resource Availability:** Colab resources vary - actual times may differ
4. **Checkpointing:** Essential for long runs on free tier
5. **Data Upload:** Need to upload dataset to Colab (or use Google Drive)

---

## 🔧 Quick Colab Setup Script

```python
# Run this in a Colab cell
!pip install xgboost catboost lightgbm optuna pandas numpy scikit-learn joblib

# Enable GPU (if available)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Upload your data file
from google.colab import files
uploaded = files.upload()

# Then run your improve_models.py script
# (with GPU modifications)
```

---

**Last Updated:** November 9, 2025

