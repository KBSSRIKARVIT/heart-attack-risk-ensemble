# 📤 GitHub Setup Guide for Hugging Face Deployment

## Step 1: Initialize Git Repository (if not done)

If you haven't initialized git yet, run:
```bash
cd /home/kbs/Documents/heart-attack-risk-ensemble
git init
```

## Step 2: Using GitHub Desktop

### Option A: Clone Existing Repository
1. Open GitHub Desktop
2. Click "File" → "Clone Repository"
3. If you already created a repo on GitHub.com:
   - Select "GitHub.com" tab
   - Choose your repository
   - Click "Clone"

### Option B: Create New Repository
1. Open GitHub Desktop
2. Click "File" → "New Repository"
3. Fill in:
   - **Name**: `heart-attack-risk-ensemble` (or your choice)
   - **Description**: "Heart Attack Risk Prediction using Ensemble ML Models"
   - **Local Path**: `/home/kbs/Documents/heart-attack-risk-ensemble`
   - **Initialize with README**: ✅ Check this
   - **Git Ignore**: Python
   - **License**: MIT (optional)
4. Click "Create Repository"

## Step 3: Add Files to GitHub Desktop

1. In GitHub Desktop, you'll see all your files listed
2. Review the changes:
   - ✅ **Include**: All Python files, requirements.txt, configs, documentation
   - ✅ **Include**: Model files (if under 100MB each)
   - ⚠️ **Check**: Large files (>100MB) - GitHub has limits

### Files to Commit:
- ✅ `streamlit_app.py`
- ✅ `requirements.txt`
- ✅ `Dockerfile`
- ✅ `render.yaml`
- ✅ `.streamlit/config.toml`
- ✅ `TEST_CASES.md`
- ✅ `DEPLOYMENT_CHECKLIST.md`
- ✅ `DEPLOYMENT_OPTIONS.md`
- ✅ `README.md`
- ✅ `model_assets/` (with optimized models)
- ✅ `content/models/` (if needed)
- ✅ `.gitignore`

## Step 4: Commit Changes

1. In GitHub Desktop, you'll see all changes
2. **Summary**: Write a commit message like:
   ```
   Initial commit: Heart Attack Risk Prediction App
   - Streamlit app with ensemble models (XGBoost, CatBoost, LightGBM)
   - Optimized models with 80.77% accuracy, 93.27% recall
   - Complete UI with model breakdown
   - Test cases and deployment documentation
   ```
3. **Description** (optional): Add more details
4. Click **"Commit to main"** (or your branch name)

## Step 5: Publish to GitHub

1. Click **"Publish repository"** button (top right)
2. If creating new repo:
   - ✅ **Keep code private**: Uncheck (make it public for Hugging Face)
   - ✅ **Add description**: "Heart Attack Risk Prediction using Ensemble ML Models"
3. Click **"Publish Repository"**

## Step 6: Verify on GitHub.com

1. Go to https://github.com/YOUR_USERNAME/heart-attack-risk-ensemble
2. Verify all files are there
3. Check that model files are uploaded (if they're not too large)

## ⚠️ Important Notes

### File Size Limits:
- **GitHub**: 100MB per file (hard limit)
- **GitHub LFS**: For files >100MB, use Git LFS
- **Model files**: Usually 10-50MB each, should be fine

### If Models Are Too Large:
1. Use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   ```
2. Or exclude from git and upload separately to Hugging Face

### Repository Visibility:
- **Public**: Required for Hugging Face Spaces (free tier)
- **Private**: Requires Hugging Face Pro for private spaces

## ✅ Next Steps After GitHub Push

Once your code is on GitHub:
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Select "Streamlit"
4. Connect your GitHub repository
5. Deploy!

---

## 🐛 Troubleshooting

### GitHub Desktop Not Showing Files:
- Make sure you're in the correct directory
- Check if `.git` folder exists
- Try refreshing GitHub Desktop

### Large File Warnings:
- If models are too large, use Git LFS or exclude them
- Hugging Face can pull models from other sources if needed

### Commit Fails:
- Check file permissions
- Make sure you're not committing sensitive files
- Review `.gitignore` file

