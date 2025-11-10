# 🚀 Hugging Face Spaces Deployment - Quick Steps

## ✅ Select Docker SDK

When creating your Hugging Face Space:
1. **SDK Options**: You'll see 3 options
   - **Gradio** ❌ (for Gradio apps)
   - **Docker** ✅ **SELECT THIS ONE!** (for Streamlit)
   - **Static** ❌ (for static HTML)

2. **Select "Docker"** - This will use your Dockerfile automatically

## 📋 Deployment Steps

### Step 1: Create Space
1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `heart-attack-risk-predictor`
   - **SDK**: **Docker** ✅
   - **Visibility**: **Public**
   - **Hardware**: **CPU basic** (free)
4. Click **"Create Space"**

### Step 2: Connect GitHub
1. Go to **"Settings"** tab in your space
2. Scroll to **"Repository"** section
3. Select **"GitHub"**
4. Repository: `KBSSRIKARVIT/heart-attack-risk-ensemble`
5. Branch: `main`
6. Click **"Save"**

### Step 3: Wait for Build
- Hugging Face will build your Docker image
- Takes 5-10 minutes (first time)
- Watch progress in **"Logs"** tab

### Step 4: Access Your App
Once built, your app is live at:
```
https://huggingface.co/spaces/KBSSRIKARVIT/heart-attack-risk-predictor
```

## ✅ What's Configured

- ✅ Dockerfile updated for Hugging Face (uses PORT env var)
- ✅ Port 7860 (Hugging Face standard)
- ✅ All model files included
- ✅ Content directory included
- ✅ All dependencies in requirements.txt

## 🎯 That's It!

Select **Docker SDK** and deploy! Your Streamlit app will run in the Docker container.

---

**Need help?** Check `HUGGINGFACE_DEPLOY_DOCKER.md` for detailed instructions.

