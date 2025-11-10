# 🐳 Deploy to Hugging Face Spaces Using Docker SDK

## ✅ Select Docker SDK

When creating your Hugging Face Space, select **"Docker"** as the SDK.

Hugging Face will automatically detect and use your `Dockerfile`.

## 🚀 Deployment Steps

### Step 1: Create New Space

1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `heart-attack-risk-predictor` (or your choice)
   - **SDK**: Select **"Docker"** ✅
   - **Visibility**: **Public** (required for free tier)
   - **Hardware**: **CPU basic** (free tier)
   - Click **"Create Space"**

### Step 2: Connect GitHub Repository

1. After creating the space, go to **"Settings"** tab
2. Scroll to **"Repository"** section
3. Select **"GitHub"** as source
4. **Repository**: `KBSSRIKARVIT/heart-attack-risk-ensemble`
5. **Branch**: `main`
6. Click **"Save"**

### Step 3: Verify Dockerfile

Hugging Face will automatically use your `Dockerfile`. Verify it's correct:

- ✅ Base image: `python:3.11-slim`
- ✅ Port: `8051` (Hugging Face will map this automatically)
- ✅ Streamlit command configured
- ✅ All dependencies in `requirements.txt`

### Step 4: Update Dockerfile for Hugging Face (if needed)

Hugging Face Spaces automatically:
- Maps port 8051 to the web interface
- Sets environment variables
- Handles HTTPS

Your current Dockerfile should work, but we may need a small adjustment for Hugging Face's port expectations.

### Step 5: Wait for Build

1. Hugging Face will:
   - Pull your code from GitHub
   - Build the Docker image
   - Deploy the container
2. **Build time**: 5-10 minutes (first time)
3. Watch progress in **"Logs"** tab

### Step 6: Access Your App

Once built, your app will be live at:
```
https://huggingface.co/spaces/KBSSRIKARVIT/heart-attack-risk-predictor
```

## 🔧 Dockerfile Configuration for Hugging Face

Hugging Face Spaces expects the app to:
- Listen on the port specified in `PORT` environment variable
- Or use port 7860 (default)

Let's check if we need to update the Dockerfile:

```dockerfile
# Current Dockerfile uses port 8051
# Hugging Face can map this, but let's make it flexible
```

## 📝 Optional: Update Dockerfile for Hugging Face

If the app doesn't start, we may need to update the Dockerfile to use Hugging Face's expected port:

```dockerfile
# Use PORT environment variable or default to 7860
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.port=${PORT:-7860}"]
```

But first, let's try with the current Dockerfile - Hugging Face should handle the port mapping.

## ✅ Verification Checklist

After deployment:

- [ ] Build completes without errors
- [ ] App loads at the Hugging Face URL
- [ ] Streamlit interface displays correctly
- [ ] Models load successfully
- [ ] Predictions work

## 🐛 Troubleshooting

### Build Fails

**Check logs** for specific errors:
- Dependency installation issues
- Dockerfile syntax errors
- Port configuration issues

### App Doesn't Start

**Possible issues**:
- Port mismatch (Hugging Face expects 7860 or PORT env var)
- Missing model files
- Path issues

**Solution**: Update Dockerfile to use `PORT` environment variable.

### Models Not Loading

**Verify**:
- Model files are in the repository
- Paths in `streamlit_app.py` are correct
- Files are copied in Dockerfile

## 🎯 Summary

1. **Select "Docker" SDK** when creating the space
2. **Connect GitHub repository**
3. **Hugging Face uses your Dockerfile automatically**
4. **Wait for build** (5-10 minutes)
5. **Access your app** at the Hugging Face URL

---

**Ready to deploy?** Follow these steps and your Streamlit app will be live on Hugging Face Spaces! 🚀

