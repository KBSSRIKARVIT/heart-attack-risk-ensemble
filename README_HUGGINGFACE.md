# 🚀 Hugging Face Spaces Deployment

## Quick Start

1. **Go to**: https://huggingface.co/spaces
2. **Click**: "Create new Space"
3. **Select**: **Docker** SDK ✅
4. **Connect**: Your GitHub repository `KBSSRIKARVIT/heart-attack-risk-ensemble`
5. **Wait**: 5-10 minutes for build
6. **Done**: Your app is live!

## SDK Selection

When creating the space, you'll see three options:

1. **Gradio** - For Gradio apps (not this one)
2. **Docker** - ✅ **Select this one!** (for Streamlit)
3. **Static** - For static HTML sites (not this one)

## Why Docker?

- ✅ Your app is a Streamlit app
- ✅ You have a `Dockerfile` ready
- ✅ Docker SDK supports Streamlit
- ✅ More control over the environment

## What Happens

1. Hugging Face pulls your code from GitHub
2. Builds Docker image using your `Dockerfile`
3. Runs the container with Streamlit
4. Maps port 7860 to web interface
5. Your app is accessible worldwide!

## Files Needed

- ✅ `Dockerfile` - Container configuration
- ✅ `streamlit_app.py` - Main app
- ✅ `requirements.txt` - Dependencies
- ✅ `model_assets/` - Model files
- ✅ All other app files

## Port Configuration

The Dockerfile is configured to:
- Use `PORT` environment variable (set by Hugging Face)
- Default to port 7860 (Hugging Face standard)
- Also expose port 8051 for compatibility

## After Deployment

Your app will be available at:
```
https://huggingface.co/spaces/KBSSRIKARVIT/heart-attack-risk-predictor
```

Replace `KBSSRIKARVIT` with your Hugging Face username and `heart-attack-risk-predictor` with your space name.

---

**That's it!** Select Docker SDK and deploy! 🎉

