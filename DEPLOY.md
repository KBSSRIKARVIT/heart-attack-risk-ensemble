# 🚀 Deploy to Render - Step by Step Guide

## Prerequisites
- GitHub account
- Render account (free - sign up at https://render.com)

## 📋 Deployment Steps

### Step 1: Push Code to GitHub
```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Ready for Render deployment"

# Create a new repository on GitHub (https://github.com/new)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign in or create account (use GitHub login for easier setup)

2. **Create New Web Service**
   - Click "New +" button → "Web Service"
   - Connect your GitHub account if not already connected
   - Select your repository

3. **Configure Service** (Render will auto-detect render.yaml)
   - **Name**: heart-attack-risk-predictor (or your choice)
   - **Runtime**: Docker
   - **Plan**: Free
   - Click "Create Web Service"

4. **Wait for Build & Deploy**
   - Render will automatically:
     - Build your Docker image
     - Deploy the container
     - Assign a public URL
   - Build takes 2-5 minutes

5. **Access Your App**
   - Once deployed, you'll get a URL like:
     `https://heart-attack-risk-predictor.onrender.com`
   - Open it in your browser!

## ⚠️ Important Notes

### Free Tier Limitations
- App sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- 750 hours/month free (enough for most usage)

### Custom Domain (Optional)
- Go to Settings → Custom Domain
- Add your domain (requires DNS setup)

### Environment Variables (if needed)
- Go to Environment → Add Environment Variable
- Currently none required for this app

### Logs & Monitoring
- View logs: Click "Logs" tab in dashboard
- Monitor performance: "Metrics" tab

## 🔄 Auto-Deploy on Updates

Once set up, any push to your GitHub main branch will automatically:
1. Trigger new build
2. Deploy updated version
3. Switch traffic to new version

No manual intervention needed!

## 🐛 Troubleshooting

### Build Fails
- Check Render logs for errors
- Verify Dockerfile builds locally: `docker build -t heart-app .`
- Check all files are committed to Git

### App Won't Start
- Check port is 8051 (matches Dockerfile EXPOSE)
- Verify model files are in model_assets/
- Check logs for Python errors

### Slow Response
- Free tier sleeps after inactivity
- Upgrade to paid plan ($7/month) for always-on

## 📦 What's Included

Your repo now has:
- ✅ `render.yaml` - Render configuration
- ✅ `Dockerfile` - Container definition
- ✅ `requirements.txt` - Python dependencies
- ✅ `streamlit_app.py` - Main application
- ✅ `model_assets/` - ML models
- ✅ `.streamlit/config.toml` - Streamlit settings

## 🎉 You're Done!

Your app is now live and accessible worldwide!

### Next Steps:
- Share your URL
- Monitor usage in Render dashboard
- Set up custom domain if needed
- Consider upgrading if you need 24/7 uptime

---

**Need Help?**
- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
