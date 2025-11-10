FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for lightgbm, xgboost, catboost (runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy app code
COPY streamlit_app.py /app/streamlit_app.py
COPY .streamlit /app/.streamlit

# Copy model assets if present
RUN mkdir -p /app/model_assets
COPY model_assets/ /app/model_assets/

# Copy content directory (for data file used in feature encoding)
RUN mkdir -p /app/content
COPY content/ /app/content/

# Optional: copy test script for quick in-container verification
COPY test_predict.py /app/test_predict.py

# Hugging Face Spaces uses PORT environment variable or defaults to 7860
# Expose both ports for compatibility
EXPOSE 7860
EXPOSE 8051

# Use PORT environment variable if set, otherwise default to 7860 (Hugging Face standard)
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.port=${PORT:-7860}"]


