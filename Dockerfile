# ============================================================
# AeroGuard — Dockerfile
#
# Multi-stage approach nahi kiya — single stage
# kyunki ML models ke saath complexity badhti hai
#
# Services:
#   FastAPI  : port 8000
#   Streamlit: port 8501
#
# Build:
#   docker build -t aeroguard .
#
# Run:
#   docker-compose up
# ============================================================

# Python 3.11 slim — lightweight base
FROM python:3.11-slim

# Working directory
WORKDIR /app

# System dependencies
# libgomp1 — PyTorch ke liye (OpenMP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Requirements pehle copy karo — Docker cache optimize
COPY requirements.txt .

# Dependencies install karo
# --no-cache-dir — image size kam karo
RUN pip install --no-cache-dir -r requirements.txt

# Poora project copy karo
COPY . .

# Ports expose karo
EXPOSE 8000 8501

# Default command — docker-compose override karega
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]