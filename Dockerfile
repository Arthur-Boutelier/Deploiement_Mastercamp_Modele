# =====================================================================
# Image Docker pour le deploiement sur Azure
# =====================================================================
# Pour une inference GPU (recommande), utilise une image de base CUDA :
#   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# et installe Python par-dessus. L'image ci-dessous cible un deploiement
# CPU par defaut, plus simple mais nettement plus lent pour Gemma.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Dependances systeme (Pillow, matplotlib, git pour unsloth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

# Un seul worker : le modele reside en memoire (et sur le GPU).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
