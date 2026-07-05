FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Dependances systeme (Pillow, matplotlib, git pour unsloth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements-gpu.txt .
RUN pip install --upgrade pip
RUN pip install fastapi "uvicorn[standard]" python-multipart pillow "numpy<2" matplotlib pydantic
RUN pip install unsloth==2026.6.9
RUN pip install unsloth_zoo==2026.6.7

COPY app ./app

EXPOSE 8000

# Un seul worker : le modele reside en memoire (et sur le GPU).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
