FROM python:3.10-slim

WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade tooling ----
RUN pip install --upgrade pip setuptools wheel

# ---- Install PyTorch FIRST (CPU-only) ----
RUN pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install Python deps ----
RUN pip install --no-cache-dir -r requirements.txt || true

# ---- Install Detectron2 LAST ----
RUN pip install \
    --no-build-isolation \
    --no-cache-dir \
    git+https://github.com/facebookresearch/detectron2.git

# ---- Copy app ----
COPY . .

EXPOSE 8001

CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio", "--workers", "1"]
