# Use official Python image
FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    git \
    curl \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for efficient caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch, torchvision, and Detectron2 together
RUN pip install --no-cache-dir torch torchvision \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Copy project files last to maximize caching
COPY . .

# Default command (optional, can be overridden in docker-compose)
CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio"]
