# Use an official PyTorch image with CUDA (for GPU) or Python base (for CPU)
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime  # For GPU
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    curl \
    libgl1-mesa-glx \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip install torch torchvision

# Install Detectron2 (recommended way)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy project files
COPY . .

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8001

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
