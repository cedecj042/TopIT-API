FROM python:3.10.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    curl \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch and torchvision
RUN pip install torch torchvision

# Install Detectron2 (recommended way)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy project files
COPY . .