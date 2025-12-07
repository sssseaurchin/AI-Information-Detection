# AI-Information-Detection – Docker GPU Workflow Guide

This document provides a complete explanation of how to run, build, and operate the AI-Information-Detection project using Docker on both CPU and GPU.

## 1) Project Structure
```
AI-Information-Detection/
│   docker-compose.yml
│
└── src/
    ├── dockerfile
    ├── requirements.txt
    ├── main.py
    └── other .py files
```

## 2) Environment Overview
All environments run fully inside Docker. No local TensorFlow, CUDA, or Python setup is required.

## 3) CPU Development Mode (Fast Testing)
```
cd C:\Users\okkah\Python\AI-Information-Detection

docker compose run --rm aid-dev
```
Stops with CTRL+C or:
```
docker ps
docker kill <id>
```

## 4) GPU Mode (Main Workflow)
### GPU Debug Shell
```
docker run --gpus all --rm -it ai-information-detection:dev bash
```
Inside:
```
python3 -m main
```
GPU test:
```
python3 - <<EOF
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
EOF
```
Exit:
```
exit
```

## 5) GPU Direct Run (non-interactive)
```
docker run --gpus all --rm ai-information-detection:dev
```
Runs `python3 -m main` automatically.

## 6) GPU Dockerfile
```
FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "-m", "main"]
```
Includes TensorFlow GPU, CUDA, cuDNN.

## 7) Requirements
```
pandas==2.3.3
numpy==1.26.4
```
TensorFlow is already in the GPU base image.

## 8) Building Images
Normal build:
```
docker compose build aid-dev
```
Clean rebuild:
```
docker compose build --no-cache aid-dev
```

## 9) Stopping Containers
CPU mode:
```
CTRL + C
```
If stuck:
```
docker ps
docker kill <id>
```
GPU shell:
```
exit
```
GPU direct run freeze:
```
docker ps
docker kill <id>
```

## 10) Quick Summary
CPU:
```
docker compose run --rm aid-dev
```
GPU:
```
docker run --gpus all --rm ai-information-detection:dev
```
GPU Debug:
```
docker run --gpus all --rm -it ai-information-detection:dev bash
```
Stop:
```
exit
```
Or:
```
CTRL + C
```
Or:
```
docker kill <id>
```
