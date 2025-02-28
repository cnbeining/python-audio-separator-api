# Use the latest CUDA 12 runtime as base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /workdir

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip fastapi litserve python-multipart

# Install the CUDA 12 compatible version of ONNXRuntime (the default CUDA version for ORT is still 11.8 so they've provided a separate package index)
# See https://onnxruntime.ai/docs/install/
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install audio-separator without any specific onnxruntime (onnxruntime should already be satisfied by the above)
RUN --mount=type=cache,target=/root/.cache \
    pip3 install "audio-separator" 

COPY . /app

WORKDIR /app

# force the model to be downloaded
RUN audio-separator --download_model_only

CMD ["python3", "main.py"]