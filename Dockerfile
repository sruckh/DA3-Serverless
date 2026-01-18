FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Create a temporary directory for source files
WORKDIR /workspace/DA3-Serverless

# Copy only essential files
COPY src/ ./src/
COPY requirements.txt ./

# Make bootstrap executable
RUN chmod +x ./src/bootstrap.sh

CMD ["bash", "/workspace/DA3-Serverless/src/bootstrap.sh"]