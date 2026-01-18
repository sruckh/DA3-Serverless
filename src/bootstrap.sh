#!/bin/bash
set -euo pipefail

echo "Starting DA3-Serverless bootstrap..."

# Define workspace directory - use network volume if available for persistence
if [ -d "/runpod-volume" ]; then
    WORKSPACE="/runpod-volume/workspace/DA3"
    export HF_HOME="/runpod-volume/huggingface"
    echo "Using network volume: ${WORKSPACE}"
else
    WORKSPACE="/workspace/DA3"
    echo "Using ephemeral storage: ${WORKSPACE}"
fi

# Create workspace if needed
if [ ! -d "${WORKSPACE}" ]; then
    echo "Creating workspace directory..."
    mkdir -p "${WORKSPACE}"
fi

# Create HF_HOME if set
if [ -n "${HF_HOME:-}" ]; then
    mkdir -p "${HF_HOME}"
    echo "HuggingFace cache: ${HF_HOME}"
fi

# Switch to workspace
cd "${WORKSPACE}"
echo "Changed to: $(pwd)"

export WORKSPACE="${WORKSPACE}"

if [ ! -d "${WORKSPACE}/output_images" ]; then
    mkdir -p "${WORKSPACE}/output_images"
fi

# 3) Check if DA3 repo exists, if not clone it
if [ ! -d "upstream" ]; then
    echo "Cloning Depth Anything 3 repository..."
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git upstream
else
    echo "DA3 repository already exists, skipping clone"
fi

# 4) Check if venv exists, if not create and install dependencies
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    /usr/bin/python3.12 -m venv "${WORKSPACE}/venv"

    # Activate venv
    source "${WORKSPACE}/venv/bin/activate"

    # Upgrade pip and install essential packages
    pip install --upgrade pip
    pip install huggingface_hub

    # Install PyTorch with CUDA 12.8 support first
    echo "Installing PyTorch with CUDA 12.8 support..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

    # Check if DA3 has requirements and filter out torch/torchvision to avoid conflicts
    if [ -f "${WORKSPACE}/upstream/requirements.txt" ]; then
        echo "Found DA3 requirements.txt, filtering out torch/torchvision to avoid conflicts"
        # Filter out torch-related packages from DA3 requirements
        grep -v "^torch" "${WORKSPACE}/upstream/requirements.txt" > "${WORKSPACE}/requirements_filtered.txt" || true
        grep -v "^torchvision" "${WORKSPACE}/requirements_filtered.txt" > "${WORKSPACE}/requirements_final.txt" || true
        # Install filtered requirements first
        pip install -r "${WORKSPACE}/requirements_final.txt"
    fi

    # Install DA3 from local source (without torch/torchvision in requirements)
    echo "Installing Depth Anything 3..."
    cd "${WORKSPACE}/upstream"
    # Create a temporary requirements file without torch for the install
    if [ -f "requirements.txt" ]; then
        # Temporarily move requirements.txt to avoid torch conflicts
        mv requirements.txt requirements.txt.backup
        pip install -e .
        # Restore requirements.txt
        mv requirements.txt.backup requirements.txt
    else
        pip install -e .
    fi

    # Install gsplat for 3D Gaussian head
    echo "Installing gsplat..."
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

    cd "${WORKSPACE}"

    # Create src directory if it doesn't exist and copy handler + model
    if [ ! -d "${WORKSPACE}/src" ]; then
        mkdir -p "${WORKSPACE}/src"
    fi
    cp /workspace/DA3-Serverless/src/handler.py "${WORKSPACE}/src/"

    # Copy model directory for SDTHead support
    if [ -d "/workspace/DA3-Serverless/src/model" ]; then
        echo "Copying SDTHead model module..."
        cp -r /workspace/DA3-Serverless/src/model "${WORKSPACE}/src/"
    fi

else
    echo "Virtual environment already exists, skipping installation"

    # Activate venv
    source "${WORKSPACE}/venv/bin/activate"
fi

# 5) Download model if needed (RunPod handles caching at platform level)
DEFAULT_MODEL="depth-anything/DA3NESTED-GIANT-LARGE"
echo "Ensuring model is available: ${DEFAULT_MODEL}"

# Set HF_TOKEN if available
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "Using HuggingFace token for authentication"
fi

# Download model using huggingface_hub (uses default HF cache, cached by RunPod)
python -c "
import os
from huggingface_hub import snapshot_download

model_id = '${DEFAULT_MODEL}'
print(f'Checking/downloading {model_id}...')
snapshot_download(
    repo_id=model_id,
    token=os.environ.get('HF_TOKEN'),
    resume_download=True
)
print('Model ready!')
"

# 6) Start the API endpoint
echo "Starting DA3-Serverless handler..."
exec python "${WORKSPACE}/src/handler.py"