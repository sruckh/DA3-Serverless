#!/bin/bash
set -euo pipefail

# Define workspace directory
WORKSPACE="/workspace/DA3"

echo "Starting DA3-Serverless bootstrap..."
echo "Workspace: ${WORKSPACE}"

# 1) Check if /workspace/DA3 exists, if not create it
if [ ! -d "${WORKSPACE}" ]; then
    echo "Creating workspace directory..."
    mkdir -p "${WORKSPACE}"
fi

# 2) Switch to the directory
cd "${WORKSPACE}"
echo "Changed to: $(pwd)"

# Setup environment variables
export HF_HOME="${HF_HOME:-${WORKSPACE}/cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${WORKSPACE}/cache}"
export WORKSPACE="${WORKSPACE}"

# Create necessary directories only if they don't exist
if [ ! -d "${HF_HOME}" ]; then
    mkdir -p "${HF_HOME}"
fi

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

    # Upgrade pip
    pip install --upgrade pip

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

    # Create src directory if it doesn't exist and copy handler
    if [ ! -d "${WORKSPACE}/src" ]; then
        mkdir -p "${WORKSPACE}/src"
    fi
    cp /workspace/DA3-Serverless/src/handler.py "${WORKSPACE}/src/"

else
    echo "Virtual environment already exists, skipping installation"

    # Activate venv
    source "${WORKSPACE}/venv/bin/activate"
fi

# 5) Check if models have been downloaded, if not download models
DEFAULT_MODEL="depth-anything/DA3NESTED-GIANT-LARGE"
echo "Checking if model is downloaded: ${DEFAULT_MODEL}"

# Function to check if model is downloaded
check_model_downloaded() {
    local model_id="$1"
    local cache_dir="${HF_HOME}/hub/${model_id/--/---}"
    if [ -d "$cache_dir" ] && [ "$(ls -A "$cache_dir" 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

if ! check_model_downloaded "$DEFAULT_MODEL"; then
    echo "Model not found, downloading..."

    # Set HF_TOKEN if available
    if [ -n "${HF_TOKEN:-}" ]; then
        export HF_TOKEN="$HF_TOKEN"
        echo "Using HuggingFace token for authentication"
    fi

    # Download model
    python -c "
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get('DEFAULT_MODEL')
print(f'Downloading {model_id}...')
snapshot_download(
    repo_id=model_id,
    cache_dir=os.environ['HF_HOME'],
    token=os.environ.get('HF_TOKEN'),
    resume_download=True
)
print('Download complete!')
"
else
    echo "Model already downloaded"
fi

# 6) Start the API endpoint
echo "Starting DA3-Serverless handler..."
exec python "${WORKSPACE}/src/handler.py"