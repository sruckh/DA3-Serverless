#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "Starting DA3-Serverless bootstrap..."

# Use WORKSPACE from environment if set, otherwise default to /workspace/DA3
# RunPod typically mounts network volume at /workspace
if [ -z "${WORKSPACE:-}" ]; then
    WORKSPACE="/workspace/DA3"
fi
echo "Workspace: ${WORKSPACE}"

# Set HF_HOME under workspace if not already set
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="${WORKSPACE}/huggingface"
fi
echo "HuggingFace cache: ${HF_HOME}"

# Create directories
mkdir -p "${WORKSPACE}" "${HF_HOME}"

# Setup logging to persistent storage
LOG_FILE="${WORKSPACE}/bootstrap_debug.log"
echo "Enabling persistent logging to: ${LOG_FILE}"
# Redirect stdout and stderr to the log file while still showing in console
# stdbuf -oL -eL helps reduce buffering if available, otherwise standard redirection
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== Bootstrap started at $(date) ==="
sync

# Trap errors to log them
error_handler() {
    echo "ERROR: Bootstrap failed at line $1"
    echo "Last command: $BASH_COMMAND"
    echo "See ${LOG_FILE} for details."
    sync
}
trap 'error_handler ${LINENO}' ERR

# Switch to workspace
cd "${WORKSPACE}"
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

# 4) Check if venv exists and is complete
if [ -d "venv" ] && [ ! -f "venv/.install_complete" ]; then
    echo "Found incomplete virtual environment (missing .install_complete marker). Removing to ensure clean install..."
    rm -rf "venv"
fi

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    /usr/bin/python3.12 -m venv "${WORKSPACE}/venv"
    sync

    # Activate venv
    source "${WORKSPACE}/venv/bin/activate"

    # Upgrade pip and install essential packages
    pip install --upgrade pip --no-cache-dir
    pip install huggingface_hub --no-cache-dir
    sync

    # Install PyTorch with CUDA 12.8 support
    # Split installation to avoid OOM kills during unpacking
    echo "Installing PyTorch (torch only)..."
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    sync
    
    echo "Installing torchvision and torchaudio..."
    pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    sync

    # Check if DA3 has requirements and filter out torch/torchvision to avoid conflicts
    if [ -f "${WORKSPACE}/upstream/requirements.txt" ]; then
        echo "Found DA3 requirements.txt, filtering out torch/torchvision to avoid conflicts"
        # Filter out torch-related packages from DA3 requirements
        grep -v "^torch" "${WORKSPACE}/upstream/requirements.txt" > "${WORKSPACE}/requirements_filtered.txt" || true
        grep -v "^torchvision" "${WORKSPACE}/requirements_filtered.txt" > "${WORKSPACE}/requirements_final.txt" || true
        # Install filtered requirements first
        pip install -r "${WORKSPACE}/requirements_final.txt" --no-cache-dir
        sync
    fi

    # Install DA3 from local source with --no-deps to avoid torch conflicts
    echo "Installing Depth Anything 3..."
    cd "${WORKSPACE}/upstream"
    pip install -e . --no-deps --no-cache-dir
    sync

    # Install gsplat for 3D Gaussian head
    echo "Installing gsplat..."
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 --no-cache-dir
    sync

    # Reinstall PyTorch to fix any corruption from dependency conflicts
    echo "Reinstalling PyTorch to ensure clean installation..."
    pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    sync

    # Mark installation as complete
    touch "${WORKSPACE}/venv/.install_complete"
    sync
    
    cd "${WORKSPACE}"

else
    echo "Virtual environment already exists and is complete, skipping installation"

    # Activate venv
    source "${WORKSPACE}/venv/bin/activate"
fi

# Copy handler and model files (always, to pick up updates)
if [ ! -d "${WORKSPACE}/src" ]; then
    mkdir -p "${WORKSPACE}/src"
fi
cp /workspace/DA3-Serverless/src/handler.py "${WORKSPACE}/src/"

# Copy model directory for SDTHead support
if [ -d "/workspace/DA3-Serverless/src/model" ]; then
    echo "Copying SDTHead model module..."
    cp -r /workspace/DA3-Serverless/src/model "${WORKSPACE}/src/"
fi

# Ensure required packages are installed
pip install --quiet huggingface_hub runpod --no-cache-dir

# 5) Download model if needed
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