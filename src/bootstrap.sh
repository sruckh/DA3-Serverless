#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1

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
    pip install --upgrade pip
    pip install huggingface_hub
    sync

    # Define constraints to prevent torch upgrades/downgrades and numpy/pillow churn
    # Based on logs showing 2.9.1 is preferred by dependencies and numpy 2.3.5 is pre-installed
    echo "Creating constraints.txt to pin versions..."
    cat > "${WORKSPACE}/constraints.txt" <<EOF
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
numpy==2.3.5
pillow==12.0.0
EOF

    # Install PyTorch with CUDA 12.8 support
    echo "Installing PyTorch 2.9.1 with CUDA 12.8 support..."
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    sync
    
    # Check if DA3 has requirements and filter out problematic packages
    if [ -f "${WORKSPACE}/upstream/requirements.txt" ]; then
        echo "Found DA3 requirements.txt, filtering out torch/torchvision/xformers/triton/numpy/pillow to avoid conflicts"
        # Robustly filter out torch-related packages, xformers/triton, and numpy/pillow.
        grep -vE "^[[:space:]]*(torch|torchvision|torchaudio|xformers|triton|numpy|pillow)" "${WORKSPACE}/upstream/requirements.txt" > "${WORKSPACE}/requirements_filtered.txt" || true
        
        # Install filtered requirements ONE BY ONE to avoid OOM
        echo "Installing filtered requirements individually to prevent OOM..."
        while read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^# ]] && continue
            echo "Installing: $line"
            pip install "$line" \
                --extra-index-url https://download.pytorch.org/whl/cu128 \
                -c "${WORKSPACE}/constraints.txt"
            sync
        done < "${WORKSPACE}/requirements_filtered.txt"
    fi

    # Install DA3 from local source with --no-deps
    echo "Installing Depth Anything 3..."
    cd "${WORKSPACE}/upstream"
    pip install -e . --no-deps
    sync

    # Install gsplat for 3D Gaussian head
    echo "Installing gsplat..."
    # Use --no-deps to prevent it from trying to reinstall torch/numpy
    pip install --no-build-isolation --no-deps git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
    sync

    # Final check of torch version
    echo "Current PyTorch version:"
    python -c "import torch; print(torch.__version__)"
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

# Copy handler and model files (always)
if [ ! -d "${WORKSPACE}/src" ]; then
    mkdir -p "${WORKSPACE}/src"
fi
cp /workspace/DA3-Serverless/src/handler.py "${WORKSPACE}/src/"

if [ -d "/workspace/DA3-Serverless/src/model" ]; then
    echo "Copying SDTHead model module..."
    cp -r /workspace/DA3-Serverless/src/model "${WORKSPACE}/src/"
fi

# Ensure required packages are installed
pip install --quiet huggingface_hub runpod

# 5) Download model if needed
DEFAULT_MODEL="depth-anything/DA3NESTED-GIANT-LARGE"
echo "Ensuring model is available: ${DEFAULT_MODEL}"

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "Using HuggingFace token for authentication"
fi

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