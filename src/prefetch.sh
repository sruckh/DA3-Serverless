#!/bin/bash
set -euo pipefail

# Script to pre-download DA3 models for the DA3-Serverless project

WORKSPACE="${WORKSPACE:-/workspace/DA3}"
MODEL_ID="${MODEL_ID:-depth-anything/DA3NESTED-GIANT-LARGE}"
HF_HOME="${HF_HOME:-${WORKSPACE}/cache}"

echo "Prefetching DA3 model: ${MODEL_ID}"
echo "Workspace: ${WORKSPACE}"
echo "HF_HOME: ${HF_HOME}"

# Create workspace if it doesn't exist
mkdir -p "${WORKSPACE}"

# Setup environment
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export WORKSPACE="${WORKSPACE}"

# Create directories only if they don't exist
if [ ! -d "${HF_HOME}" ]; then
    mkdir -p "${HF_HOME}"
fi

if [ ! -d "${WORKSPACE}/models" ]; then
    mkdir -p "${WORKSPACE}/models"
fi

# Set HF_TOKEN if available
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "Using HuggingFace token for authentication"
fi

# Function to check if model is already downloaded
check_model_downloaded() {
    local model_id="$1"
    local cache_dir="${HF_HOME}/hub/${model_id/--/---}"
    if [ -d "$cache_dir" ] && [ "$(ls -A "$cache_dir" 2>/dev/null)" ]; then
        echo "Model ${model_id} is already cached"
        return 0
    else
        return 1
    fi
}

# Download the model if not already cached
if check_model_downloaded "$MODEL_ID"; then
    echo "Model already downloaded"
else
    echo "Downloading HuggingFace model: ${MODEL_ID}"
    export model_id="$MODEL_ID"

    python -c "
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get('model_id')
print(f'Downloading {model_id}...')
snapshot_download(
    repo_id=model_id,
    cache_dir=os.environ['HF_HOME'],
    token=os.environ.get('HF_TOKEN'),
    resume_download=True
)
print('Download complete!')
"
fi

echo "Prefetch complete!"