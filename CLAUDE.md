# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🛑 STOP — Run codemap before ANY task

```bash
codemap .                     # Project structure
codemap --deps                # How files connect
codemap --diff                # What changed vs main
codemap --diff --ref <branch> # Changes vs specific branch
```

## Required Usage

**BEFORE starting any task**, run `codemap .` first.

**ALWAYS run `codemap --deps` when:**
- User asks how something works
- Refactoring or moving code
- Tracing imports or dependencies

**ALWAYS run `codemap --diff` when:**
- Reviewing or summarizing changes
- Before committing code
- User asks what changed
- Use `--ref <branch>` when comparing against something other than main

## Project Overview

DA3 Serverless packages the Depth Anything 3 (DA3) model as a RunPod serverless endpoint that accepts an image and returns a metric depth map. It's a Docker-based service that runs on GPU-enabled containers with CUDA support.

## Common Development Commands

### Building and Running
```bash
# Build Docker image
docker build -t da3-serverless .

# Run container with GPU support
docker run --gpus all -p 8080:8080 da3-serverless

# Run with volume mount for persistence
docker run --gpus all -p 8080:8080 -v /runpod-volume:/runpod-volume da3-serverless
```

### Testing the Handler
```bash
# Test model warmup (from within container)
python src/handler.py --warmup

# Prefetch models locally
./src/prefetch.sh
```

### Environment Setup
Required environment variables:
- `DA3_API_KEY` - Required for all API calls
- `HF_HOME` - HuggingFace cache directory (defaults to workspace/cache)
- `MODEL_ID` - DA3 model to use (default: depth-anything/DA3NESTED-GIANT-LARGE)
- `WORKSPACE` - Workspace directory (affects where models/cache are stored)

## Architecture

### Core Components

1. **handler.py** - Main RunPod serverless handler
   - Validates API key via header or input field
   - Accepts image via URL or base64
   - Enforces max resolution of 4096x4096
   - Runs DA3 inference and returns depth map as PNG
   - Supports multiple DA3 model variants

2. **bootstrap.sh** - Container initialization script
   - Sets up workspace (prefers /runpod-volume, falls back to /workspace)
   - Creates Python venv and installs dependencies
   - Installs PyTorch with CUDA support
   - Installs DA3 from local source
   - Warms up model handler before starting

3. **prefetch.sh** - Model pre-downloading utility
   - Downloads DA3 models using huggingface_hub
   - Stores models in workspace/models/ directory
   - Useful for preloading models to avoid download delays

### Directory Structure (Runtime)
```
<WORKSPACE>/          # /runpod-volume/DA3-Serverless or /workspace/DA3-Serverless
├── venv/            # Python virtual environment
├── src/             # Handler and scripts
├── cache/           # HF_HOME cache directory
├── models/          # Manual model snapshots
├── upstream/        # DA3 source code (cloned at build time)
└── output_images/   # Generated depth maps (auto-pruned after 14 days)
```

### API Contract
Endpoint: `/runsync` (synchronous)

Input:
```json
{
  "input": {
    "api_key": "YOUR_KEY",
    "image_url": "https://example.com/sample.jpg",
    "model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
    "output_metric": true,
    "use_ray_pose": false
  }
}
```

Output:
```json
{
  "image_base64": "...png...",
  "min_depth": 0.5,
  "max_depth": 10.2,
  "file_path": "/runpod-volume/DA3-Serverless/output_images/<uuid>.png"
}
```

## CI/CD
GitHub Actions workflow builds and pushes Docker image on pushes to `main` branch:
- Builds image with CUDA 12.8.1 base
- Publishes to Docker Hub as `yourname/da3-serverless`
- Uses GitHub Actions caching for faster builds
