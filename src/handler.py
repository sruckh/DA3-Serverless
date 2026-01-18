import argparse
import base64
import io
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import runpod
import torch
from PIL import Image

# Ensure DA3 repo is importable
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace/DA3"))
UPSTREAM = WORKSPACE / "upstream"
SRC = WORKSPACE / "src"

# Add paths to sys.path
for p in (str(UPSTREAM), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import DA3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    # Fallback if not installed yet
    DepthAnything3 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_ENV = "DA3_API_KEY"
MODEL_ID_DEFAULT = "da3nested-giant-large"
OUTPUT_DIR = WORKSPACE / "output_images"
MAX_PIXELS = 4096 * 4096

_model = None
_device = None

def ensure_model_downloaded(model_id):
    """Ensure model is downloaded, download if necessary"""
    # Only for HuggingFace models
    if not model_id.startswith("depth-anything/"):
        return

    try:
        from huggingface_hub import snapshot_download
        hf_home = os.environ.get("HF_HOME", f"{WORKSPACE}/cache")
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(f"Checking model availability: {model_id}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=hf_home,
            token=hf_token,
            resume_download=True
        )
        logger.info(f"Model {model_id} is available")
    except Exception as e:
        logger.warning(f"Could not verify model download: {e}")

def load_model(model_id=None):
    """Load DA3 model"""
    global _model, _device

    # Clear cached model if different model_id is requested
    if _model is not None and model_id and model_id != getattr(load_model, '_last_model_id', None):
        logger.info("Different model requested, clearing cache")
        _model = None
        _device = None

    if _model is not None:
        return _model, _device

    model_id = model_id or MODEL_ID_DEFAULT
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if DepthAnything3 is None:
            raise ImportError("DepthAnything3 not available")

        logger.info(f"Loading DA3 model: {model_id}")

        # Ensure model is downloaded for HuggingFace models
        ensure_model_downloaded(model_id)

        # DA3 uses model_name instead of from_pretrained for local models
        # For HuggingFace models, use the full path
        if model_id.startswith("depth-anything/"):
            _model = DepthAnything3.from_pretrained(model_id)
        else:
            _model = DepthAnything3(model_name=model_id)
        _model = _model.to(device=_device)
        _model.eval()

        # Cache the model_id for future checks
        load_model._last_model_id = model_id

        logger.info(f"DA3 model loaded successfully on {_device}")
        return _model, _device

    except Exception as e:
        logger.error(f"Failed to load DA3 model: {e}")
        raise

def check_api_key(job):
    """Validate API key"""
    supplied = None
    headers = job.get("http", {}).get("headers") if job.get("http") else None
    if headers:
        supplied = headers.get("da3-api-key") or headers.get("DA3-API-KEY")
    if not supplied:
        supplied = job.get("input", {}).get("api_key")
    expected = os.environ.get(API_KEY_ENV)
    if not expected or supplied != expected:
        raise PermissionError("Unauthorized")

def load_image_from_job(job_input):
    """Load image from URL or base64"""
    if "image_url" in job_input:
        url = job_input["image_url"]
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.content
        # Try to determine image format from URL or content-type
        content_type = resp.headers.get('content-type', '').lower()
    elif "image_base64" in job_input:
        # Check for data URI format (e.g., "data:image/jpeg;base64,....")
        b64_data = job_input["image_base64"]
        if b64_data.startswith('data:'):
            # Extract the base64 part after the comma
            try:
                # Split on comma and take the second part
                b64_parts = b64_data.split(',', 1)
                if len(b64_parts) == 2:
                    b64_data = b64_parts[1]
                # Extract content type from the data URI
                if 'image/' in b64_parts[0]:
                    content_type = b64_parts[0].split(';')[0].lower()
                else:
                    content_type = ''
            except:
                b64_data = job_input["image_base64"]
                content_type = ''

        try:
            data = base64.b64decode(b64_data)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")
    else:
        raise ValueError("No image provided. Use 'image_url' or 'image_base64'.")

    # Load image with PIL - supports many formats including:
    # JPEG, PNG, WebP, BMP, TIFF, GIF (first frame), ICO, etc.
    try:
        img = Image.open(io.BytesIO(data))

        # Log the detected format
        logger.info(f"Loaded image format: {img.format}, size: {img.size}, mode: {img.mode}")

        # Convert to RGB for DA3 processing
        # This handles various image modes (RGBA, L, P, etc.)
        if img.mode != 'RGB':
            logger.debug(f"Converting image from {img.mode} to RGB")
            img = img.convert('RGB')

    except Exception as e:
        raise ValueError(f"Failed to load image: {e}. Supported formats: JPEG, PNG, WebP, BMP, TIFF, GIF, ICO, etc.")

    # Check image size
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f"Image too large ({img.width}x{img.height}). Max resolution: 4096x4096 pixels.")

    return img

def run_inference(job):
    """Run DA3 inference"""
    check_api_key(job)
    job_input = job.get("input") or {}
    job_id = job.get("id") or job.get("requestId") or "unknown_job"

    # Load image
    img = load_image_from_job(job_input)

    # Load model
    model_id = job_input.get("model_id", MODEL_ID_DEFAULT)
    model, device = load_model(model_id)

    # Convert image for DA3
    # DA3 expects file paths or numpy arrays
    images = [np.array(img)]

    # Prepare inference parameters
    inference_kwargs = {}

    # Add optional parameters if provided
    if "use_ray_pose" in job_input:
        inference_kwargs["use_ray_pose"] = job_input["use_ray_pose"]

    # Run inference
    with torch.no_grad():
        prediction = model.inference(
            images,
            **inference_kwargs
        )

    # Extract depth map - DA3 returns depth as [N, H, W] numpy array
    depth = prediction.depth[0]  # First image
    depth_min, depth_max = float(depth.min()), float(depth.max())

    # Normalize for visualization (0-255)
    norm_depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    png = Image.fromarray((norm_depth * 255).astype(np.uint8), mode="L")

    # Save image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    png_path = OUTPUT_DIR / fname
    png.save(png_path)

    # Convert to base64
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    logger.info(
        "job=%s saved=%s min_depth=%.3f max_depth=%.3f",
        job_id, png_path, depth_min, depth_max
    )

    return {
        "image_base64": b64,
        "min_depth": depth_min,
        "max_depth": depth_max,
        "file_path": str(png_path)
    }

def handler(job):
    """Main handler for RunPod serverless"""
    try:
        return run_inference(job)
    except PermissionError as e:
        return {"error": str(e), "status": 401}
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": str(e), "status": 400}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    args, _ = parser.parse_known_args()

    if args.warmup:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
        sys.exit(0)

    runpod.serverless.start({"handler": handler})