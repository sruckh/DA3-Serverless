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

# Import SDTHead utilities (optional - graceful fallback if not available)
try:
    from model.da3_sdt import swap_head_to_sdt
    SDT_AVAILABLE = True
except ImportError:
    swap_head_to_sdt = None
    SDT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_ENV = "DA3_API_KEY"
# Default to MODEL_ID env var if set, otherwise giant nested large
MODEL_ID_DEFAULT = os.environ.get("MODEL_ID", "da3nested-giant-large")
OUTPUT_DIR = WORKSPACE / "output_images"
MAX_PIXELS = 4096 * 4096

# Map short names to HuggingFace Repo IDs to ensure weights are loaded
MODEL_MAP = {
    "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "da3-giant": "depth-anything/DA3-Giant",
    "da3-large": "depth-anything/DA3-Large",
    "da3-small": "depth-anything/DA3-Small",
    "da3metric-large": "depth-anything/DA3METRIC-Large",
}

# SDTHead configuration
USE_SDT_HEAD_DEFAULT = os.environ.get("USE_SDT_HEAD", "false").lower() == "true"
SDT_FUSION_CHANNELS = int(os.environ.get("SDT_FUSION_CHANNELS", "256"))

_model = None
_device = None
_model_config = None  # Tracks (model_id, use_sdt_head) for cache invalidation

def ensure_model_downloaded(model_id):
    """Ensure model is downloaded, download if necessary"""
    # Only for HuggingFace models
    if not model_id.startswith("depth-anything/"):
        return

    try:
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(f"Checking model availability: {model_id}")
        # NOTE: Do not set cache_dir - let huggingface_hub use default
        # RunPod caches models at the platform level using the default HF cache
        snapshot_download(
            repo_id=model_id,
            token=hf_token,
            resume_download=True
        )
        logger.info(f"Model {model_id} is available")
    except Exception as e:
        logger.warning(f"Could not verify model download: {e}")

def load_model(model_id=None, use_sdt_head=None):
    """
    Load DA3 model with optional SDTHead.

    Args:
        model_id: Model identifier (default: MODEL_ID_DEFAULT)
        use_sdt_head: Whether to use SDTHead instead of DPT (default: USE_SDT_HEAD_DEFAULT)

    Returns:
        Tuple of (model, device)
    """
    global _model, _device, _model_config

    model_id = model_id or MODEL_ID_DEFAULT
    if use_sdt_head is None:
        use_sdt_head = USE_SDT_HEAD_DEFAULT

    # Resolve short name to HF repo ID if possible
    # This ensures we use from_pretrained() and actually load weights!
    if model_id in MODEL_MAP:
        logger.info(f"Mapping short name '{model_id}' to HuggingFace repo '{MODEL_MAP[model_id]}'")
        model_id = MODEL_MAP[model_id]

    current_config = (model_id, use_sdt_head)

    # Clear cached model if configuration changed
    if _model is not None and _model_config != current_config:
        logger.info(f"Model configuration changed from {_model_config} to {current_config}, clearing cache")
        del _model
        _model = None
        _device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if DepthAnything3 is None:
            raise ImportError("DepthAnything3 not available")

        logger.info(f"Loading DA3 model: {model_id} (use_sdt_head={use_sdt_head})")

        # Ensure model is downloaded for HuggingFace models
        ensure_model_downloaded(model_id)

        # DA3 uses model_name instead of from_pretrained for local models
        # For HuggingFace models, use the full path to call from_pretrained
        if model_id.startswith("depth-anything/"):
            _model = DepthAnything3.from_pretrained(model_id)
        else:
            logger.warning(f"Loading model '{model_id}' without weights (random init) because it's not a known HF path.")
            _model = DepthAnything3(model_name=model_id)
        _model = _model.to(device=_device)

        # Swap head to SDTHead if requested
        if use_sdt_head:
            if not SDT_AVAILABLE:
                raise ImportError("SDTHead not available. Ensure model package is properly installed.")
            logger.info(f"Swapping head to SDTHead (fusion_channels={SDT_FUSION_CHANNELS})")
            _model = swap_head_to_sdt(
                _model,
                output_dim=2,  # depth + confidence
                fusion_channels=SDT_FUSION_CHANNELS,
                use_sky_head=True,
            )

        _model.eval()

        # Cache the configuration
        _model_config = current_config

        # Identify head type for logging
        if hasattr(_model, 'model'):
            inner_net = _model.model
            if hasattr(inner_net, 'da3'):
                head_obj = inner_net.da3.head
            else:
                head_obj = getattr(inner_net, 'head', None)
        else:
            head_obj = getattr(_model, 'head', None)
            
        head_type = type(head_obj).__name__ if head_obj else "Unknown"
        logger.info(f"DA3 model loaded successfully on {_device}. Active head: {head_type}")
        
        return _model, _device

    except Exception as e:
        logger.error(f"Failed to load DA3 model: {e}")
        raise

def check_api_key(job):
    """Validate API key"""
    expected = os.environ.get(API_KEY_ENV)
    
    # If no app-level key is configured, assume platform auth (RunPod) is sufficient
    if not expected:
        return

    supplied = None
    headers = job.get("http", {}).get("headers") if job.get("http") else None
    if headers:
        supplied = headers.get("da3-api-key") or headers.get("DA3-API-KEY")
    if not supplied:
        supplied = job.get("input", {}).get("api_key")
    
    if supplied != expected:
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


def apply_histogram_equalization(img_array, clip_limit=0.0):
    """Apply histogram equalization to improve tonal distribution.
    
    Args:
        img_array: 2D numpy array with values in [0, 1] range
        clip_limit: If > 0, applies contrast limiting (similar to CLAHE).
                   Values typically 0.01-0.03. 0 = standard histogram eq.
    
    Returns:
        Equalized array with values in [0, 1] range
    """
    # Convert to 8-bit for histogram computation
    img_8bit = (img_array * 255).astype(np.uint8)
    
    # Compute histogram
    hist, bins = np.histogram(img_8bit.flatten(), bins=256, range=(0, 256))
    
    # Apply contrast limiting if specified
    if clip_limit > 0:
        clip_threshold = clip_limit * img_8bit.size / 256
        excess = np.sum(np.maximum(hist - clip_threshold, 0))
        hist = np.minimum(hist, clip_threshold)
        # Redistribute excess uniformly
        hist += excess / 256
    
    # Compute CDF
    cdf = hist.cumsum()
    
    # Normalize CDF to [0, 255]
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-8) * 255
    
    # Map original values through equalized CDF
    equalized = cdf_normalized[img_8bit].astype(np.float32) / 255.0
    
    return equalized


def apply_gamma_correction(img_array, gamma):
    """Apply gamma correction to expand or compress tonal range.
    
    Args:
        img_array: 2D numpy array with values in [0, 1] range
        gamma: Gamma value. < 1 brightens midtones, > 1 darkens midtones.
    
    Returns:
        Gamma-corrected array with values in [0, 1] range
    """
    return np.power(img_array, gamma)

def run_inference(job):
    """Run DA3 inference"""
    check_api_key(job)
    job_input = job.get("input") or {}
    job_id = job.get("id") or job.get("requestId") or "unknown_job"

    # Load image
    img = load_image_from_job(job_input)
    original_size = img.size  # (Width, Height)

    # Load model with optional SDTHead
    model_id = job_input.get("model_id", MODEL_ID_DEFAULT)
    use_sdt_head = job_input.get("use_sdt_head", None)  # None = use default from env
    model, device = load_model(model_id, use_sdt_head=use_sdt_head)

    # Identify active head for response
    if hasattr(model, 'model'):
        inner_net = model.model
        if hasattr(inner_net, 'da3'):
            head_obj = inner_net.da3.head
        else:
            head_obj = getattr(inner_net, 'head', None)
    else:
        head_obj = getattr(model, 'head', None)
    head_type = type(head_obj).__name__ if head_obj else "Unknown"

    # Convert image for DA3
    # DA3 expects file paths or numpy arrays
    images = [np.array(img)]

    # Prepare inference parameters
    inference_kwargs = {}
    
    # Allow user to control processing resolution (default 504)
    # Higher values = more detail, slower
    if "processing_res" in job_input:
        inference_kwargs["process_res"] = int(job_input["processing_res"])

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
    
    # 1. Resize back to original resolution
    # Convert to PIL for high-quality resizing
    depth_pil = Image.fromarray(depth)
    depth_pil = depth_pil.resize(original_size, resample=Image.BICUBIC)
    depth = np.array(depth_pil)

    # === DEPTH MAP NORMALIZATION PIPELINE ===
    # Configurable parameters for tonal range optimization
    
    # Percentile clipping bounds (wider defaults for better tonal range)
    clip_low = float(job_input.get("clip_percentile_low", 0.5))
    clip_high = float(job_input.get("clip_percentile_high", 99.5))
    
    # Gamma correction (< 1 brightens midtones, > 1 darkens)
    gamma = float(job_input.get("gamma", 1.0))
    
    # Histogram equalization (improves tonal distribution)
    use_hist_eq = job_input.get("histogram_equalization", True)
    hist_clip_limit = float(job_input.get("hist_clip_limit", 0.01))
    
    # 2. Use Inverse Depth for better visualization contrast
    # Metric depth is linear (0..100m), which makes near objects look dark/flat.
    # Inverse depth (1/d) emphasizes near structure.
    inv_depth = 1.0 / (depth + 1e-6)
    
    # 3. Robust normalization using percentiles on INVERSE depth
    viz_min = float(np.percentile(inv_depth, clip_low))
    viz_max = float(np.percentile(inv_depth, clip_high))
    
    # Clip and normalize to [0, 1]
    inv_depth_clipped = np.clip(inv_depth, viz_min, viz_max)
    norm_depth = (inv_depth_clipped - viz_min) / (viz_max - viz_min + 1e-8)
    
    # 4. Apply histogram equalization for better tonal distribution
    if use_hist_eq:
        norm_depth = apply_histogram_equalization(norm_depth, clip_limit=hist_clip_limit)
    
    # 5. Apply gamma correction to fine-tune midtone response
    if gamma != 1.0:
        norm_depth = apply_gamma_correction(norm_depth, gamma)
    
    # Create 8-bit grayscale PNG
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

    # Get raw min/max from original metric depth for stats
    depth_min = float(depth.min())
    depth_max = float(depth.max())

    logger.info(
        "job=%s saved=%s head=%s res=%s depth_range=%.2fm-%.2fm hist_eq=%s gamma=%.2f",
        job_id, png_path, head_type, original_size, depth_min, depth_max,
        use_hist_eq, gamma
    )

    return {
        "image_base64": b64,
        "min_depth": depth_min,
        "max_depth": depth_max,
        "head_type": head_type,
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
    parser.add_argument("--use-sdt-head", action="store_true", help="Use SDTHead instead of DPT")
    args, _ = parser.parse_known_args()

    if args.warmup:
        try:
            # Use SDTHead from CLI arg or environment variable
            use_sdt = args.use_sdt_head or USE_SDT_HEAD_DEFAULT
            logger.info(f"Warming up model (use_sdt_head={use_sdt}, SDT_AVAILABLE={SDT_AVAILABLE})")
            load_model(use_sdt_head=use_sdt)
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
        sys.exit(0)

    runpod.serverless.start({"handler": handler})