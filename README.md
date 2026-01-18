# DA3 Serverless

[![Docker Build](https://img.shields.io/badge/docker-build-blue)](https://hub.docker.com/r/yourname/da3-serverless)
[![CUDA](https://img.shields.io/badge/CUDA-12.8.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> GPU-accelerated Depth Anything 3 (DA3) model as a RunPod serverless endpoint for metric depth estimation.

DA3 Serverless packages the Depth Anything 3 model into a production-ready Docker container that runs on GPU-enabled infrastructure. It accepts images via URL or base64 encoding and returns metric depth maps with accurate depth measurements in meters.

## ✨ Features

- **Production-Ready API**: RunPod serverless endpoint with synchronous execution (`/runsync`)
- **Multiple Input Formats**: Accepts images via URL or base64 (including data URI format)
- **Format Support**: JPEG, PNG, WebP, BMP, TIFF, GIF, ICO, and more
- **GPU Acceleration**: CUDA 12.8.1 support with PyTorch 2.8.0
- **Model Variants**: Support for multiple DA3 model architectures (Small, Large, Giant, Nested-Giant)
- **SDTHead Support**: Optional Stable Depth Transformer head with DySample upsampling for enhanced detail preservation
- **Auto-Conversion**: Handles various image modes (RGBA, L, P, etc.) with automatic RGB conversion
- **Metric Depth**: Returns accurate depth measurements in meters
- **Secure**: API key authentication via headers or input fields
- **RunPod Model Caching**: Uses RunPod's platform-level model caching for fast cold starts

## 🏗️ Architecture

![Architecture Diagram](./docs/diagrams/architecture.svg)

The system is designed as a layered Docker-based service:

1. **Infrastructure Layer**: CUDA-enabled GPU container with Ubuntu 24.04 and Python 3.12
2. **Application Layer**: RunPod serverless handler with DA3 model inference
3. **Data Layer**: HuggingFace model cache, output images, and optional persistent volume
4. **External Services**: HuggingFace Hub for model downloads, RunPod for serverless execution

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- NVIDIA GPU with CUDA 12.8 capability
- 16GB+ GPU RAM recommended for large models
- HuggingFace account (for model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DA3-Serverless.git
cd DA3-Serverless

# Build the Docker image
docker build -t da3-serverless .
```

### Running the Container

```bash
# Run with GPU support (recommended for production)
docker run --gpus all -p 8080:8080 \
  -e DA3_API_KEY="your-secret-api-key" \
  da3-serverless

# Run with persistent volume for model caching
docker run --gpus all -p 8080:8080 \
  -e DA3_API_KEY="your-secret-api-key" \
  -v /runpod-volume:/runpod-volume \
  da3-serverless
```

### Pre-fetching Models

To avoid first-run delays, pre-download models locally:

```bash
# From within the container or after setup
./src/prefetch.sh

# Or with environment variables
MODEL_ID="depth-anything/DA3NESTED-GIANT-LARGE" \
HF_TOKEN="your-hf-token" \
./src/prefetch.sh
```

## 📖 Documentation

### API Reference

#### Endpoint: `/runsync` (Synchronous)

**Authentication**: Provide API key via header or input field
- Header: `da3-api-key` or `DA3-API-KEY`
- Input field: `input.api_key`

#### Input Schema

```typescript
{
  input: {
    // Required: API authentication key
    api_key: string;

    // Required: Image source (one of)
    image_url?: string;      // Public URL to image
    image_base64?: string;   // Base64 encoded image (with or without data URI prefix)

    // Optional: Model selection
    model_id?: string;       // Default: "da3nested-giant-large"
                             // Options: "da3-small", "da3-large", "da3-giant",
                             //          "da3nested-giant-large", "da3metric-large"

    // Optional: Inference parameters
    use_ray_pose?: boolean;  // Use ray-based pose estimation (default: false)
    use_sdt_head?: boolean;  // Use SDTHead instead of DPT (default: false or USE_SDT_HEAD env)
  }
}
```

#### Output Schema

```typescript
{
  // Grayscale PNG as base64 (8-bit, 0-255)
  image_base64: string;

  // Depth statistics in meters
  min_depth: number;        // Minimum depth value
  max_depth: number;        // Maximum depth value

  // File path (when using persistent volume)
  file_path: string;        // Path to saved PNG on volume

  // Error responses
  error?: string;           // Error message (if failed)
  status?: number;          // HTTP status code (if failed)
}
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DA3_API_KEY` | Required authentication key for API access | *none* |
| `HF_TOKEN` | HuggingFace token for private models | *none* |
| `MODEL_ID` | Default DA3 model to load | `da3nested-giant-large` |
| `USE_SDT_HEAD` | Use SDTHead instead of DPT head | `false` |
| `SDT_FUSION_CHANNELS` | SDTHead fusion channel dimension | `256` |
| `WORKSPACE` | Workspace root directory | `/workspace/DA3` |

> **Note**: HuggingFace cache location is managed by RunPod at the platform level. Do not override `HF_HOME` as this will break RunPod's model caching.

### Model Variants

| Model ID | Description | GPU Memory | Speed | Accuracy |
|----------|-------------|------------|-------|----------|
| `da3-small` | Fast inference, lower accuracy | ~4GB | ⚡⚡⚡ | ★★☆ |
| `da3-large` | Balanced performance | ~8GB | ⚡⚡ | ★★★ |
| `da3-giant` | High accuracy | ~12GB | ⚡ | ★★★★ |
| `da3nested-giant-large` | State-of-the-art metric depth | ~16GB | ⚡ | ★★★★★ |
| `da3metric-large` | Metric-optimized variant | ~10GB | ⚡⚡ | ★★★★ |

### SDTHead (Stable Depth Transformer Head)

SDTHead is an alternative decoder head adapted from [AnythingDepth](https://github.com/AnythingDepth/AnythingDepth) that can replace the default DPT head. It offers:

- **DySample Upsampling**: Dynamic content-aware upsampling instead of bilinear interpolation
- **Weighted Multi-Scale Fusion**: Learnable importance weights for feature fusion
- **Spatial Detail Enhancement**: Depthwise convolution for better edge preservation

**When to use SDTHead:**
- When you need better fine detail preservation
- For scenes with complex edges and textures
- When experimenting with alternative architectures

**Enable via environment variable:**
```bash
USE_SDT_HEAD=true
```

**Enable via API parameter:**
```json
{
  "input": {
    "image_url": "...",
    "use_sdt_head": true
  }
}
```

> **Note**: SDTHead weights are randomly initialized when swapping from a pre-trained DPT model. For best results, the head should be fine-tuned on depth datasets (not included in this release).

### Directory Structure (Runtime)

```
<WORKSPACE>/          # /workspace/DA3
├── venv/            # Python virtual environment with dependencies
├── src/             # Handler and scripts
│   ├── handler.py   # Main RunPod serverless handler
│   └── model/       # SDTHead module (optional)
│       ├── sdt_head.py          # SDTHead implementation
│       ├── sdt_head_adapter.py  # DA3 interface adapter
│       └── da3_sdt.py           # Head swapping utilities
├── upstream/        # DA3 source code (cloned at runtime)
└── output_images/   # Generated depth maps

# HuggingFace models cached by RunPod at platform level (default HF_HOME)
```

## 🧪 Testing

### Test Model Warmup

```bash
# Warmup with default DPT head
python src/handler.py --warmup

# Warmup with SDTHead
python src/handler.py --warmup --use-sdt-head
```

### Test API Call

```bash
# Standard request with DPT head
curl -X POST https://your-runpod-endpoint/runsync \
  -H "Content-Type: application/json" \
  -H "da3-api-key: YOUR_KEY" \
  -d '{
    "input": {
      "image_url": "https://example.com/sample.jpg",
      "model_id": "da3nested-giant-large"
    }
  }'

# Request with SDTHead enabled
curl -X POST https://your-runpod-endpoint/runsync \
  -H "Content-Type: application/json" \
  -H "da3-api-key: YOUR_KEY" \
  -d '{
    "input": {
      "image_url": "https://example.com/sample.jpg",
      "use_sdt_head": true
    }
  }'
```

### Test with Base64 Image

```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 sample.jpg)

# Make request
curl -X POST https://your-runpod-endpoint/runsync \
  -H "Content-Type: application/json" \
  -H "da3-api-key: YOUR_KEY" \
  -d "{
    \"input\": {
      \"image_base64\": \"${IMAGE_BASE64}\",
      \"model_id\": \"da3nested-giant-large\"
    }
  }"
```

## 🔧 Development

### Local Development Setup

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Clone DA3 repository
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git ext_code/Depth-Anything-3

# Install DA3 from source
cd ext_code/Depth-Anything-3
pip install -e .

# Install gsplat for 3D Gaussian Splatting support
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### Building for Production

```bash
# Build optimized image
docker build -t da3-serverless:latest .

# Tag for registry
docker tag da3-serverless:latest yourname/da3-serverless:v1.0.0

# Push to registry
docker push yourname/da3-serverless:v1.0.0
```

## 📊 Data Flow

![Data Flow Diagram](./docs/diagrams/data-flow.svg)

1. **Request Reception**: RunPod receives HTTP POST to `/runsync`
2. **Authentication**: API key validated from header or input field
3. **Image Loading**: Image loaded from URL (HTTP GET) or decoded from base64
4. **Format Validation**: Image format validated and converted to RGB if needed
5. **Resolution Check**: Image size verified (max 4096×4096)
6. **Model Loading**: DA3 model loaded (cached between requests)
7. **Inference**: Depth prediction run on GPU
8. **Output Processing**: Depth map normalized to 8-bit grayscale
9. **Encoding**: PNG encoded to base64
10. **Response**: JSON returned with depth statistics and image data

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code style
- Add tests for new features
- Update documentation for API changes
- Use semantic versioning for releases

## 🐛 Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"
- **Solution**: Use a smaller model variant (e.g., `da3-small` or `da3-large`)

**Issue**: "Model download failed"
- **Solution**: Set `HF_TOKEN` environment variable for HuggingFace authentication

**Issue**: "Image too large"
- **Solution**: Resize input image to max 4096×4096 pixels before sending

**Issue**: "Unauthorized" error
- **Solution**: Ensure `DA3_API_KEY` is set and matches the provided API key

## 📄 License

This project includes code from:

- **Depth Anything 3**: [Apache 2.0 License](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE)
- **DA3 Serverless**: MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Seed Team
- [RunPod](https://www.runpod.io/) for serverless GPU infrastructure
- [HuggingFace](https://huggingface.co/) for model hosting and distribution
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📚 Additional Resources

- [Depth Anything 3 Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [DA3 API Documentation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)
- [RunPod Serverless Docs](https://docs.runpod.io/serverless)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/DA3-Serverless)
- [Docker Hub](https://hub.docker.com/r/yourname/da3-serverless)
- [Issue Tracker](https://github.com/yourusername/DA3-Serverless/issues)
- [Discussions](https://github.com/yourusername/DA3-Serverless/discussions)
