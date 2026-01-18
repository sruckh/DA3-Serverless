"""
DA3 with SDTHead Integration

This module provides utilities to create DA3 models with SDTHead
instead of the default DPT head.

Usage:
    from model.da3_sdt import create_da3_with_sdt, swap_head_to_sdt

    # Option 1: Create from scratch with SDTHead
    model = create_da3_with_sdt("da3nested-giant-large")

    # Option 2: Swap head of existing model
    model = DepthAnything3.from_pretrained("depth-anything/DA3...")
    model = swap_head_to_sdt(model)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_backbone_dim(model_name: str) -> int:
    """
    Get the backbone feature dimension for a given DA3 model variant.

    DA3 uses DinoV2 backbones with different dimensions:
    - ViT-Small: 384
    - ViT-Base: 768
    - ViT-Large: 1024
    - ViT-Giant: 1536

    Args:
        model_name: DA3 model name (e.g., 'da3-large', 'da3nested-giant-large')

    Returns:
        Feature dimension of the backbone
    """
    model_name_lower = model_name.lower()

    if 'giant' in model_name_lower:
        return 1536
    elif 'large' in model_name_lower:
        return 1024
    elif 'base' in model_name_lower:
        return 768
    elif 'small' in model_name_lower:
        return 384
    else:
        # Default to large
        logger.warning(f"Unknown model variant '{model_name}', defaulting to large (1024 dim)")
        return 1024


def create_sdt_head_for_da3(
    dim_in: int,
    output_dim: int = 2,
    activation: str = "exp",
    conf_activation: str = "expp1",
    fusion_channels: int = 256,
    use_sky_head: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an SDTHead adapter configured for DA3.

    Args:
        dim_in: Input feature dimension from backbone
        output_dim: Number of output channels (1 for depth, 2 for depth+conf)
        activation: Activation for depth output
        conf_activation: Activation for confidence output
        fusion_channels: SDTHead fusion channel dimension
        use_sky_head: Whether to include sky prediction
        **kwargs: Additional arguments passed to SDTHeadAdapter

    Returns:
        SDTHeadAdapter instance
    """
    from .sdt_head_adapter import SDTHeadAdapter

    return SDTHeadAdapter(
        dim_in=dim_in,
        patch_size=14,  # DinoV2 patch size
        output_dim=output_dim,
        activation=activation,
        conf_activation=conf_activation,
        fusion_channels=fusion_channels,
        use_sky_head=use_sky_head,
        **kwargs
    )


def swap_head_to_sdt(
    model: nn.Module,
    dim_in: Optional[int] = None,
    output_dim: int = 2,
    activation: str = "exp",
    conf_activation: str = "expp1",
    fusion_channels: int = 256,
    use_sky_head: bool = True,
    **kwargs
) -> nn.Module:
    """
    Swap the head of a DA3 model to SDTHead.

    This function replaces the DPT/DualDPT head with SDTHeadAdapter
    while preserving the backbone and other components.

    Args:
        model: DA3 model (DepthAnything3 or DepthAnything3Net)
        dim_in: Input dimension (auto-detected if None)
        output_dim: Number of output channels
        activation: Activation for depth output
        conf_activation: Activation for confidence output
        fusion_channels: SDTHead fusion channel dimension
        use_sky_head: Whether to include sky prediction
        **kwargs: Additional arguments passed to SDTHeadAdapter

    Returns:
        Model with SDTHead (same object, modified in-place)
    """
    # Get the actual network (DepthAnything3 wraps DepthAnything3Net)
    if hasattr(model, 'model'):
        net = model.model
        model_name = getattr(model, 'model_name', 'da3-large')
    else:
        net = model
        model_name = 'da3-large'

    # Auto-detect dimension from backbone if not provided
    if dim_in is None:
        dim_in = get_backbone_dim(model_name)
        logger.info(f"Auto-detected backbone dimension: {dim_in}")

    # Create SDTHead adapter
    sdt_head = create_sdt_head_for_da3(
        dim_in=dim_in,
        output_dim=output_dim,
        activation=activation,
        conf_activation=conf_activation,
        fusion_channels=fusion_channels,
        use_sky_head=use_sky_head,
        **kwargs
    )

    # Move to same device as original head
    device = next(net.head.parameters()).device
    sdt_head = sdt_head.to(device)

    # Swap the head
    old_head_type = type(net.head).__name__
    net.head = sdt_head

    logger.info(f"Swapped head from {old_head_type} to SDTHeadAdapter")

    return model


def create_da3_with_sdt(
    model_name: str = "da3nested-giant-large",
    device: Optional[torch.device] = None,
    output_dim: int = 2,
    fusion_channels: int = 256,
    use_sky_head: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a DA3 model with SDTHead from scratch.

    This function:
    1. Loads the specified DA3 model
    2. Swaps its head to SDTHead
    3. Returns the modified model

    Args:
        model_name: DA3 model preset name
        device: Device to load model on (default: auto-detect)
        output_dim: Number of output channels
        fusion_channels: SDTHead fusion channel dimension
        use_sky_head: Whether to include sky prediction
        **kwargs: Additional arguments for SDTHead

    Returns:
        DA3 model with SDTHead
    """
    # Import here to avoid circular dependency
    from depth_anything_3.api import DepthAnything3

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    logger.info(f"Loading DA3 model: {model_name}")
    model = DepthAnything3(model_name=model_name)
    model = model.to(device)

    # Swap head to SDT
    model = swap_head_to_sdt(
        model,
        output_dim=output_dim,
        fusion_channels=fusion_channels,
        use_sky_head=use_sky_head,
        **kwargs
    )

    model.eval()
    return model


def load_da3_with_sdt_from_pretrained(
    repo_id: str,
    device: Optional[torch.device] = None,
    output_dim: int = 2,
    fusion_channels: int = 256,
    use_sky_head: bool = True,
    **kwargs
) -> nn.Module:
    """
    Load a pre-trained DA3 model from HuggingFace Hub and swap to SDTHead.

    Args:
        repo_id: HuggingFace Hub repository ID
        device: Device to load model on
        output_dim: Number of output channels
        fusion_channels: SDTHead fusion channel dimension
        use_sky_head: Whether to include sky prediction
        **kwargs: Additional arguments for SDTHead

    Returns:
        DA3 model with SDTHead
    """
    from depth_anything_3.api import DepthAnything3

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load from HuggingFace Hub
    logger.info(f"Loading DA3 model from HuggingFace: {repo_id}")
    model = DepthAnything3.from_pretrained(repo_id)
    model = model.to(device)

    # Infer model name from repo_id for dimension detection
    model_name = repo_id.split('/')[-1].lower() if '/' in repo_id else repo_id.lower()
    model.model_name = model_name

    # Swap head to SDT
    model = swap_head_to_sdt(
        model,
        output_dim=output_dim,
        fusion_channels=fusion_channels,
        use_sky_head=use_sky_head,
        **kwargs
    )

    model.eval()
    return model
