"""
DA3-Serverless Model Module

This module provides SDTHead integration for DA3 models.

Components:
- SDTHead: Stable Depth Transformer Head with DySample upsampling
- SDTHeadAdapter: Adapter matching DA3's head interface
- da3_sdt utilities: Functions to create/modify DA3 models with SDTHead
"""

from .sdt_head import (
    SDTHead,
    DySample,
    WeightedFusion,
    SpatialDetailEnhancer,
    DySampleUpsamplerWrapper,
)

from .sdt_head_adapter import SDTHeadAdapter

from .da3_sdt import (
    create_da3_with_sdt,
    swap_head_to_sdt,
    load_da3_with_sdt_from_pretrained,
    create_sdt_head_for_da3,
    get_backbone_dim,
)

__all__ = [
    # SDTHead components
    "SDTHead",
    "DySample",
    "WeightedFusion",
    "SpatialDetailEnhancer",
    "DySampleUpsamplerWrapper",
    # Adapter
    "SDTHeadAdapter",
    # DA3 integration utilities
    "create_da3_with_sdt",
    "swap_head_to_sdt",
    "load_da3_with_sdt_from_pretrained",
    "create_sdt_head_for_da3",
    "get_backbone_dim",
]
