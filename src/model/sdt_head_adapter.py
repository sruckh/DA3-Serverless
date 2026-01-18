"""
SDTHead Adapter for DA3 Integration

This module provides an adapter that wraps SDTHead to match the DA3 head interface.
It handles the feature format transformation between DA3's backbone output and
SDTHead's expected input format.

DA3 Feature Format:
    - List of 4 tuples: [(tensor[B,S,N,C], cls_token[B,S,C]), ...]
    - N = number of patches (H_patch * W_patch)
    - S = sequence dimension (number of views)

SDTHead Expected Format:
    - List of 4 tuples: [(tensor[B,C,H,W], cls_token[B,C]), ...]
    - Spatial 2D format

Output Format (DA3 compatible):
    - Dict with 'depth' key: tensor[B, S, H, W]
    - Optional 'depth_conf' and 'sky' keys for compatibility
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from addict import Dict as AdDict

from .sdt_head import SDTHead


class SDTHeadAdapter(nn.Module):
    """
    Adapter that wraps SDTHead to match DA3's head interface.

    This adapter:
    1. Transforms DA3 features [B,S,N,C] -> [B*S,C,H,W]
    2. Calls SDTHead for depth prediction
    3. Applies activation and formats output as DA3-compatible Dict

    Args:
        dim_in: Input feature dimension from backbone
        patch_size: Patch size used by backbone (default: 14 for DinoV2)
        output_dim: Number of output channels (1 for depth, 2 for depth+conf)
        activation: Activation function for depth output ('exp', 'relu', 'sigmoid', 'linear')
        conf_activation: Activation for confidence output ('expp1', 'sigmoid', 'relu')
        fusion_channels: SDTHead fusion channel dimension (default: 256)
        head_name: Name for the depth output key (default: 'depth')
        use_sky_head: Whether to include a sky prediction head (default: False)
        sky_name: Name for sky output key (default: 'sky')
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        fusion_channels: int = 256,
        head_name: str = "depth",
        use_sky_head: bool = False,
        sky_name: str = "sky",
        sky_activation: str = "relu",
        **kwargs
    ):
        super().__init__()

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.head_name = head_name
        self.sky_name = sky_name
        self.sky_activation = sky_activation
        self.use_sky_head = use_sky_head

        # Output dimension handling
        self.out_dim = output_dim
        self.has_conf = output_dim > 1

        # Total output channels: depth (+ optional conf) + optional sky
        n_output_channels = output_dim
        if use_sky_head:
            n_output_channels += 1

        # SDTHead expects 4 layers with same input dimension (from DinoV2)
        in_channels = [dim_in] * 4

        self.sdt_head = SDTHead(
            in_channels=in_channels,
            fusion_channels=fusion_channels,
            n_output_channels=n_output_channels,
            use_cls_token=True,
        )

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int = 0,
        chunk_size: int = 8,
        **kwargs,
    ) -> AdDict:
        """
        Forward pass matching DA3's DPT head interface.

        Args:
            feats: List of 4 feature entries from backbone.
                   Each entry is a tuple (tensor[B,S,N,C], cls_token[B,S,C])
                   or just tensor[B,S,N,C]
            H: Original image height
            W: Original image width
            patch_start_idx: Starting index of patch tokens (for cropping non-patch tokens)
            chunk_size: Chunk size for processing along sequence dimension

        Returns:
            AdDict with keys:
                - {head_name}: Depth prediction [B, S, H_out, W_out]
                - {head_name}_conf: Confidence (if output_dim > 1) [B, S, H_out, W_out]
                - {sky_name}: Sky prediction (if use_sky_head) [B, S, H_out, W_out]
        """
        # Extract batch and sequence dimensions
        B, S, N, C = feats[0][0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        # Transform features to SDTHead format
        # From: [B, S, N, C] with cls_token [B, S, C]
        # To: [B*S, C, H_patch, W_patch] with cls_token [B*S, C]
        transformed_feats = []
        for feat in feats:
            spatial = feat[0]  # [B, S, N, C]
            cls_token = feat[1] if len(feat) > 1 else None  # [B, S, C]

            # Crop patch tokens and reshape
            spatial = spatial[:, :, patch_start_idx:]  # [B, S, N_patch, C]
            spatial = spatial.reshape(B * S, -1, C)  # [B*S, N_patch, C]
            spatial = spatial.permute(0, 2, 1).contiguous()  # [B*S, C, N_patch]
            spatial = spatial.reshape(B * S, C, ph, pw)  # [B*S, C, H_patch, W_patch]

            if cls_token is not None:
                cls_token = cls_token.reshape(B * S, C)  # [B*S, C]
                transformed_feats.append((spatial, cls_token))
            else:
                transformed_feats.append(spatial)

        # Process through SDTHead (with optional chunking for memory efficiency)
        if chunk_size is None or chunk_size >= S:
            output = self._forward_impl(transformed_feats, B, S, H, W)
        else:
            # Chunk processing along sequence dimension
            outputs = []
            for s0 in range(0, S, chunk_size):
                s1 = min(s0 + chunk_size, S)
                chunk_feats = self._extract_chunk(transformed_feats, s0, s1, B)
                chunk_out = self._forward_impl(chunk_feats, B, s1 - s0, H, W)
                outputs.append(chunk_out)
            output = self._merge_chunks(outputs)

        return output

    def _extract_chunk(
        self,
        feats: List,
        s0: int,
        s1: int,
        B: int
    ) -> List:
        """Extract a chunk of features along sequence dimension."""
        chunk_feats = []
        for feat in feats:
            if isinstance(feat, tuple):
                spatial, cls = feat
                # spatial is [B*S, C, H, W], need to extract [B*(s1-s0), C, H, W]
                # Reshape to [B, S, C, H, W], slice, reshape back
                C, H_p, W_p = spatial.shape[1:]
                spatial = spatial.reshape(B, -1, C, H_p, W_p)[:, s0:s1].reshape(-1, C, H_p, W_p)
                cls = cls.reshape(B, -1, cls.shape[-1])[:, s0:s1].reshape(-1, cls.shape[-1])
                chunk_feats.append((spatial, cls))
            else:
                C, H_p, W_p = feat.shape[1:]
                feat = feat.reshape(B, -1, C, H_p, W_p)[:, s0:s1].reshape(-1, C, H_p, W_p)
                chunk_feats.append(feat)
        return chunk_feats

    def _merge_chunks(self, outputs: List[AdDict]) -> AdDict:
        """Merge chunked outputs along sequence dimension."""
        merged = AdDict()
        for key in outputs[0].keys():
            merged[key] = torch.cat([o[key] for o in outputs], dim=1)
        return merged

    def _forward_impl(
        self,
        feats: List,
        B: int,
        S: int,
        H: int,
        W: int
    ) -> AdDict:
        """
        Internal forward implementation.

        Args:
            feats: Transformed features in SDTHead format
            B: Batch size
            S: Sequence length
            H, W: Original image dimensions

        Returns:
            AdDict with depth (and optional conf/sky) predictions
        """
        # Run SDTHead
        raw_output = self.sdt_head(feats)  # [B*S, n_channels, H_out, W_out]

        # Split output channels
        outs = AdDict()
        channel_idx = 0

        if self.has_conf:
            # Depth + confidence
            depth_logits = raw_output[:, channel_idx:channel_idx + self.out_dim - 1]
            channel_idx += self.out_dim - 1
            conf_logits = raw_output[:, channel_idx:channel_idx + 1]
            channel_idx += 1

            depth = self._apply_activation(depth_logits, self.activation)
            conf = self._apply_activation(conf_logits, self.conf_activation)

            # Reshape to [B, S, H, W]
            depth = depth.reshape(B, S, *depth.shape[1:]).squeeze(2)
            conf = conf.reshape(B, S, *conf.shape[1:]).squeeze(2)

            outs[self.head_name] = depth
            outs[f"{self.head_name}_conf"] = conf
        else:
            # Depth only
            depth_logits = raw_output[:, channel_idx:channel_idx + 1]
            channel_idx += 1

            depth = self._apply_activation(depth_logits, self.activation)
            depth = depth.reshape(B, S, *depth.shape[1:]).squeeze(2)

            outs[self.head_name] = depth

        # Sky head (if enabled)
        if self.use_sky_head:
            sky_logits = raw_output[:, channel_idx:channel_idx + 1]
            sky = self._apply_sky_activation(sky_logits)
            sky = sky.reshape(B, S, *sky.shape[1:]).squeeze(2)
            outs[self.sky_name] = sky

        return outs

    def _apply_activation(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """Apply activation function to output."""
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x

    def _apply_sky_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to sky output."""
        act = self.sky_activation.lower() if isinstance(self.sky_activation, str) else self.sky_activation
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "relu":
            return torch.relu(x)
        return x
