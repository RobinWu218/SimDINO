# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


XFORMERS_AVAILABLE = False
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat
        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
except ImportError:
    warnings.warn("xFormers is not available (Block)")
if not XFORMERS_AVAILABLE:
        def scaled_index_add(input, index, source, scaling, alpha):
            return torch.index_add(input, dim=0, source=scaling * source, index=index, alpha=alpha)
        def index_select_cat(sources, indices):
            return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)

def setup_layer_scales(dim, init_value):
    if init_value:
        init_value = (init_value, init_value)
        ls1 = LayerScale(dim, init_values=init_value[0])
        ls2 = LayerScale(dim, init_values=init_value[1])
    else:
        ls1 = nn.Identity()
        ls2 = nn.Identity()
    return ls1, ls2

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        layerscale=None,
        drop_path: float = 0.0,
        qknorm=False,
        inline_linear=False,
        max_resolution=14,
        ext_token_num=5,
        base=20,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        **kwargs,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=ffn_drop,
            qknorm=qknorm,
            inline_linear=inline_linear,
            max_resolution=max_resolution,
            ext_token_num=ext_token_num,
            base=base
        )
        self.ls1, self.ls2 = setup_layer_scales(dim, layerscale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_drop,
            bias=ffn_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.ext_token_num = ext_token_num

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path(attn_residual_func(x))
            x = x + self.drop_path(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    keep_index = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[keep_index]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, keep_index, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


def get_indexs_scales(x, sample_drop_ratio=0.0):
    """
    Computes a subset of batch indices and a residual scale factor based on the given sample drop ratio.

    Args:
        x (torch.Tensor): Input tensor of shape (b, n, d), where b is the batch size, n is the sequence length, and d is the feature dimension.
        sample_drop_ratio (float, optional): The ratio of samples to drop. Default is 0.0.

    Returns:
        tuple: A tuple containing:
            - keep_index (torch.Tensor): A tensor of selected batch indices.
            - residual_scale_factor (float): The scale factor to adjust for the reduced batch size.
    """
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    keep_index = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return keep_index, residual_scale_factor


def add_residual(x, keep_index, residual, residual_scale_factor, scaling_vector=None):
    """
    Adds a residual to the input tensor `x` with optional scaling.

    Parameters:
    x (torch.Tensor): The input tensor to which the residual will be added.
    keep_index (torch.Tensor): The indices at which to add the residual.
    residual (torch.Tensor): The residual tensor to be added to `x`.
    residual_scale_factor (float): A scaling factor applied to the residual before addition.
    scaling_vector (torch.Tensor, optional): An optional scaling vector applied to the residual. Defaults to None.

    Returns:
    torch.Tensor: The tensor resulting from adding the scaled residual to `x`.
    """
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat,
            0,
            keep_index,
            residual.to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            keep_index,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, keep_indexs=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = (
        [b.shape[0] for b in keep_indexs]
        if keep_indexs is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if keep_indexs is not None:
        cat_tensors = index_select_cat(
            [x.flatten(1) for x in x_list], keep_indexs
        ).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    keep_indexs_scales = [
        get_indexs_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
    ]
    keep_indexs = [s[0] for s in keep_indexs_scales]
    residual_scale_factors = [s[1] for s in keep_indexs_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, keep_indexs)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, keep_index, residual, residual_scale_factor in zip(
        x_list, keep_indexs, residual_list, residual_scale_factors
    ):
        outputs.append(
            add_residual(
                x, keep_index, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        # assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            ls1 = ls2 = None
            if isinstance(self.ls1, LayerScale):
                ls1 = self.ls1.gamma
            elif isinstance(self.ls1, OffsetLayerScale):
                ls1 = self.ls1.lam + self.ls1.offset
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=ls1,
            )
            if isinstance(self.ls2, LayerScale):
                ls2 = self.ls2.gamma
            elif isinstance(self.ls2, OffsetLayerScale):
                ls2 = self.ls2.lam + self.ls2.offset
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=ls2,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
