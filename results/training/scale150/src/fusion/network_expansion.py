"""Expand a fitted residual branch while preserving its prediction function."""
import math

import torch
from torch import nn

from .advanced_models import PODDeepONet


@torch.no_grad()
def expand_linear(source, target, generator, noise_fraction=0., output_scale=1.):
    """Replicate outputs and split repeated inputs; balanced noise breaks symmetry."""
    if (target.in_features % source.in_features or target.out_features % source.out_features
            or (source.bias is None) != (target.bias is None)):
        raise ValueError("Linear dimensions must expand by integer factors")
    inputs = target.in_features // source.in_features
    outputs = target.out_features // source.out_features
    weight = source.weight.repeat(outputs, inputs) / inputs
    if inputs > 1 and noise_fraction:
        # Repeated inputs are equal at initialization. Exactly zero-sum paired
        # columns preserve that function while allowing independent gradients.
        shape = (target.out_features, inputs, source.in_features)
        noise = torch.randn(shape, generator=generator, dtype=torch.float64)
        noise -= noise.mean(dim=1, keepdim=True)
        noise *= noise_fraction * float(source.weight.std()) / inputs
        weight = (weight.double().reshape(shape) + noise).reshape_as(weight).to(weight)
    target.weight.copy_(weight * output_scale)
    if source.bias is not None:
        target.bias.copy_(source.bias.repeat(outputs) * output_scale)


@torch.no_grad()
def expand_model(source, width, depth, seed=2091, noise_fraction=.01):
    if (type(source) is not PODDeepONet or source.trunk_rank or width % source.width
            or width < source.width or depth < len(source.blocks) or not 0 <= noise_fraction <= .1):
        raise ValueError("Only integer widening and added identity blocks are supported")
    if any(not isinstance(block.norm, nn.LayerNorm) for block in source.blocks):
        raise ValueError("The fitted branch must use the declared layer normalization")
    dropout = source.blocks[0].layers[2].p
    if any(block.layers[2].p != dropout or block.layers[4].p != dropout for block in source.blocks):
        raise ValueError("Inconsistent residual dropout settings")
    generator = torch.Generator().manual_seed(seed)
    target = PODDeepONet(source.input_dim, source.n_coefficients, width=width, depth=depth,
                        dropout=dropout, normalization="layer", trunk_rank=0,
                        linear_skip=source.linear_coefficients is not None)
    factor = width // source.width
    expand_linear(source.stem[0], target.stem[0], generator)
    for before, after in zip(source.blocks, target.blocks):
        after.norm.weight.copy_(before.norm.weight.repeat(factor))
        after.norm.bias.copy_(before.norm.bias.repeat(factor))
        after.norm.eps = before.norm.eps
        expand_linear(before.layers[0], after.layers[0], generator, noise_fraction)
        expand_linear(before.layers[3], after.layers[3], generator, noise_fraction,
                      output_scale=before.scale / after.scale)
    for block in target.blocks[len(source.blocks):]:
        block.layers[3].weight.zero_()
        block.layers[3].bias.zero_()
    expand_linear(source.coefficient_head, target.coefficient_head, generator, noise_fraction)
    expand_linear(source.scalar_head[0], target.scalar_head[0], generator, noise_fraction)
    expand_linear(source.scalar_head[2], target.scalar_head[2], generator, noise_fraction)
    if source.linear_coefficients is not None:
        target.linear_coefficients.load_state_dict(source.linear_coefficients.state_dict())
        target.linear_scalars.load_state_dict(source.linear_scalars.state_dict())
    return target
