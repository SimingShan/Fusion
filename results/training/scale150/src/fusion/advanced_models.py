"""Efficient POD DeepONet variants for the fixed equilibrium grid.

PODDeepONet learns coefficients of a *training-only* POD coordinate trunk.  The
POD basis, field mean, and coefficient normalization remain outside this module:
the caller reconstructs ``coefficients @ basis + mean`` in the desired units.
An optional learned coordinate trunk can add a residual to that reconstruction.

PODFourierDeepONet adds a conditioned Fourier decoder to the reconstructed field.
This is a POD/Fourier operator variant, rather than the original paper's exact
architecture.  The decoder is initialized to produce zero residual, so a fitted
POD predictor can be transferred without changing its initial predictions.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import SpectralConv2d


def _grid_coordinates(grid_shape: Sequence[int]) -> Tensor:
    if len(grid_shape) != 2 or any(int(n) != n or n < 2 for n in grid_shape):
        raise ValueError("grid_shape must contain two integer dimensions >= 2")
    zz, rr = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(grid_shape[0])),
        torch.linspace(-1.0, 1.0, int(grid_shape[1])), indexing="ij",
    )
    return torch.stack((rr, zz), dim=0)


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual block that preserves the main feature scale."""

    def __init__(self, width: int, dropout: float, scale: float,
                 normalization: str = "layer", expansion: int = 2) -> None:
        super().__init__()
        if normalization == "layer":
            self.norm = nn.LayerNorm(width)
        elif normalization == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError("normalization must be 'layer' or 'none'")
        self.layers = nn.Sequential(
            nn.Linear(width, expansion * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(expansion * width, width), nn.Dropout(dropout),
        )
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.layers(self.norm(x))


class PODDeepONet(nn.Module):
    """Residual branch network with a fixed external POD coordinate trunk.

    Args:
        input_dim: Number of externally normalized input features.
        n_coefficients: Number of retained POD components.
        width: Hidden branch width.
        depth: Number of residual branch blocks.
        dropout: Dropout probability inside the branch blocks.
        normalization: ``layer`` (inside residual blocks) or ``none``.
        trunk_rank: Optional additional learned coordinate-trunk rank; zero
            disables this path. Its output uses the caller's flux loss units.
        grid_shape: Fixed output grid in (Z, R) order.
        trunk_width: Hidden width of the optional coordinate trunk.
        linear_skip: Learn a direct input-to-output linear map in parallel.

    ``forward(x)`` returns ``coefficients`` with shape (B, K), ``scalars`` with
    shape (B, 2), and optionally ``flux_residual`` with shape (B, Z, R). The
    scalars are ordered q95, betaN. Normalization and denormalization of all
    outputs are the caller's responsibility.
    """

    def __init__(self, input_dim: int, n_coefficients: int, width: int = 512,
                 depth: int = 4, dropout: float = 0.0,
                 normalization: str = "layer", trunk_rank: int = 0,
                 grid_shape: Sequence[int] = (65, 65), trunk_width: int = 128,
                 linear_skip: bool = True) -> None:
        super().__init__()
        if min(input_dim, n_coefficients, width, depth, trunk_width) < 1:
            raise ValueError("Model dimensions and depth must be positive")
        if not 0.0 <= dropout < 1.0 or trunk_rank < 0:
            raise ValueError("dropout must lie in [0, 1) and trunk_rank must be >= 0")
        coords = _grid_coordinates(grid_shape)
        self.input_dim = int(input_dim)
        self.n_coefficients = int(n_coefficients)
        self.width = int(width)
        self.trunk_rank = int(trunk_rank)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(width, dropout, 1.0 / math.sqrt(depth), normalization)
            for _ in range(depth)
        ])
        self.coefficient_head = nn.Linear(width, n_coefficients)
        self.scalar_head = nn.Sequential(
            nn.Linear(width, width // 2 or 1), nn.GELU(),
            nn.Linear(width // 2 or 1, 2),
        )
        self.linear_coefficients = nn.Linear(input_dim, n_coefficients) if linear_skip else None
        self.linear_scalars = nn.Linear(input_dim, 2) if linear_skip else None
        for head in (self.linear_coefficients, self.linear_scalars):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        # Externally fitted ridge coefficients can initialize the linear heads;
        # zero nonlinear heads preserve those predictions exactly at transfer.
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)
        if trunk_rank:
            self.register_buffer("coordinates", coords.permute(1, 2, 0).reshape(-1, 2))
            self.trunk = nn.Sequential(
                nn.Linear(2, trunk_width), nn.Tanh(),
                nn.Linear(trunk_width, trunk_width), nn.Tanh(),
                nn.Linear(trunk_width, trunk_rank),
            )
            self.trunk_branch = nn.Linear(width, trunk_rank)
            nn.init.zeros_(self.trunk_branch.weight)
            nn.init.zeros_(self.trunk_branch.bias)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x of shape (B, {self.input_dim}), got {tuple(x.shape)}")
        return self.blocks(self.stem(x))

    def _outputs(self, x: Tensor, hidden: Tensor) -> dict[str, Tensor]:
        coefficients = self.coefficient_head(hidden)
        scalars = self.scalar_head(hidden)
        if self.linear_coefficients is not None:
            coefficients = coefficients + self.linear_coefficients(x)
        if self.linear_scalars is not None:
            scalars = scalars + self.linear_scalars(x)
        outputs = {"coefficients": coefficients, "scalars": scalars}
        if self.trunk_rank:
            trunk = self.trunk(self.coordinates.to(dtype=x.dtype))
            residual = self.trunk_branch(hidden) @ trunk.T / math.sqrt(self.trunk_rank)
            outputs["flux_residual"] = residual.reshape(x.shape[0], *self.grid_shape)
        return outputs

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self._outputs(x, self.encode(x))


class DerivedInputPOD(PODDeepONet):
    """Preserve a fitted branch and add zero-initialized physical input paths."""

    def __init__(self, input_dim, n_coefficients, physical_schema, **kwargs):
        super().__init__(input_dim, n_coefficients, **kwargs)
        if self.trunk_rank:
            raise ValueError("This branch extension requires fixed output coordinates")
        from .input_derived import DerivedInputs
        self.derived_inputs = DerivedInputs(physical_schema)
        self.derived_stem = nn.Linear(self.derived_inputs.width, self.width, bias=False)
        self.derived_coefficients = nn.Linear(self.derived_inputs.width, n_coefficients, bias=False)
        self.derived_scalars = nn.Linear(self.derived_inputs.width, 2, bias=False)
        for layer in (self.derived_stem, self.derived_coefficients, self.derived_scalars):
            nn.init.zeros_(layer.weight)

    def encode(self, x):
        physical = self.derived_inputs(x)
        return self.blocks(self.stem[1](self.stem[0](x) + self.derived_stem(physical)))

    def _outputs(self, x, hidden):
        outputs = super()._outputs(x, hidden)
        physical = self.derived_inputs(x)
        outputs["coefficients"] = outputs["coefficients"] + self.derived_coefficients(physical)
        outputs["scalars"] = outputs["scalars"] + self.derived_scalars(physical)
        return outputs


class _ResidualFourierBlock(nn.Module):
    def __init__(self, channels: int, modes: int, scale: float) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes)
        self.local = nn.Conv2d(channels, channels, 1)
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * F.gelu(self.spectral(x) + self.local(x))


class FourierFluxRefiner(nn.Module):
    """Conditioned spectral residual on a supplied normalized coarse flux map.

    Coordinates are normalized R, Z and follow the grid's rows-Z/columns-R order.
    The default seven-cell right/bottom padding maps a 65x65 grid to 72x72 for
    efficient FFTs. The final projection starts at zero; freeze a fitted branch
    if initially training the decoder alone.
    """

    def __init__(self, conditioning_dim: int = 8, spectral_width: int = 16,
                 spectral_layers: int = 2, modes: int = 10, padding: int = 7,
                 grid_shape: Sequence[int] = (65, 65)) -> None:
        super().__init__()
        if min(conditioning_dim, spectral_width, spectral_layers, modes) < 1 or padding < 0:
            raise ValueError("Spectral dimensions must be positive and padding nonnegative")
        self.register_buffer("coordinates", _grid_coordinates(grid_shape).unsqueeze(0))
        self.conditioning_dim = int(conditioning_dim)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.padding = int(padding)
        self.lift = nn.Conv2d(3 + conditioning_dim, spectral_width, 1)
        self.blocks = nn.Sequential(*[
            _ResidualFourierBlock(spectral_width, modes, 1.0 / math.sqrt(spectral_layers))
            for _ in range(spectral_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(spectral_width, 2 * spectral_width, 1), nn.GELU(),
            nn.Conv2d(2 * spectral_width, 1, 1),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, coarse_field: Tensor, conditioning: Tensor) -> Tensor:
        if coarse_field.ndim != 3 or tuple(coarse_field.shape[1:]) != self.grid_shape:
            raise ValueError(f"Expected coarse_field of shape (B, {self.grid_shape[0]}, "
                             f"{self.grid_shape[1]}), got {tuple(coarse_field.shape)}")
        batch = coarse_field.shape[0]
        if tuple(conditioning.shape) != (batch, self.conditioning_dim):
            raise ValueError(f"Expected conditioning of shape (B, {self.conditioning_dim})")
        coordinates = self.coordinates.to(dtype=coarse_field.dtype).expand(batch, -1, -1, -1)
        condition_grid = conditioning.to(dtype=coarse_field.dtype)[:, :, None, None].expand(
            -1, -1, *self.grid_shape,
        )
        field = self.lift(torch.cat((coarse_field[:, None], coordinates, condition_grid), dim=1))
        if self.padding:
            field = F.pad(field, (0, self.padding, 0, self.padding))
        field = self.blocks(field)[..., :self.grid_shape[0], :self.grid_shape[1]]
        return self.project(field).squeeze(1)


class PODFourierDeepONet(PODDeepONet):
    """POD branch/trunk reconstruction followed by spectral residual refinement.

    Recommended single-encoding use::

        outputs = model(normalized_inputs)
        coarse = ((outputs['coefficients'] * coefficient_scale) @ basis + mean)
        coarse = coarse.reshape(-1, *model.grid_shape)
        if 'flux_residual' in outputs:
            coarse = coarse + outputs['flux_residual']
        flux = coarse + model.refine(coarse, outputs['conditioning'])

    The basis and mean must be in compatible *normalized* flux units when using
    the spectral decoder. The reconstructed flux is fully differentiable back
    to the coefficient branch. ``forward(x, coarse_field)`` instead accepts a
    caller-supplied coarse field and includes ``fourier_residual`` in its output;
    the caller must ensure that field corresponds to the same input batch.
    """

    def __init__(self, input_dim: int, n_coefficients: int, width: int = 512,
                 depth: int = 4, dropout: float = 0.0,
                 normalization: str = "layer", trunk_rank: int = 0,
                 grid_shape: Sequence[int] = (65, 65), trunk_width: int = 128,
                 linear_skip: bool = True, conditioning_dim: int = 8,
                 spectral_width: int = 16, spectral_layers: int = 2,
                 modes: int = 10, padding: int = 7) -> None:
        super().__init__(input_dim, n_coefficients, width, depth, dropout,
                         normalization, trunk_rank, grid_shape, trunk_width, linear_skip)
        self.conditioning_head = nn.Linear(width, conditioning_dim)
        self.refiner = FourierFluxRefiner(conditioning_dim, spectral_width,
                                         spectral_layers, modes, padding, grid_shape)

    def forward(self, x: Tensor, coarse_field: Tensor | None = None) -> dict[str, Tensor]:
        hidden = self.encode(x)
        outputs = self._outputs(x, hidden)
        outputs["conditioning"] = self.conditioning_head(hidden)
        if coarse_field is not None:
            outputs["fourier_residual"] = self.refine(coarse_field, outputs["conditioning"])
        return outputs

    def refine(self, coarse_field: Tensor, conditioning: Tensor) -> Tensor:
        return self.refiner(coarse_field, conditioning)


class ChordTokenPOD(PODDeepONet):
    """Chord-token TRANSFORMER encoder with the shared POD coefficient/scalar heads (same outputs as PODDeepONet).
    Tokens: one per Thomson core chord (Te, ne, observed flags, Z, geometry flag) and per edge chord (Te, ne, flags, R, geometry
    flag) taken from the standardised feature vector by NAME, plus one global token (every non-chord feature through an MLP).
    ``depth`` = number of transformer layers; ``width`` = hidden width fed to the heads (pooled global token + mean chord token).
    Parameter names put the transformer under ``blocks.layers.*`` so the Muon partition can be addressed by MUON_PATTERN."""

    def __init__(self, input_dim: int, n_coefficients: int, feature_names: Sequence[str], width: int = 256, depth: int = 4,
                 dropout: float = 0.0, token_dim: int = 128, heads: int = 4, **kwargs) -> None:
        import re
        super().__init__(input_dim, n_coefficients, width=width, depth=1, dropout=dropout)
        names = list(feature_names)
        if len(names) != int(input_dim):
            raise ValueError(f"feature_names ({len(names)}) must match input_dim ({input_dim})")
        pos = {n: j for j, n in enumerate(names)}

        def chord(system):
            feats = ["Te_log1p_channel", "ne_log1p_channel", "Te_observed_channel", "ne_observed_channel",
                     ("Z_channel" if system == "core" else "R_channel"), "geometry_observed_channel"]
            chans = sorted({m.group(1) for n in names for m in [re.fullmatch(rf"thomson_{system}_Te_log1p_channel_(\d+)", n)] if m})
            cols = [[pos.get(f"thomson_{system}_{f}_{c}", -1) for f in feats] for c in chans]
            return cols

        core, edge = chord("core"), chord("edge")
        if not core or not edge:
            raise ValueError("chord tokens need thomson_core/edge *_channel_* features")
        used = {c for cc in core + edge for c in cc if c >= 0}
        glob = [j for j in range(int(input_dim)) if j not in used]
        self.register_buffer("core_idx", torch.tensor([[max(c, 0) for c in cc] for cc in core]))
        self.register_buffer("core_msk", torch.tensor([[float(c >= 0) for c in cc] for cc in core]))
        self.register_buffer("edge_idx", torch.tensor([[max(c, 0) for c in cc] for cc in edge]))
        self.register_buffer("edge_msk", torch.tensor([[float(c >= 0) for c in cc] for cc in edge]))
        self.register_buffer("glob_idx", torch.tensor(glob))
        self.n_core, self.n_edge, self.token_dim = len(core), len(edge), int(token_dim)
        F = 6
        self.core_emb = nn.Linear(F + 1, token_dim); self.edge_emb = nn.Linear(F + 1, token_dim)
        self.glob_emb = nn.Sequential(nn.Linear(len(glob), 2 * token_dim), nn.GELU(), nn.Linear(2 * token_dim, token_dim))
        self.sys_tok = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        self.chan_pos = nn.Parameter(torch.randn(self.n_core + self.n_edge, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(token_dim, heads, 4 * token_dim, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.stem = nn.Identity()                                  # replaces the MLP stem
        self.blocks = nn.TransformerEncoder(layer, int(depth))     # replaces the residual MLP blocks (Muon target: blocks.layers.*)
        self.token_norm = nn.LayerNorm(token_dim)
        self.pool = nn.Sequential(nn.Linear(2 * token_dim, width), nn.GELU())

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x of shape (B, {self.input_dim}), got {tuple(x.shape)}")
        B = x.shape[0]; ones = x.new_ones(B, 1, 1)
        core = x[:, self.core_idx] * self.core_msk; edge = x[:, self.edge_idx] * self.edge_msk
        core = torch.cat([core, ones.expand(B, core.shape[1], 1)], 2); edge = torch.cat([edge, ones.expand(B, edge.shape[1], 1)], 2)
        tok = torch.cat([self.core_emb(core) + self.sys_tok[0] + self.chan_pos[: self.n_core],
                         self.edge_emb(edge) + self.sys_tok[1] + self.chan_pos[self.n_core:],
                         (self.glob_emb(x[:, self.glob_idx]) + self.sys_tok[2])[:, None, :]], 1)
        h = self.token_norm(self.blocks(tok))
        return self.pool(torch.cat([h[:, -1], h[:, :-1].mean(1)], 1))


def build_advanced_model(name: str, input_dim: int, n_coefficients: int,
                         **kwargs) -> PODDeepONet:
    """Build a POD model; shared configs may contain Fourier-only options."""
    normalized = name.lower().replace("-", "_")
    if normalized == "pod_derived_inputs":
        return DerivedInputPOD(input_dim, n_coefficients, **kwargs)
    if normalized in ("pod", "pod_deeponet", "poddeeponet"):
        for key in ("conditioning_dim", "spectral_width", "spectral_layers", "modes", "padding"):
            kwargs.pop(key, None)
        return PODDeepONet(input_dim, n_coefficients, **kwargs)
    if normalized in ("pod_fourier", "pod_fourier_deeponet", "podfourierdeeponet"):
        return PODFourierDeepONet(input_dim, n_coefficients, **kwargs)
    if normalized in ("pod_token", "chord_token", "pod_chord_token"):
        for key in ("conditioning_dim", "spectral_width", "spectral_layers", "modes", "padding"):
            kwargs.pop(key, None)
        return ChordTokenPOD(input_dim, n_coefficients, **kwargs)
    if normalized in ("pod_attention", "sensor_attention", "pod_sensor_attention"):
        for key in ("conditioning_dim", "spectral_width", "spectral_layers", "modes", "padding"):
            kwargs.pop(key, None)
        return SensorAttentionPOD(input_dim, n_coefficients, **kwargs)
    raise ValueError(f"Unknown advanced model {name!r}; choose pod_deeponet, "
                     f"pod_fourier_deeponet or pod_attention")


class _AttentionBlock(nn.Module):
    """Pre-normalized self-attention + feed-forward, residual-scaled like ResidualMLPBlock."""

    def __init__(self, dim: int, heads: int, dropout: float, scale: float,
                 expansion: int = 2) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention dim {dim} must divide by heads {heads}")
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_feed = nn.LayerNorm(dim)
        self.feed = nn.Sequential(
            nn.Linear(dim, expansion * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim), nn.Dropout(dropout),
        )
        self.scale = float(scale)

    def forward(self, latents: Tensor) -> Tensor:
        normalized = self.norm(latents)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        latents = latents + self.scale * attended
        return latents + self.scale * self.feed(self.norm_feed(latents))


class _SensorCrossAttention(nn.Module):
    """Latent queries attending to one token per input feature.

    Each feature's key and value are affine in that feature's value, so both are built
    straight from x and the per-feature parameters *are* the projection.  The (B, F, d)
    token tensor is never materialized apart from them, which is the whole cost of this
    layer at F=708: a first draft that built tokens and then projected them ran at 84 s
    per epoch against the control's 4 s.
    """

    def __init__(self, input_dim: int, dim: int, heads: int, dropout: float,
                 tokens: int = 128) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention dim {dim} must divide by heads {heads}")
        if tokens < 1:
            raise ValueError("tokens must be positive")
        self.heads = int(heads)
        self.dim = int(dim)
        # Feature order is generated diagnostic by diagnostic, so a contiguous run of
        # features is a run of adjacent channels on the same instrument.  Grouping them
        # into one token per run keeps that identity while cutting the (B, F, d)
        # intermediate -- the actual cost here -- by the group size.
        self.tokens = min(int(tokens), int(input_dim))
        self.group = math.ceil(int(input_dim) / self.tokens)
        self.padding = self.tokens * self.group - int(input_dim)
        scale = 1.0 / math.sqrt(self.group * dim)
        self.key_weight = nn.Parameter(torch.randn(self.tokens, self.group, dim) * scale)
        self.key_bias = nn.Parameter(torch.randn(self.tokens, dim) / math.sqrt(dim))
        self.value_weight = nn.Parameter(torch.randn(self.tokens, self.group, dim) * scale)
        self.value_bias = nn.Parameter(torch.randn(self.tokens, dim) / math.sqrt(dim))
        self.query = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def _split(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.dim // self.heads).transpose(1, 2)

    def _project(self, grouped: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        # (tokens, batch, group) @ (tokens, group, dim) -> (tokens, batch, dim)
        return torch.baddbmm(bias.unsqueeze(1), grouped.transpose(0, 1), weight).transpose(0, 1)

    def forward(self, latents: Tensor, x: Tensor) -> Tensor:
        if self.padding:
            x = F.pad(x, (0, self.padding))
        grouped = x.view(x.shape[0], self.tokens, self.group)
        keys = self._split(self._project(grouped, self.key_weight, self.key_bias))
        content = self._split(self._project(grouped, self.value_weight, self.value_bias))
        queries = self._split(self.query(latents))
        attended = F.scaled_dot_product_attention(
            queries, keys, content, dropout_p=self.dropout if self.training else 0.0)
        batch, _, length, _ = attended.shape
        return self.out(attended.transpose(1, 2).reshape(batch, length, self.dim))


class _SensorCrossBlock(nn.Module):
    def __init__(self, input_dim: int, dim: int, heads: int, dropout: float, scale: float,
                 tokens: int = 128, expansion: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = _SensorCrossAttention(input_dim, dim, heads, dropout, tokens)
        self.norm_feed = nn.LayerNorm(dim)
        self.feed = nn.Sequential(
            nn.Linear(dim, expansion * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim), nn.Dropout(dropout),
        )
        self.scale = float(scale)

    def forward(self, latents: Tensor, x: Tensor) -> Tensor:
        latents = latents + self.scale * self.attention(self.norm(latents), x)
        return latents + self.scale * self.feed(self.norm_feed(latents))


class SensorAttentionPOD(PODDeepONet):
    """Perceiver-style sensor encoder replacing the residual MLP branch.

    Every input feature is one reading from one diagnostic at a fixed place in the
    machine.  The MLP branch receives those numbers as a flat vector, so which features
    sit next to each other on the vessel is something it has to infer from data.  Here
    each feature gets its own learned key and value, a small set of latent queries
    attends over them, the latents then attend among themselves, and the pooled latents
    stand in for the MLP hidden state.

    Motivation is the measured underfit rather than capacity: kappa scores 0.980 on
    training shots as well as on held-out ones against a 0.9959 basis ceiling, which the
    branch cannot reach even with the labels in front of it.

    The coefficient head, scalar head, zero-initialized nonlinear path and linear skip
    are inherited unchanged, so an untrained model still reproduces whatever ridge
    solution the caller installs.
    """

    def __init__(self, input_dim: int, n_coefficients: int,
                 attention_dim: int = 64, attention_latents: int = 32,
                 attention_heads: int = 4, cross_layers: int = 1,
                 self_layers: int = 4, attention_tokens: int = 128, **kwargs) -> None:
        super().__init__(input_dim, n_coefficients, **kwargs)
        if self.trunk_rank:
            raise ValueError("The sensor encoder expects fixed output coordinates")
        if min(attention_dim, attention_latents, attention_heads, cross_layers) < 1:
            raise ValueError("Attention dimensions and cross_layers must be positive")
        if self_layers < 0:
            raise ValueError("self_layers must be nonnegative")
        dim = int(attention_dim)
        self.attention_dim = dim
        self.attention_latents = int(attention_latents)
        dropout = self.blocks[0].layers[2].p if len(self.blocks) else 0.0
        # The branch MLP is inherited but never called; drop it so the parameter budget
        # stays comparable with the control at the same --width.
        self.stem = nn.Identity()
        self.blocks = nn.Identity()
        self.latents = nn.Parameter(torch.randn(self.attention_latents, dim) / math.sqrt(dim))
        depth = int(cross_layers) + int(self_layers)
        scale = 1.0 / math.sqrt(depth)
        self.cross_blocks = nn.ModuleList([
            _SensorCrossBlock(input_dim, dim, attention_heads, dropout, scale, attention_tokens)
            for _ in range(int(cross_layers))
        ])
        self.self_blocks = nn.ModuleList([
            _AttentionBlock(dim, attention_heads, dropout, scale) for _ in range(int(self_layers))
        ])
        self.pool = nn.Sequential(
            nn.LayerNorm(dim), nn.Flatten(1),
            nn.Linear(self.attention_latents * dim, self.width), nn.GELU(),
        )

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x of shape (B, {self.input_dim}), got {tuple(x.shape)}")
        latents = self.latents.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        for block in self.cross_blocks:
            latents = block(latents, x)
        for block in self.self_blocks:
            latents = block(latents)
        return self.pool(latents)
