"""Operator baselines for flux maps and two equilibrium scalars.

DeepONet uses a learned branch--coordinate-trunk inner product (Lu et al.,
Nature Machine Intelligence 2021, https://doi.org/10.1038/s42256-021-00302-5).
FourierDeepONet multiplies branch/trunk features before an FFT-based FNO decoder,
adapting Zhu et al. 2023 (https://doi.org/10.1016/j.cma.2023.116300) to spatial
coordinates and a lightweight plain-FNO decoder; this is not the paper's FWI
source-parameter/U-FNO configuration. Inputs/targets are normalized externally.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(input_dim: int, output_dim: int, width: int, depth: int,
         activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(input_dim, width), activation()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class _OperatorBase(nn.Module):
    """Shared MLP encoders; supplied coordinates must use training normalization."""

    def __init__(self, input_dim: int, width: int = 128, latent_dim: int = 64,
                 depth: int = 3, grid_shape: Sequence[int] = (65, 65)) -> None:
        super().__init__()
        if min(input_dim, width, latent_dim, depth) < 1:
            raise ValueError("input_dim, width, latent_dim, and depth must be positive")
        if len(grid_shape) != 2 or min(grid_shape) < 2:
            raise ValueError("grid_shape must contain two dimensions of at least 2")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        # Flux maps have Z rows and R columns; flattened points are ordered (R,Z).
        zz, rr = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.grid_shape[0]),
            torch.linspace(-1.0, 1.0, self.grid_shape[1]), indexing="ij",
        )
        self.register_buffer("coords", torch.stack((rr, zz), dim=-1).reshape(-1, 2))
        self.branch = _mlp(input_dim, latent_dim, width, depth)
        self.trunk = _mlp(2, latent_dim, width, depth, nn.Tanh)
        self.scalar_head = _mlp(latent_dim, 2, width, depth=1)

    def _encode(self, x: Tensor, coords: Tensor | None) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}); got {tuple(x.shape)}")
        points = self.coords if coords is None else coords
        # Accept shared flattened/grid coordinates or one flattened grid per sample.
        if tuple(points.shape) == (*self.grid_shape, 2):
            points = points.reshape(-1, 2)
        n_points = math.prod(self.grid_shape)
        if points.ndim == 2 and tuple(points.shape) == (n_points, 2):
            pass
        elif points.ndim == 3 and tuple(points.shape) == (x.shape[0], n_points, 2):
            pass
        else:
            raise ValueError(
                f"Expected coordinates ({n_points}, 2), {(*self.grid_shape, 2)}, "
                f"or (B, {n_points}, 2); got {tuple(points.shape)}"
            )
        points = points.to(device=x.device, dtype=x.dtype)
        return self.branch(x), self.trunk(points)


class DeepONet(_OperatorBase):
    """Learned branch/trunk dot product; returns psi (B,Z,R), scalars (B,2)."""

    def __init__(self, input_dim: int, width: int = 128, latent_dim: int = 64,
                 depth: int = 3, grid_shape: Sequence[int] = (65, 65)) -> None:
        super().__init__(input_dim, width, latent_dim, depth, grid_shape)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, coords: Tensor | None = None) -> tuple[Tensor, Tensor]:
        branch, trunk = self._encode(x, coords)
        if trunk.ndim == 2:
            psi = branch @ trunk.T
        else:
            psi = torch.einsum("bk,bnk->bn", branch, trunk)
        psi = psi / math.sqrt(self.latent_dim) + self.bias
        return psi.reshape(x.shape[0], *self.grid_shape), self.scalar_head(branch)


class SpectralConv2d(nn.Module):
    """Truncated 2-D Fourier convolution with independent +/- Z mode weights."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        if channels < 1 or modes < 1:
            raise ValueError("channels and modes must be positive")
        self.modes = int(modes)
        # Store real/imaginary parts explicitly for portable checkpoints/optimizers.
        shape = (channels, channels, modes, modes, 2)
        scale = 1.0 / math.sqrt(channels)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        # Float32 FFT also supports non-power-of-two grids under CUDA autocast.
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        spectrum = torch.fft.rfft2(x.to(work_dtype), norm="ortho")
        modes_z = min(self.modes, height // 2)
        modes_r = min(self.modes, width // 2 + 1)
        output = spectrum.new_zeros(spectrum.shape)
        positive = torch.view_as_complex(self.weight_positive.to(work_dtype).contiguous())
        negative = torch.view_as_complex(self.weight_negative.to(work_dtype).contiguous())
        output[:, :, :modes_z, :modes_r] = torch.einsum(
            "bizr,iozr->bozr", spectrum[:, :, :modes_z, :modes_r],
            positive[:, :, :modes_z, :modes_r],
        )
        output[:, :, -modes_z:, :modes_r] = torch.einsum(
            "bizr,iozr->bozr", spectrum[:, :, -modes_z:, :modes_r],
            negative[:, :, :modes_z, :modes_r],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho").to(x.dtype)


class _FourierBlock(nn.Module):
    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes)
        self.local = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.spectral(x) + self.local(x))


class FourierDeepONet(_OperatorBase):
    """Branch/trunk multiplication followed by a trainable spectral decoder.

    The decoder acts on a structured (Z,R) grid. Explicit coordinates must retain
    that grid order and spacing. Padding reduces periodic-boundary interactions;
    seven cells give efficient 72x72 FFTs for the default 65x65 output grid.
    """

    def __init__(self, input_dim: int, width: int = 128, latent_dim: int = 64,
                 depth: int = 3, grid_shape: Sequence[int] = (65, 65),
                 modes: int = 12, spectral_width: int = 24,
                 spectral_layers: int = 3, padding: int = 7) -> None:
        super().__init__(input_dim, width, latent_dim, depth, grid_shape)
        if min(modes, spectral_width, spectral_layers) < 1 or padding < 0:
            raise ValueError("Spectral dimensions must be positive and padding nonnegative")
        self.padding = int(padding)
        self.lift = nn.Linear(latent_dim, spectral_width)
        self.decoder = nn.Sequential(*[
            _FourierBlock(spectral_width, modes) for _ in range(spectral_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(spectral_width, 2 * spectral_width, 1), nn.GELU(),
            nn.Conv2d(2 * spectral_width, 1, 1),
        )

    def forward(self, x: Tensor, coords: Tensor | None = None) -> tuple[Tensor, Tensor]:
        branch, trunk = self._encode(x, coords)
        if trunk.ndim == 2:
            trunk = trunk.unsqueeze(0)
        merged = branch.unsqueeze(1) * trunk
        field = self.lift(merged).transpose(1, 2).reshape(x.shape[0], -1, *self.grid_shape)
        if self.padding:
            field = F.pad(field, (0, self.padding, 0, self.padding))
        field = self.decoder(field)
        field = field[..., :self.grid_shape[0], :self.grid_shape[1]]
        psi = self.project(field).squeeze(1)
        return psi, self.scalar_head(branch)


def build_model(name: str, input_dim: int, **kwargs) -> nn.Module:
    """Build 'deeponet' or 'fourier_deeponet' using a common config dictionary.

    Fourier-only options are accepted and discarded for vanilla DeepONet so
    experiments can share one configuration. Other unknown options raise errors.
    """
    normalized = name.lower().replace("-", "_")
    if normalized == "deeponet":
        for option in ("modes", "spectral_width", "spectral_layers", "padding"):
            kwargs.pop(option, None)
        return DeepONet(input_dim, **kwargs)
    if normalized in ("fourier_deeponet", "fourierdeeponet"):
        return FourierDeepONet(input_dim, **kwargs)
    raise ValueError(f"Unknown model {name!r}; choose 'deeponet' or 'fourier_deeponet'")
