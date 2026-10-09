"""Small FiLM-conditioned U-Net for the Challenge-2 residual flux field.

Input ``(B, C_in, 65, 65)`` dimensionless channels plus TWO conditioning vectors:
``cond`` (field conditioning, ``fusion.c2_fields_data.FIELD_COND_NAMES``) drives the
FiLM layers of the trunk and the per-frame gauge offset; ``scalar_cond``
(``SCALAR_COND_NAMES``, a superset that adds the Bt-derived ratio) reaches ONLY the
z-scored ``q95``/``betaN`` heads, so a quantity that extrapolates on MAST can never
modulate the field. Output: residual field ``(B, 65, 65)`` in units of
``F = mu0 |Ip| L / 2``, offset ``(B,)`` and scalars ``(B, n_scalars)``. The 65x65 grid
is replicate-padded to 72x72 so three stride-2 levels divide evenly
(72 -> 36 -> 18 -> 9). Deterministic, float32, GroupNorm + GELU.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

PAD = (3, 4, 3, 4)  # left, right, top, bottom: 65 -> 72


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        gamma, beta = self.linear(cond).chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.film(h, cond)
        return F.gelu(h + self.skip(x))


class C2UNet(nn.Module):
    def __init__(self, in_channels: int = 8, cond_dim: int = 5, scalar_cond_dim: int = 9, base: int = 32,
                 levels: int = 3, cond_hidden: int = 64, head_hidden: int = 128, n_scalars: int = 2):
        super().__init__()
        if levels not in (2, 3):
            raise ValueError("levels must be 2 or 3 for the 72x72 padded grid")
        self.in_channels, self.cond_dim, self.scalar_cond_dim, self.levels = in_channels, cond_dim, scalar_cond_dim, levels
        widths = [base * 2 ** i for i in range(levels + 1)]
        self.cond_net = nn.Sequential(nn.Linear(cond_dim, cond_hidden), nn.GELU(), nn.Linear(cond_hidden, cond_hidden), nn.GELU())
        # Head-only conditioning (Bt-derived entries included); never touches the trunk.
        self.scalar_cond_net = nn.Sequential(nn.Linear(scalar_cond_dim, cond_hidden), nn.GELU())
        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths[:-1]:
            self.encoders.append(ConvBlock(prev, w, cond_hidden))
            prev = w
        self.bottleneck = ConvBlock(prev, widths[-1], cond_hidden)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(nn.Conv2d(prev, w, 3, padding=1))
            self.decoders.append(ConvBlock(2 * w, w, cond_hidden))
            prev = w
        self.out = nn.Conv2d(prev, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.offset_head = nn.Sequential(nn.Linear(widths[-1] + cond_hidden, head_hidden), nn.GELU(),
                                         nn.Linear(head_hidden, 1))
        self.scalar_head = nn.Sequential(nn.Linear(widths[-1] + 2 * cond_hidden, head_hidden), nn.GELU(),
                                         nn.Linear(head_hidden, n_scalars))
        for head in (self.offset_head, self.scalar_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.n_scalars = n_scalars

    def forward(self, channels: torch.Tensor, cond: torch.Tensor, scalar_cond: torch.Tensor) -> dict:
        if cond.shape[1] != self.cond_dim or scalar_cond.shape[1] != self.scalar_cond_dim:
            raise ValueError(f"conditioning dims {tuple(cond.shape)}/{tuple(scalar_cond.shape)} do not match "
                             f"the model ({self.cond_dim}/{self.scalar_cond_dim})")
        c = self.cond_net(cond)
        x = F.pad(channels, PAD, mode="replicate")
        skips = []
        for enc in self.encoders:
            x = enc(x, c)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottleneck(x, c)
        pooled = x.mean(dim=(2, 3))
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(F.interpolate(x, scale_factor=2, mode="nearest"))
            x = dec(torch.cat([x, skip], dim=1), c)
        field = self.out(x)[:, 0, PAD[2]:PAD[2] + 65, PAD[0]:PAD[0] + 65]
        offset = self.offset_head(torch.cat([pooled, c], dim=1))[:, 0]
        scalars = self.scalar_head(torch.cat([pooled, c, self.scalar_cond_net(scalar_cond)], dim=1))
        return {"field": field, "offset": offset, "scalars": scalars}


def build_c2_unet(**kwargs) -> C2UNet:
    return C2UNet(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
