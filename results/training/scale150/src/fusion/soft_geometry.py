"""Differentiable surrogates of the scorer's geometry terms that MOVE with the predicted field (unlike the fixed-contour linearisations).

Convention: phi = axis_sign * psi so that the magnetic axis is a maximum (DIII-D axis_sign = -1).
Given the TRUE boundary level phi_b (from the training labels: boundary_flux of the geometry store) the predicted plasma region is
m = sigmoid((phi - phi_b) / tau_b); extents of that region are soft maxima/minima of the coordinates (log-sum-exp with temperature tau_x),
the axis is the softmax-weighted position of phi, volume is the R-weighted soft area.  All statistics are computed IDENTICALLY for the
prediction and for the truth, and the loss penalises the standardised difference, so the surrogate's own bias cancels to first order.
Verification against the official scorer (correlation of surrogate error with official error over frames) must pass before training.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor, nn


class SoftGeometry(nn.Module):
    NAMES = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "Zmax", "Zmin", "Rmax", "Rmin")

    def __init__(self, R, Z, mask, axis_sign: float, tau_axis: float, tau_b: float, tau_x: float = 0.02):
        super().__init__()
        R, Z = np.asarray(R, np.float64), np.asarray(Z, np.float64); RR, ZZ = np.meshgrid(R, Z)
        self.register_buffer("RR", torch.as_tensor(RR, dtype=torch.float32)); self.register_buffer("ZZ", torch.as_tensor(ZZ, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(np.asarray(mask, bool)))
        self.axis_sign, self.tau_axis, self.tau_b, self.tau_x = float(axis_sign), float(tau_axis), float(tau_b), float(tau_x)
        self.dA = float((R[1] - R[0]) * (Z[1] - Z[0]))

    def stats(self, psi: Tensor, phi_b: Tensor) -> Tensor:
        """psi (B,Z,R) physical; phi_b (B,) true boundary level in phi units -> (B, 10) soft statistics."""
        phi = self.axis_sign * psi.float(); B = phi.shape[0]; mask = self.mask
        neg = torch.finfo(phi.dtype).min / 4
        # axis: softmax over the envelope of phi / tau_axis (relative to the per-frame max for stability)
        logits = torch.where(mask[None], phi / self.tau_axis, torch.full_like(phi, neg)).flatten(1)
        w = torch.softmax(logits, dim=1)
        R_ax = (w * self.RR.flatten()[None]).sum(1); Z_ax = (w * self.ZZ.flatten()[None]).sum(1)
        # soft plasma region above the true boundary level
        m = torch.sigmoid((phi - phi_b[:, None, None]) / self.tau_b) * mask[None]
        mf = m.flatten(1).clamp_min(1e-12)
        def soft_max(coord):          # log-sum-exp over the region, weighted by membership
            c = coord.flatten()[None].expand(B, -1)
            return self.tau_x * torch.logsumexp(c / self.tau_x + torch.log(mf), dim=1)
        Zmax = soft_max(self.ZZ); Zmin = -soft_max(-self.ZZ); Rmax = soft_max(self.RR); Rmin = -soft_max(-self.RR)
        # R at the top / bottom: membership weighted toward the extreme Z
        wt = torch.softmax(((self.ZZ.flatten()[None] - Zmax[:, None]) / self.tau_x) + torch.log(mf), dim=1); R_top = (wt * self.RR.flatten()[None]).sum(1)
        wb = torch.softmax(((Zmin[:, None] - self.ZZ.flatten()[None]) / self.tau_x) + torch.log(mf), dim=1); R_bot = (wb * self.RR.flatten()[None]).sum(1)
        width = (Rmax - Rmin).clamp_min(1e-3); a = 0.5 * width; Rgeo = 0.5 * (Rmax + Rmin)
        kappa = (Zmax - Zmin) / width; tri_top = (Rgeo - R_top) / a; tri_bot = (Rgeo - R_bot) / a
        volume = 2 * np.pi * (m * self.RR[None]).flatten(1).sum(1) * self.dA
        return torch.stack([R_ax, Z_ax, kappa, tri_top, tri_bot, volume, Zmax, Zmin, Rmax, Rmin], dim=1)

    def forward(self, psi: Tensor, truth_stats: Tensor, phi_b: Tensor, weights: Tensor) -> Tensor:
        """Standardised squared differences of the first six statistics vs precomputed TRUE soft statistics; weights (6,) = 1/var."""
        s = self.stats(psi, phi_b)[:, :6]
        return (((s - truth_stats[:, :6]) ** 2) * weights[None, :6]).sum(1)


class SoftAxis(nn.Module):
    """Scorer-exact differentiable magnetic-axis position: softmax over the envelope of phi/tau, phi = axis_sign * psi.
    Verified on native dev (m00 r100): correlation of (soft - soft_true) with (official - official_true) = +0.999 (R) / +0.997 (Z) at tau = 0.01 psi_std."""

    def __init__(self, R, Z, mask, axis_sign: float, tau: float):
        super().__init__()
        R, Z = np.asarray(R, np.float64), np.asarray(Z, np.float64); RR, ZZ = np.meshgrid(R, Z)
        self.register_buffer("Rf", torch.as_tensor(RR.ravel(), dtype=torch.float32)); self.register_buffer("Zf", torch.as_tensor(ZZ.ravel(), dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(np.asarray(mask, bool).ravel())); self.axis_sign, self.tau = float(axis_sign), float(tau)

    def forward(self, psi: Tensor) -> Tensor:
        phi = (self.axis_sign * psi.float()).flatten(1) / self.tau
        logits = torch.where(self.mask[None], phi, torch.full_like(phi, torch.finfo(phi.dtype).min / 4))
        w = torch.softmax(logits, dim=1)
        return torch.stack([(w * self.Rf[None]).sum(1), (w * self.Zf[None]).sum(1)], dim=1)
