"""Training-only local geometric linearization at the target axis and LCFS.

These fixed target locations are auxiliary supervision, never inference inputs.
The official scorer still independently extracts geometry from predictions.

Per-frame form
--------------
``FixedGeometryLoss.per_frame`` returns one (B,) vector per term so that the
trainer can apply per-frame weights (for example amplitude weights) exactly as
it does for the li term. ``parts``/``forward`` keep the original unweighted
reductions bit-identical when the optional emphasis terms are off.

Optional emphasis terms (all off by default)
--------------------------------------------
* ``boundary_extremal_weight`` W: the LCFS displacement loss weights each
  contour point by ``w_i = 1 + W * (g_top_i + g_bot_i + g_out_i + g_in_i)``
  where each g is a Gaussian bump (``boundary_extremal_sigma`` metres of arc
  length along the closed target contour, ~6 of 512 points for 0.06 m on a
  5.2 m DIII-D perimeter) centred on the target contour's argmax Z, argmin Z,
  argmax R and argmin R vertices. Weights are normalised so the arc-length
  weighted mean point weight is 1 (mean over points for the near-uniform
  official 512-point resampling). This targets the scorer's kappa (LCFS Z
  extents, the dominant kappa error source and singular at the lower x-point),
  triangularity (R at the Z extremes relative to the R extents) and volume
  (whose leading error is the extent of the boundary). The free LCFS level
  offset is still removed with the plain arc-length weights.
* ``axis_margin_ratio`` rho with ``per_frame(..., margin=True)``: peak-cell
  margin hinge. The scorer takes the strongest interior local maximum of
  phi = -psi (DIII-D) as the integer axis cell and refines it with a 1-D
  three-point parabola; when the predicted integer peak lands in a different
  cell from the target the refined axis jumps by up to a cell, which is the
  main R_axis error source. For each of the 4 stencil neighbours k (R-1, R+1,
  Z-1, Z+1; the stored 5-point stencil has no diagonal neighbours, so only
  these 4 are used) with target margin m_k = phi_gt[peak] - phi_gt[k] >= 0 the
  hinge requires phi_pred[peak] - phi_pred[k] >= rho * m_k and penalises
  ``mean_k relu(rho*m_k - (phi_pred[peak]-phi_pred[k]))^2 / margin_scale^2``
  where ``margin_scale`` is the RMS target margin over the store (computed
  once). Frames whose target peak is nearly on a cell edge (m_k ~ 0) are
  thereby pushed to keep the ordering; frames with a large margin keep at
  least half of it.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .li_loss import LiTargetStore


# Development-fold variances of the scored scalars, so the surrogates below are expressed in the
# metric's own units (its per-scalar weight is (0.20/7)/Var_j).
KAPPA_VARIANCE = 0.011183
VOLUME_VARIANCE = 1.70287


class FixedGeometryLoss(nn.Module):
    def __init__(self, axis_variance, boundary_length_scale=0.1, boundary_extremal_weight=0.0,
                 boundary_extremal_sigma=0.06, axis_margin_ratio=0.5, margin_scale=None):
        super().__init__()
        self.register_buffer("axis_variance", torch.as_tensor(axis_variance, dtype=torch.float64))
        self.boundary_length_scale = float(boundary_length_scale)
        self.boundary_extremal_weight = float(boundary_extremal_weight)
        self.boundary_extremal_sigma = float(boundary_extremal_sigma)
        self.axis_margin_ratio = float(axis_margin_ratio)
        self.margin_scale = None if margin_scale is None else float(margin_scale)
        if self.boundary_extremal_weight < 0 or self.boundary_extremal_sigma <= 0:
            raise ValueError("Extremal weight must be nonnegative and its sigma positive")
        if not 0 <= self.axis_margin_ratio <= 1:
            raise ValueError("Axis margin ratio must lie in [0, 1]")
        if self.margin_scale is not None and not self.margin_scale > 0:
            raise ValueError("Axis margin scale must be positive")

    def extremal_point_weights(self, contours, segments, point_weights):
        """(B, points) multiplicative weights, arc-length-weighted mean 1 per frame."""
        # Arc length position of each point on the closed target contour;
        # segments[i] is the length from point i to point i+1 (as in li_loss).
        perimeter = segments.sum(1, keepdim=True)
        position = torch.cumsum(segments, dim=1) - segments
        vertices = torch.stack((contours[..., 1].argmax(1), contours[..., 1].argmin(1),
                                contours[..., 0].argmax(1), contours[..., 0].argmin(1)), dim=1)
        anchor = position.gather(1, vertices)
        distance = (position[:, None, :] - anchor[:, :, None]).abs()
        distance = torch.minimum(distance, perimeter[:, :, None] - distance)
        bumps = torch.exp(-0.5 * (distance / self.boundary_extremal_sigma).square()).sum(1)
        weights = 1.0 + self.boundary_extremal_weight * bumps
        return weights / (weights * point_weights).sum(1, keepdim=True)

    def _pieces(self, psi, targets, margin=False):
        work = psi if psi.dtype == torch.float64 else psi.float()
        stencil = work.flatten(1).gather(1, targets["axis_indices"].long())
        axis_flux = targets["axis_flux"].to(work)
        error = stencil - axis_flux
        # Exact derivative of the scorer's three-point parabolic refinement
        # with respect to its five stencil values, evaluated at target flux.
        displacement = torch.einsum("bki,bi->bk", targets["axis_response"].to(work), error)
        axis_terms = displacement.square() / self.axis_variance.to(work)
        boundary = F.grid_sample(work[:, None], targets["sampling_grid"].to(work)[:, :, None],
                                 mode="bilinear", padding_mode="border", align_corners=True)[:, 0, :, 0]
        boundary_error = boundary - targets["boundary_flux"].to(work)
        segments = targets["segment_lengths"].to(work)
        point_weights = 0.5 * (segments + segments.roll(1, dims=1))
        point_weights = point_weights / point_weights.sum(1, keepdim=True)
        # The LCFS level is free to change: remove its constant flux offset.
        boundary_error = boundary_error - (boundary_error * point_weights).sum(1, keepdim=True)
        displacement = boundary_error / targets["boundary_gradient_floor"].to(work)
        squared = (displacement / self.boundary_length_scale).square()
        if self.boundary_extremal_weight:
            squared = squared * self.extremal_point_weights(targets["contours"].to(work), segments, point_weights)
        lcfs = (squared * point_weights).sum(1)
        pieces = {"axis_terms": axis_terms, "lcfs": lcfs}
        # Differentiable surrogates for two SCORED shape functionals, in the metric's own units.
        # A boundary perturbation moves the surface along its normal by d = -dpsi/|grad psi|, which
        # is exactly `displacement` above (its global sign squares away below). The scorer's
        # elongation is kappa = (Zmax-Zmin)/(Rmax-Rmin), so to first order
        #     dkappa = [(d_top + d_bot) - kappa (d_right + d_left)] / width
        # at the four extremal vertices; and the toroidal volume moves by the line integral
        #     dV = sum_i 2 pi R_i d_i ds_i.
        # Unlike the separatrix-band pixel weighting (measured as no gain), this is not a positive
        # reweighting of the pixel L2 inside the basis span: it is a signed contrast, so it can
        # cancel errors the L2 cannot distinguish.
        contours = targets["contours"].to(work)
        Rc, Zc = contours[..., 0], contours[..., 1]
        i_top, i_bot = Zc.argmax(1), Zc.argmin(1)
        i_right, i_left = Rc.argmax(1), Rc.argmin(1)
        gather = lambda a, i: a.gather(1, i[:, None]).squeeze(1)
        width = gather(Rc, i_right) - gather(Rc, i_left)
        height = gather(Zc, i_top) - gather(Zc, i_bot)
        kappa_gt = height / width.clamp_min(1e-6)
        d_top, d_bot = gather(displacement, i_top), gather(displacement, i_bot)
        d_right, d_left = gather(displacement, i_right), gather(displacement, i_left)
        d_kappa = ((d_top + d_bot) - kappa_gt * (d_right + d_left)) / width.clamp_min(1e-6)
        d_volume = (2.0 * math.pi * Rc * displacement * segments).sum(1)
        # VERIFIED 2026-09-28 against the official scorer under realistic coefficient noise
        # (results/analysis/shape_surrogate_105): the VOLUME surrogate tracks the scorer at signed
        # corr +0.83, but the ELONGATION surrogate is corr -0.10, i.e. useless. The cause is
        # structural, not a coding slip: boundary_gradient_floor is deliberately clamped at 10% of
        # its mean to keep the displacement linearization bounded near the saddle, and one of
        # elongation's two vertical vertices is always at that saddle, where ~64% of the Zmax/Zmin
        # error lives. Reaching it needs the inverse-Hessian saddle-response formulation, not this.
        # The kappa piece is therefore kept OUT of the trained loss; do not enable it without
        # re-running the verification.
        pieces["shape_kappa"] = d_kappa.square() / KAPPA_VARIANCE
        pieces["shape_volume"] = d_volume.square() / VOLUME_VARIANCE
        if margin:
            if self.margin_scale is None:
                raise ValueError("Axis margin hinge requires the store's margin scale")
            # phi = -psi for DIII-D: margins are psi[neighbour] - psi[peak].
            target_margin = self.axis_margin_ratio * (axis_flux[:, 1:] - axis_flux[:, :1])
            actual_margin = stencil[:, 1:] - stencil[:, :1]
            hinge = F.relu(target_margin - actual_margin).square().mean(1)
            pieces["axis_margin"] = hinge / (self.margin_scale ** 2)
        return pieces

    def per_frame(self, psi, targets, margin=False):
        """Per-frame (B,) loss vectors: axis, lcfs and optionally axis_margin."""
        pieces = self._pieces(psi, targets, margin=margin)
        result = {"axis": pieces["axis_terms"].mean(1), "lcfs": pieces["lcfs"],
                  "shape": pieces["shape_volume"]}   # kappa excluded: fails verification
        if margin:
            result["axis_margin"] = pieces["axis_margin"]
        return result

    def parts(self, psi, targets):
        pieces = self._pieces(psi, targets)
        return {"axis": pieces["axis_terms"].mean(), "lcfs": pieces["lcfs"].mean(),
                "shape": pieces["shape_volume"].mean()}   # kappa excluded: fails verification

    def forward(self, psi, targets):
        parts = self.parts(psi, targets)
        return parts["axis"] + parts["lcfs"]


class ExactAxisLoss(nn.Module):
    """Differentiate the scorer's actual piecewise-smooth axis functional."""

    def __init__(self, R, Z, mask, axis_variance):
        super().__init__()
        self.register_buffer("R", torch.as_tensor(np.asarray(R), dtype=torch.float64))
        self.register_buffer("Z", torch.as_tensor(np.asarray(Z), dtype=torch.float64))
        self.register_buffer("axis_variance", torch.as_tensor(axis_variance, dtype=torch.float64))
        mask = np.asarray(mask, bool)
        eroded = mask.copy()
        for _ in range(2):
            step = eroded.copy()
            step[1:] &= eroded[:-1]; step[:-1] &= eroded[1:]
            step[:, 1:] &= eroded[:, :-1]; step[:, :-1] &= eroded[:, 1:]
            eroded = step
        eroded[[0, -1], :] = False; eroded[:, [0, -1]] = False
        self.register_buffer("mask", torch.from_numpy(mask.copy()))
        self.register_buffer("local_mask", torch.from_numpy(eroded))

    def from_stencil(self, centers, values):
        # The official functional promotes fields before parabolic refinement.
        # Five promoted values reproduce that arithmetic without a full copy.
        values = values.double()
        iz = torch.div(centers, len(self.R), rounding_mode="floor")
        ir = centers % len(self.R)
        result = []
        for left, right, coordinate, location in ((1, 2, self.R, ir), (3, 4, self.Z, iz)):
            denominator = values[:, left]-2*values[:, 0]+values[:, right]
            safe = torch.where(denominator == 0, 1., denominator)
            shift = (.5*(values[:, left]-values[:, right])/safe).clamp(-1., 1.)
            shift = torch.where((denominator != 0) & (location > 0) & (location < len(coordinate)-1), shift, 0.)
            result.append(coordinate[location]+shift*(coordinate[1]-coordinate[0]))
        return torch.stack(result, dim=1)

    def predict(self, psi):
        if psi.ndim != 3 or tuple(psi.shape[1:]) != tuple(self.mask.shape):
            raise ValueError("Expected native-grid DIII-D fields")
        phi = -psi
        maxima = (phi >= F.max_pool2d(phi[:, None], 3, stride=1, padding=1)[:, 0]) & self.local_mask
        has_local = maxima.flatten(1).any(1)
        eligible = torch.where(has_local[:, None, None], maxima, self.mask)
        centers = torch.where(eligible, phi, -torch.inf).flatten(1).argmax(1)
        iz = torch.div(centers, len(self.R), rounding_mode="floor"); ir = centers % len(self.R)
        indices = torch.stack((centers, torch.where(ir > 0, centers-1, centers),
            torch.where(ir < len(self.R)-1, centers+1, centers),
            torch.where(iz > 0, centers-len(self.R), centers),
            torch.where(iz < len(self.Z)-1, centers+len(self.R), centers)), dim=1)
        return self.from_stencil(centers, phi.flatten(1).gather(1, indices))

    def per_frame(self, psi, targets):
        """Per-frame (B,) variance-normalised squared axis error (mean over R, Z)."""
        reference = self.from_stencil(targets["axis_indices"][:, 0].long(), targets["axis_flux"])
        return ((self.predict(psi)-reference).square()/self.axis_variance).mean(1)

    def forward(self, psi, targets):
        reference = self.from_stencil(targets["axis_indices"][:, 0].long(), targets["axis_flux"])
        return ((self.predict(psi)-reference).square()/self.axis_variance).mean()


class GeometryTargetStore(LiTargetStore):
    def __init__(self, directory):
        super().__init__(directory)
        if not self.metadata.get("geometry", {}).get("complete"):
            raise ValueError("Geometry targets are not complete")
        self.geometry_arrays = {
            key: np.load(self.directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in ("axis_indices", "axis_flux", "axis_response", "boundary_flux", "boundary_gradient_floor")
        }

    def axis_margin_scale(self):
        """RMS target peak margin phi[peak]-phi[neighbour] over valid frames (psi units)."""
        rows = np.flatnonzero(self.arrays["valid"])
        flux = np.asarray(self.geometry_arrays["axis_flux"][rows], dtype=np.float64)
        margins = flux[:, 1:] - flux[:, :1]
        if margins.size == 0 or np.any(margins < 0):
            raise ValueError("Target axis stencil must peak in phi = -psi at its centre")
        return float(np.sqrt(np.mean(np.square(margins))))

    def geometry_loss_module(self, device=None, boundary_extremal_weight=0.0, boundary_extremal_sigma=0.06,
                             axis_margin_ratio=0.5, axis_margin=False):
        return FixedGeometryLoss(self.metadata["geometry"]["axis_variance"],
                                 self.metadata["geometry"]["boundary_length_scale"],
                                 boundary_extremal_weight=boundary_extremal_weight,
                                 boundary_extremal_sigma=boundary_extremal_sigma,
                                 axis_margin_ratio=axis_margin_ratio,
                                 margin_scale=self.axis_margin_scale() if axis_margin else None).to(device)

    def exact_axis_loss_module(self, device=None):
        path = Path(__file__).resolve().parents[1] / "starter/fusion_scoring/masks/d3d_envelope.npz"
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["grid_R"], self.R) or not np.array_equal(saved["grid_Z"], self.Z):
                raise ValueError("Axis supervision grid differs from the official mask")
            mask = saved["mask_coarse"]
        return ExactAxisLoss(self.R, self.Z, mask, self.metadata["geometry"]["axis_variance"]).to(device)

    def batch(self, global_frame_indices, device=None):
        positions, targets = super().batch(global_frame_indices, device)
        if not len(positions):
            return positions, targets
        indices = global_frame_indices.detach().cpu().numpy() if isinstance(global_frame_indices, torch.Tensor) else np.asarray(global_frame_indices)
        rows = self.lookup[indices]
        rows = rows[rows >= 0]
        for key, values in self.geometry_arrays.items():
            targets[key] = torch.as_tensor(np.asarray(values[rows]), device=positions.device)
        # Physical target contour vertices (extremal emphasis); training targets only.
        targets["contours"] = torch.as_tensor(np.asarray(self.arrays["contours"][rows]), device=positions.device)
        return positions, targets


def band_pixel_weights(inside, axis_centers, band_weight=8.0, core_weight=8.0, band_cells=2, core_cells=2):
    """Pixel weights ``w = 1 + band_weight * BAND + core_weight * CORE`` on the (B, Z, R) grid.

    ``BAND = dilate(inside, band_cells) & ~erode(inside, band_cells)`` is the ring
    of cells within ``band_cells`` (Chebyshev distance) of the target LCFS on
    either side, from max pooling of the float inside mask (dilation) and of the
    inverted mask (erosion). ``CORE`` is the ``(2*core_cells+1)**2`` patch centred
    on the target peak cell ``axis_centers`` (flattened ``iz*n_R+ir`` indices, the
    first entry of the stored axis stencil). Returns float32 ``weights`` and the
    boolean ``band`` and ``core`` masks. Training targets only.
    """
    mask = inside.to(torch.float32)
    if mask.ndim != 3:
        raise ValueError("Expected inside masks of shape (B, Z, R)")
    band_cells, core_cells = int(band_cells), int(core_cells)
    if band_cells < 0 or core_cells < 0:
        raise ValueError("Band and core half-widths must be nonnegative")
    n_Z, n_R = mask.shape[1:]
    kernel = 2 * band_cells + 1
    dilated = F.max_pool2d(mask[:, None], kernel, stride=1, padding=band_cells)[:, 0]
    eroded = 1.0 - F.max_pool2d(1.0 - mask[:, None], kernel, stride=1, padding=band_cells)[:, 0]
    band = (dilated > 0.5) & (eroded < 0.5)
    centers = axis_centers.long()
    iz, ir = torch.div(centers, n_R, rounding_mode="floor"), centers % n_R
    zz = torch.arange(n_Z, device=mask.device)[None, :, None]
    rr = torch.arange(n_R, device=mask.device)[None, None, :]
    core = ((zz - iz[:, None, None]).abs() <= core_cells) & ((rr - ir[:, None, None]).abs() <= core_cells)
    weights = 1.0 + float(band_weight) * band.to(torch.float32) + float(core_weight) * core.to(torch.float32)
    return weights, band, core
