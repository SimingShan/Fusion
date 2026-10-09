"""Training-only local geometric linearization at the target axis and LCFS.

These fixed target locations are auxiliary supervision, never inference inputs.
The official scorer still independently extracts geometry from predictions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .li_loss import LiTargetStore


class FixedGeometryLoss(nn.Module):
    def __init__(self, axis_variance, boundary_length_scale=0.1):
        super().__init__()
        self.register_buffer("axis_variance", torch.as_tensor(axis_variance, dtype=torch.float64))
        self.boundary_length_scale = float(boundary_length_scale)

    def parts(self, psi, targets):
        work = psi if psi.dtype == torch.float64 else psi.float()
        stencil = work.flatten(1).gather(1, targets["axis_indices"].long())
        error = stencil - targets["axis_flux"].to(work)
        # Exact derivative of the scorer's three-point parabolic refinement
        # with respect to its five stencil values, evaluated at target flux.
        displacement = torch.einsum("bki,bi->bk", targets["axis_response"].to(work), error)
        axis = (displacement.square() / self.axis_variance.to(work)).mean()
        boundary = F.grid_sample(work[:, None], targets["sampling_grid"].to(work)[:, :, None],
                                 mode="bilinear", padding_mode="border", align_corners=True)[:, 0, :, 0]
        boundary_error = boundary - targets["boundary_flux"].to(work)
        segments = targets["segment_lengths"].to(work)
        point_weights = 0.5 * (segments + segments.roll(1, dims=1))
        point_weights = point_weights / point_weights.sum(1, keepdim=True)
        # The LCFS level is free to change: remove its constant flux offset.
        boundary_error = boundary_error - (boundary_error * point_weights).sum(1, keepdim=True)
        displacement = boundary_error / targets["boundary_gradient_floor"].to(work)
        lcfs = ((displacement / self.boundary_length_scale).square() * point_weights).sum(1).mean()
        return {"axis": axis, "lcfs": lcfs}

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

    def geometry_loss_module(self, device=None):
        return FixedGeometryLoss(self.metadata["geometry"]["axis_variance"],
                                 self.metadata["geometry"]["boundary_length_scale"]).to(device)

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
        return positions, targets
