"""Differentiable internal-inductance supervision on fixed training contours.

The official scorer extracts a separate LCFS from every predicted flux map.
This surrogate holds the *training target's* LCFS fixed, so it differentiates
through the field profile but not through contour motion. At the target field
it reproduces the official li functional. It supplements flux/geometry losses;
it must never replace official validation with a surrogate validation score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def contour_to_sampling_grid(contours: np.ndarray, R: np.ndarray,
                             Z: np.ndarray) -> np.ndarray:
    """Match the scorer's physical-coordinate -> fractional-grid interpolation."""
    contours = np.asarray(contours, dtype=np.float64)
    rr = np.interp(contours[..., 0], R, np.arange(len(R)))
    zz = np.interp(contours[..., 1], Z, np.arange(len(Z)))
    return np.stack((2 * rr / (len(R) - 1) - 1,
                     2 * zz / (len(Z) - 1) - 1), axis=-1)


class FixedContourLiLoss(nn.Module):
    """Scorer-compatible field functional on supplied, fixed target contours.

    ``psi`` has physical units and shape (B, Z, R). ``targets`` contains:
    ``inside`` (B, Z, R), ``sampling_grid`` (B, points, 2),
    ``segment_lengths`` (B, points), and ``li`` (B,).
    Only invalid *ground-truth* references may be omitted by the target store.
    Finite but poor predictions are retained and are never plausibility-clipped.

    R and Z are the native physical coordinate arrays. Calculations preserve
    float64 input; other inputs use float32 so the functional is safe under AMP.
    """

    def __init__(self, R, Z, li_variance: float, denominator_floor: float = 1e-12):
        super().__init__()
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        if R.ndim != 1 or Z.ndim != 1 or min(len(R), len(Z)) < 3:
            raise ValueError("R and Z must be one-dimensional coordinate arrays of length >= 3")
        if not np.isfinite(R).all() or not np.isfinite(Z).all() or np.any(R <= 0):
            raise ValueError("Coordinates must be finite and R must be positive")
        if np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0):
            raise ValueError("Coordinates must be strictly increasing")
        if not np.isfinite(li_variance) or li_variance <= 0 or denominator_floor <= 0:
            raise ValueError("li variance and denominator floor must be positive")
        self.register_buffer("R", torch.as_tensor(R))
        self.register_buffer("Z", torch.as_tensor(Z))
        self.register_buffer("li_variance", torch.tensor(float(li_variance), dtype=torch.float64))
        self.denominator_floor = float(denominator_floor)

    def predict(self, psi: Tensor, targets: dict[str, Tensor]) -> Tensor:
        if psi.ndim != 3 or tuple(psi.shape[1:]) != (len(self.Z), len(self.R)):
            raise ValueError("Expected psi shape (B, len(Z), len(R))")
        if not len(psi):
            return psi.new_empty((0,))
        # torch.gradient uses the same centered interior / one-sided boundary
        # first derivatives as np.gradient(..., Z, R), including nonuniform grids.
        work = psi if psi.dtype == torch.float64 else psi.float()
        R, Z = self.R.to(work), self.Z.to(work)
        dZ, dR = torch.gradient(work, spacing=(Z, R), dim=(1, 2), edge_order=1)
        Bp = torch.linalg.vector_norm(torch.stack((dR, dZ), dim=0), dim=0) / R[None, None]
        inside = targets["inside"].to(work)
        volume_weights = inside * R[None, None]
        volume = volume_weights.sum(dim=(1, 2))
        if torch.any(volume <= 0):
            raise ValueError("Target contours must contain a nonempty plasma volume")
        energy_mean = (Bp.square() * volume_weights).sum(dim=(1, 2)) / volume
        # Interpolate Bp *after* taking its gridwise norm, as in the scorer.
        grid = targets["sampling_grid"].to(work)[:, :, None, :]
        boundary_Bp = F.grid_sample(Bp[:, None], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)[:, 0, :, 0]
        segments = targets["segment_lengths"].to(work)
        perimeter = segments.sum(dim=1)
        if torch.any(perimeter <= 0):
            raise ValueError("Target contours must have positive perimeter")
        boundary_mean = (0.5 * (boundary_Bp + boundary_Bp.roll(-1, dims=1)) * segments).sum(dim=1) / perimeter
        return energy_mean / boundary_mean.clamp_min(self.denominator_floor).square()

    def forward(self, psi: Tensor, targets: dict[str, Tensor]) -> Tensor:
        if not len(psi):
            return psi.sum() * 0.0
        li = self.predict(psi, targets)
        truth = targets["li"].to(li)
        return (li - truth).square().mean() / self.li_variance.to(li)


class LiTargetStore:
    """Compact memory-mapped training references indexed by global frame index.

    ``positions, targets = store.batch(training_indices, device)`` returns only
    the supervised, valid rows in the optimization batch. Reconstruct model
    flux at ``positions`` and pass it to ``store.loss_module(device)``.
    Ground-truth contours, masks, and li values are training targets only; none
    of these files is needed or read by inference.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if not self.metadata.get("complete"):
            raise ValueError("li target preparation is not complete")
        if self.metadata.get("source_split") != "train":
            raise ValueError("Only training targets are permitted")
        self.R = np.load(self.directory / "R.npy", allow_pickle=False)
        self.Z = np.load(self.directory / "Z.npy", allow_pickle=False)
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in ("global_frame_indices", "valid", "inside_packed", "contours", "segment_lengths", "li")}
        indices = self.arrays["global_frame_indices"]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate supervised frame indices")
        self.lookup = np.full(self.metadata["total_training_frames"], -1, dtype=np.int64)
        valid_rows = np.flatnonzero(self.arrays["valid"])
        self.lookup[indices[valid_rows]] = valid_rows

    def loss_module(self, device=None) -> FixedContourLiLoss:
        return FixedContourLiLoss(self.R, self.Z, self.metadata["li_variance"]).to(device)

    def batch(self, global_frame_indices, device=None) -> tuple[Tensor, dict[str, Tensor]]:
        if isinstance(global_frame_indices, Tensor):
            if device is None:
                device = global_frame_indices.device
            indices = global_frame_indices.detach().cpu().numpy()
        else:
            indices = np.asarray(global_frame_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Training frame indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= len(self.lookup)):
            raise ValueError("Training frame index is out of bounds")
        rows = self.lookup[indices]
        positions = np.flatnonzero(rows >= 0)
        rows = rows[positions]
        positions_tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
        if not len(rows):
            return positions_tensor, {}
        inside = np.unpackbits(self.arrays["inside_packed"][rows], axis=1,
                               count=len(self.R) * len(self.Z), bitorder="little")
        values = {
            "inside": inside.reshape(-1, len(self.Z), len(self.R)),
            "sampling_grid": contour_to_sampling_grid(self.arrays["contours"][rows], self.R, self.Z),
            "segment_lengths": np.asarray(self.arrays["segment_lengths"][rows]),
            "li": np.asarray(self.arrays["li"][rows]),
        }
        return positions_tensor, {key: torch.as_tensor(value, device=device) for key, value in values.items()}
