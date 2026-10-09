"""Amplitude-aware per-frame training loss weights for POD coefficient targets.

Ramp-down and low-|Ip| frames have a small axis-to-boundary flux range
(~0.1 Wb/rad against 0.4-0.6 Wb/rad in flat-top), so a pooled coefficient
loss gives them almost no gradient although they carry most of the internal
inductance error and a large share of the axis/shape errors. These weights
up-weight small-amplitude target frames::

    s_f   = peak-to-peak over the 65x65 grid of the rank-k reconstruction
            psi_mean + (c_f * coefficient_std) @ basis
    s_ref = median s_f over the frames actually optimized
    w_f   = clip((s_ref / s_f) ** power, 1, max_weight), rescaled to mean 1
            over the frames actually optimized

Only the *training* coefficient targets are read; nothing here touches
validation, audit or test labels.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def frame_amplitudes(coefficients, norm, device=None, chunk=4096, amplitude_floor=1e-6):
    """Peak-to-peak flux amplitude of every reconstructed target frame.

    ``coefficients`` is an (N, k) array of standardized POD coefficients (a
    memory map is fine); ``norm`` holds ``basis`` (k, 4225), ``coefficient_std``
    (k,) and ``psi_mean`` (65, 65) as numpy arrays or tensors. The product is
    evaluated in float32 chunks (GPU when available), matching the precision
    of the training-time ``reconstruct`` path. Returns a float64 (N,) array
    floored at ``amplitude_floor`` so the weights stay finite.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = torch.as_tensor(np.asarray(norm["basis"]), dtype=torch.float32, device=device)
    std = torch.as_tensor(np.asarray(norm["coefficient_std"]), dtype=torch.float32, device=device)
    mean = torch.as_tensor(np.asarray(norm["psi_mean"]), dtype=torch.float32, device=device).reshape(1, -1)
    if basis.ndim != 2 or basis.shape[1] != mean.shape[1] or std.shape != basis.shape[:1]:
        raise ValueError("Basis, coefficient std and psi mean shapes are inconsistent")
    n = len(coefficients)
    amplitudes = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for first in range(0, n, chunk):
            c = torch.as_tensor(np.asarray(coefficients[first:first + chunk], dtype=np.float32), device=device)
            if c.shape[1] != len(std):
                raise ValueError("Coefficient rank differs from the POD basis")
            psi = (c * std) @ basis + mean
            amplitudes[first:first + len(c)] = (psi.amax(1) - psi.amin(1)).double().cpu().numpy()
    if not np.isfinite(amplitudes).all():
        raise ValueError("Nonfinite reconstructed flux amplitude")
    return np.maximum(amplitudes, amplitude_floor)


def amplitude_weights(amplitudes, power=1.0, max_weight=8.0, selected=None):
    """Mean-one weights ``clip((s_ref / s) ** power, 1, max_weight)``.

    ``selected`` optionally lists the frame indices actually optimized (for
    example the li-supervised subset); the reference median and the mean-one
    rescaling are then taken over those frames only, while every frame still
    receives a weight so the full array can be indexed by global frame index.
    Returns ``(weights[float32, N], metadata)``.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.ndim != 1 or not len(amplitudes):
        raise ValueError("Amplitudes must be a nonempty vector")
    if not np.isfinite(power) or power < 0 or not np.isfinite(max_weight) or max_weight < 1:
        raise ValueError("Amplitude power must be nonnegative and the weight cap at least 1")
    if selected is None:
        selected = np.arange(len(amplitudes))
    selected = np.asarray(selected)
    if selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Selected frames must be a nonempty integer index vector")
    if selected.min() < 0 or selected.max() >= len(amplitudes) or len(np.unique(selected)) != len(selected):
        raise ValueError("Selected frames must be unique valid indices")
    reference = float(np.median(amplitudes[selected]))
    raw = np.clip((reference / amplitudes) ** power, 1.0, max_weight)
    used = raw[selected]
    scale = float(used.mean())
    weights = (raw / scale).astype(np.float32)
    quantile = lambda values: {f"q{int(round(100 * q)):02d}": float(np.quantile(values, q)) for q in QUANTILES}
    effective = float(used.sum() ** 2 / np.square(used).sum())
    metadata = {
        "method": "target_flux_peak_to_peak_amplitude",
        "formula": "clip((s_ref / s_f) ** power, 1, max_weight) / mean_over_used_frames",
        "power": float(power), "max_weight": float(max_weight),
        "reference_amplitude_median": reference,
        "reference_frames": int(len(selected)), "total_frames": int(len(amplitudes)),
        "amplitude_quantiles_used_frames": quantile(amplitudes[selected]),
        "amplitude_quantiles_all_frames": quantile(amplitudes),
        "amplitude_median_all_frames": float(np.median(amplitudes)),
        "raw_weight_mean_used_frames": scale,
        "normalized_weight_quantiles_used_frames": quantile(weights[selected]),
        "normalized_weight_mean_used_frames": float(weights[selected].mean()),
        "normalized_weight_max": float(weights.max()),
        "fraction_clipped_at_max_used_frames": float(np.mean(used >= max_weight)),
        "fraction_at_unit_floor_used_frames": float(np.mean(used <= 1.0)),
        "effective_sample_fraction_used_frames": effective / len(selected),
        "normalization": "raw weights divided by their mean over the optimized frames; loss uses batch weighted means",
        "target_values_read": "training coefficient targets only", "validation_labels_read": False,
        "audit_labels_read": False, "labels_interpolated": False,
        "amplitude_sha256": hashlib.sha256(amplitudes.tobytes()).hexdigest(),
    }
    return weights, metadata
