"""Training-only differentiable replay of official marching-square LCFS geometry.

No production/scorer modifications. Trace discrete choices using the official
scorer, identify the active termination level (grid vertex or mask-edge midpoint),
then replay edge interpolation and arc-length resampling in PyTorch. Topology is
frozen at the trace field; it must be refreshed when the prediction moves.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
from lcfs import _best_contour_at
from o_point import find_o_point
from contour import grid_to_physical, resample_closed


@dataclass
class Trace:
    edge_a: np.ndarray
    edge_b: np.ndarray
    xy_a: np.ndarray
    xy_b: np.ndarray
    critical_indices: np.ndarray
    critical_weights: np.ndarray
    level_offset: float
    level: float
    gap_fraction: float
    event_kind: str
    official_contour: np.ndarray


def trace(psi, R, Z, mask, n_iter=22):
    """Reproduce the scorer's bisection, then recover its active event exactly."""
    phi = -np.asarray(psi, dtype=np.float64)
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mask)
    lo = float(np.concatenate([phi[0], phi[-1], phi[:, 0], phi[:, -1]]).min())
    hi = float(phi[iz, ir]) - 1e-9
    span = hi - lo
    best = _best_contour_at(phi, hi, iz, ir, mask.astype(float))
    if span <= 0 or best is None:
        raise ValueError("No initial closed contour")
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        c = _best_contour_at(phi, mid, iz, ir, mask.astype(float))
        if c is None:
            lo = mid
        else:
            hi, best = mid, c

    # Keep the scorer's actual 22-step output. Its bracket is too wide to identify
    # the responsible event reliably: an unrelated vertex can have a nearby flux.
    # Refine only the event identification, leaving forward parity untouched.
    official_hi, official_best = hi, best
    for _ in range(max(40 - n_iter, 0)):
        mid = (lo + hi) / 2
        c = _best_contour_at(phi, mid, iz, ir, mask.astype(float))
        if c is None:
            lo = mid
        else:
            hi = mid

    # Marching squares changes connectivity at vertex levels. An interpolated
    # contour vertex exits the binary envelope at an edge's 0.5 crossing.
    flat = np.arange(phi.size).reshape(phi.shape)
    a = [flat.ravel()]
    b = [flat.ravel()]
    for aa, bb, mm, nn in [(flat[:, :-1], flat[:, 1:], mask[:, :-1], mask[:, 1:]),
                           (flat[:-1], flat[1:], mask[:-1], mask[1:])]:
        crossing = mm != nn
        a.append(aa[crossing]); b.append(bb[crossing])
    a, b = np.concatenate(a), np.concatenate(b)
    events = (phi.ravel()[a] + phi.ravel()[b]) / 2
    j = int(np.argmin(abs(events - (hi + lo) / 2)))
    gap = max(lo - events[j], events[j] - hi, 0.0)
    if gap > max(span * 2**-max(n_iter, 40), 1e-13):
        raise ValueError(f"Unidentified critical event, relative gap {gap / span}")

    # Official raw contours are closed with the last vertex repeating the first.
    raw = official_best[:-1]
    rr, cc = raw.T
    horizontal = abs(rr - np.rint(rr)) < 1e-12
    ar = np.where(horizontal, np.rint(rr), np.floor(rr)).astype(int)
    ac = np.where(horizontal, np.floor(cc), np.rint(cc)).astype(int)
    br = ar + (~horizontal)
    bc = ac + horizontal
    ea, eb = flat[ar, ac], flat[br, bc]
    xy_a = np.column_stack([R[ac], Z[ar]])
    xy_b = np.column_stack([R[bc], Z[br]])
    return Trace(ea, eb, xy_a, xy_b, np.array([a[j], b[j]]),
                 np.array([0.5, 0.5]), float(official_hi - events[j]), official_hi,
                 float(gap / span), "vertex" if a[j] == b[j] else "mask_edge",
                 resample_closed(grid_to_physical(official_best, R, Z), 512))


def replay(psi, tr, clip=False):
    """All geometry stays differentiable; only edge/searchsorted choices detach."""
    phi = -psi.flatten()
    idx = lambda a: torch.as_tensor(a, dtype=torch.long, device=psi.device)
    tensor = lambda a: torch.as_tensor(a, dtype=psi.dtype, device=psi.device)
    level = (phi[idx(tr.critical_indices)] * tensor(tr.critical_weights)).sum() + tr.level_offset
    pa, pb = phi[idx(tr.edge_a)], phi[idx(tr.edge_b)]
    frac = (level - pa) / (pb - pa)
    if clip:
        frac = frac.clamp(0, 1)
    raw = tensor(tr.xy_a) + frac[:, None] * tensor(tr.xy_b - tr.xy_a)
    closed = torch.cat([raw, raw[:1]], dim=0)
    seg = torch.linalg.vector_norm(closed[1:] - closed[:-1], dim=1)
    cumulative = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    targets = torch.arange(512, device=psi.device, dtype=psi.dtype) / 512 * cumulative[-1]
    interval = torch.searchsorted(cumulative.detach(), targets.detach(), right=True).sub(1)
    interval = interval.clamp(0, len(seg) - 1)
    t = (targets - cumulative[interval]) / seg[interval].clamp_min(1e-30)
    contour = closed[interval] + t[:, None] * (closed[interval + 1] - closed[interval])
    return contour


def shape(contour):
    R, Z = contour.T
    Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
    width = Rmax - Rmin
    kappa = (Zmax - Zmin) / width
    top, bot = Z.argmax(), Z.argmin()
    tri_top = (Rmax + Rmin - 2 * R[top]) / width
    tri_bot = (Rmax + Rmin - 2 * R[bot]) / width
    cross = R * Z.roll(-1) - R.roll(-1) * Z
    volume = 2 * torch.pi / 6 * ((R + R.roll(-1)) * cross).sum().abs()
    return torch.stack([kappa, tri_top, tri_bot, volume, Zmax, Zmin])


def load_training(n, seed=124):
    norm = np.load(ROOT / "data/pod_history128_robust_v4/normalization.npz")
    coeff = np.load(ROOT / "data/pod_history128_robust_v4/train_coefficients.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(coeff), min(n, len(coeff)), replace=False))
    c = np.asarray(coeff[rows], dtype=np.float64)
    decoder = norm["coefficient_std"].astype(np.float64)[:, None] * norm["basis"].astype(np.float64)
    mean = norm["psi_mean"].astype(np.float64)
    mask = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz")
    return rows, c, decoder, mean, mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), rng
