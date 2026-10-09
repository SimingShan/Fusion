"""Machine-agnostic physics input fields for zero-shot MAST transfer (Challenge 2).

Everything here is computed from public *inputs* only: shipped conductor
rectangles, coil/plasma currents, and Thomson chord geometry. No target flux
map is read by this module (the check script reads DIII-D TRAIN labels for
validation only).

Conventions
-----------
* Green function: ``psi = R * A_phi`` in Wb/rad per ampere of a circular
  filament, ``G = mu0/(pi k) sqrt(R Rc) [(1 - k^2/2) K(k) - E(k)]`` with
  ``k^2 = 4 R Rc / ((R+Rc)^2 + (Z-Zc)^2)``. Positive current gives positive psi
  and ``B_Z = (1/R) dpsi/dR > 0`` on axis.
* Coil response tables are in Wb/rad per **kA of the shipped signal**,
  including the ``1e3`` (kA -> A) and the per-element split:
    - DIII-D ``F*``: kA-turn on one rectangle -> ``1000 A`` per rectangle.
    - DIII-D ``ECOILA``: kA feed on one lumped rectangle; the table is the
      *nominal single-turn* response (``1000 A`` on the rectangle). The
      effective turn count is a DIII-D-only calibration constant stored in
      ``data/physics_tables/diii_d_calibration.json``.
    - MAST ``P2..P6 {u,l}``: total kA-turn spread over N rectangles ->
      ``1000/N A`` each.
    - MAST ``sol``: kA feed through 656 series turns -> ``1000 A`` each
      (convention unverified against labels; see check script).
    - ``bcoil``, ``tf``, ``efps`` carry no geometry and are omitted.
* Axis-maximum external field, machine agnostic::

      phi_ext = sign(Ip) * sum_c T_c * I_c[kA] + |Ip|[kA] * T_plasma

  With ``Ip>0`` this is physical psi. The DIII-D training target in this
  convention is ``phi = -efit_psirz``; the MAST target is ``+efit_psirz``.
* Grids are uniform 65x65, arrays indexed ``[Z, R]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import sys
import time as wall_time
from typing import Any

import numpy as np
from scipy.special import ellipe, ellipk

from fusion.data import arrow_to_numpy, interpolate_signal
from starter.data_fixes import align_mast_thomson_core, fix_d3d_ip_times

MU0 = 4.0e-7 * np.pi
E_CHARGE = 1.602176634e-19
ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "starter" / "fusion_scoring"
TABLE_DIR = ROOT / "data" / "physics_tables"
ENVELOPE_FILES = {"DIII-D": SCORER / "masks" / "d3d_envelope.npz",
                  "MAST": SCORER / "masks" / "mast_envelope.npz"}
MACHINE_TAGS = {"DIII-D": "diii_d", "MAST": "mast"}
D3D_IP_KEY = "magnetics_plasma_current"

INPUT_COLUMNS = {
    "source", "efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time",
    "magnetics_plasma_current", "magnetics_plasma_current_times",
    "coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
    "coil_angle1", "coil_angle2",
    "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    "thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
    "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial",
}
LABEL_COLUMNS = {"efit_psirz"}


# --------------------------------------------------------------------------- #
# Green function
# --------------------------------------------------------------------------- #
def green_function(R, Z, Rc, Zc, min_separation: float = 1.0e-3) -> np.ndarray:
    """Flux ``psi = R A_phi`` [Wb/rad per A] of a circular filament at ``(Rc, Zc)``.

    Vectorized with broadcasting. Field points closer than ``min_separation``
    [m] to a filament are evaluated at that separation (a finite wire radius),
    which keeps the logarithmic self-field finite; exact coincidence is
    therefore allowed. ``R = 0`` gives 0 exactly via the small-``m`` series.
    """
    R, Z, Rc, Zc = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (R, Z, Rc, Zc)))
    if np.any(R < 0) or np.any(Rc <= 0):
        raise ValueError("Nonnegative field radius and positive source radius required")
    distance_sq = np.maximum((R - Rc) ** 2 + (Z - Zc) ** 2, min_separation ** 2)
    m = 4.0 * R * Rc / (4.0 * R * Rc + distance_sq)  # k^2 in (0, 1)
    out = np.zeros(m.shape, dtype=np.float64)
    small = m < 1.0e-5
    if small.any():
        ms = m[small]
        # Series of (1 - m/2) K - E in m avoids cancellation of the two terms.
        out[small] = MU0 * np.sqrt(R[small] * Rc[small]) * ms ** 1.5 / 32.0 * (
            1.0 + 0.75 * ms + 75.0 / 128.0 * ms ** 2)
    big = ~small
    if big.any():
        mb = m[big]
        k = np.sqrt(mb)
        out[big] = MU0 / (np.pi * k) * np.sqrt(R[big] * Rc[big]) * (
            (1.0 - 0.5 * mb) * ellipk(mb) - ellipe(mb))
    return out


def biot_savart_psi(R: float, Z: float, Rc: float, Zc: float, n: int = 200001) -> float:
    """Independent check: ``psi = R A_phi`` by direct azimuthal integration (per ampere)."""
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    denominator = np.sqrt(R ** 2 + Rc ** 2 - 2.0 * R * Rc * np.cos(phi) + (Z - Zc) ** 2)
    a_phi = MU0 * Rc / (4.0 * np.pi) * np.trapz(np.cos(phi) / denominator, phi)
    return float(R * a_phi)


# --------------------------------------------------------------------------- #
# Conductor geometry
# --------------------------------------------------------------------------- #
def rectangle_filaments(Rc, Zc, width, height, angle1=0.0, angle2=0.0, n=4):
    """Split one shipped conductor cross-section into ``n_r x n_z`` equal-current filaments.

    ``n`` is an int (``n x n``) or a ``(n_r, n_z)`` pair. ``angle1`` (EFIT
    ``AF``, degrees) rotates the whole rectangle about its centre. ``angle2``
    (EFIT ``AF2``, degrees) makes a parallelogram whose horizontal sides have
    length ``width`` and whose slanted sides make angle ``angle2`` with the R
    axis (90 deg = upright rectangle). Returns ``(R_f, Z_f)`` arrays of shape
    ``(n_r*n_z,)``; each filament carries an equal share of the element current.
    """
    n_r, n_z = (int(n), int(n)) if np.isscalar(n) else (int(n[0]), int(n[1]))
    ur = (np.arange(n_r) + 0.5) / n_r - 0.5  # cell centres in [-1/2, 1/2)
    uz = (np.arange(n_z) + 0.5) / n_z - 0.5
    du, dv = np.meshgrid(ur, uz, indexing="ij")
    x = du.ravel() * float(width)
    y = dv.ravel() * float(height)
    if angle2 != 0.0 and abs(float(angle2) % 180.0 - 90.0) > 1.0e-9:
        x = x + y / np.tan(np.deg2rad(float(angle2)))
    if angle1 != 0.0:
        c, s = np.cos(np.deg2rad(float(angle1))), np.sin(np.deg2rad(float(angle1)))
        x, y = c * x - s * y, s * x + c * y
    return float(Rc) + x, float(Zc) + y


def element_current_per_kA(machine: str, key: str, count: int) -> float:
    """Amperes flowing in each shipped rectangle per kA of the shipped signal."""
    if machine == "MAST":
        if key == "magnetics_sol_current":
            return 1000.0  # kA feed, 656 series turns each carrying the feed current
        if re.fullmatch(r"magnetics_p[2-6][lu]_current", key):
            return 1000.0 / count  # total kA-turn spread over the supplied rectangles
    elif machine == "DIII-D":
        if re.fullmatch(r"magnetics_F[1-9][AB]", key):
            return 1000.0  # kA-turn on one rectangle
        if key == "magnetics_ECOILA":
            return 1000.0  # nominal single-turn response; effective turns are a calibration
    raise ValueError(f"Unknown conductor current convention: {machine} {key}")


def filament_counts(width, height, grid_R, grid_Z, n: int = 4, adaptive: bool = True,
                    n_max: int = 256) -> tuple[int, int]:
    """Per-dimension filament counts for one element.

    ``adaptive`` targets a filament spacing of half the finest grid spacing, so
    small conductors (MAST solenoid turns) get 1-2 filaments while tall ones
    (DIII-D ECOILA, 3.5 m) get hundreds; otherwise a fixed ``n x n``.
    """
    if not adaptive:
        return int(n), int(n)
    spacing = 0.5 * min(float(np.min(np.diff(grid_R))), float(np.min(np.diff(grid_Z))))
    n_r = int(np.clip(np.ceil(float(width) / spacing), 1, n_max))
    n_z = int(np.clip(np.ceil(float(height) / spacing), 1, n_max))
    return n_r, n_z


def geometry_hash(row: dict, n: int, adaptive: bool = True) -> str:
    geometry = np.stack([np.asarray(row[k], dtype=np.float64) for k in
                         ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")])
    grid = np.concatenate([np.asarray(row["efit_grid_R"], np.float64), np.asarray(row["efit_grid_Z"], np.float64)])
    join = "|".join(str(k) for k in row["coil_input_column"])
    payload = geometry.tobytes() + grid.tobytes() + join.encode() + f"|n={n}|adaptive={adaptive}|v2".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _grid_mesh(grid_R, grid_Z):
    RR, ZZ = np.meshgrid(np.asarray(grid_R, np.float64), np.asarray(grid_Z, np.float64))
    return RR, ZZ


def filament_field(RR, ZZ, R_f, Z_f, weights=None, chunk: int = 512) -> np.ndarray:
    """Sum of filament Green functions over a grid, chunked to bound memory. Per ampere total."""
    R_f = np.asarray(R_f, np.float64).ravel()
    Z_f = np.asarray(Z_f, np.float64).ravel()
    weights = np.ones(len(R_f)) / len(R_f) if weights is None else np.asarray(weights, np.float64).ravel()
    out = np.zeros(RR.shape, dtype=np.float64)
    flatR, flatZ = RR.ravel()[:, None], ZZ.ravel()[:, None]
    for start in range(0, len(R_f), chunk):
        stop = min(start + chunk, len(R_f))
        G = green_function(flatR, flatZ, R_f[None, start:stop], Z_f[None, start:stop])
        out += (G @ weights[start:stop]).reshape(RR.shape)
    return out


def coil_response_table(shot_row: dict, n: int = 4, cache_dir: Path | None = TABLE_DIR,
                        verbose: bool = False, adaptive: bool = True) -> dict[str, np.ndarray]:
    """Map each geometric current signal to a ``(nz, nr)`` response in Wb/rad per kA of signal.

    Rectangles (or parallelograms, honoring ``coil_angle1/2``) are split into
    ``n x n`` filaments, or with ``adaptive`` (default) into as many filaments
    per dimension as needed for a spacing of half the grid step (see
    ``filament_counts``). Results are cached to ``<cache_dir>/<machine>_<hash>.npz``.
    Signals without geometry (bcoil, tf, efps) are absent from the result.
    """
    machine = str(shot_row["source"])
    tag = MACHINE_TAGS[machine]
    digest = geometry_hash(shot_row, n, adaptive)
    cache_path = None if cache_dir is None else Path(cache_dir) / f"{tag}_{digest}.npz"
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return {str(k): data[k] for k in data.files if k != "__meta__"}
    started = wall_time.perf_counter()
    RR, ZZ = _grid_mesh(shot_row["efit_grid_R"], shot_row["efit_grid_Z"])
    join = np.asarray([str(k) for k in shot_row["coil_input_column"]])
    geometry = np.stack([np.asarray(shot_row[k], dtype=np.float64) for k in
                         ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")])
    tables: dict[str, np.ndarray] = {}
    for key in sorted(set(join.tolist())):
        members = np.flatnonzero(join == key)
        per_element = element_current_per_kA(machine, key, len(members))
        R_f, Z_f, W_f = [], [], []
        for i in members:
            counts = filament_counts(geometry[2, i], geometry[3, i], shot_row["efit_grid_R"],
                                     shot_row["efit_grid_Z"], n=n, adaptive=adaptive)
            r, z = rectangle_filaments(*geometry[:, i], n=counts)
            R_f.append(r)
            Z_f.append(z)
            # Each rectangle carries per_element amperes split equally over its filaments.
            W_f.append(np.full(len(r), per_element / len(r)))
        R_f, Z_f, W_f = np.concatenate(R_f), np.concatenate(Z_f), np.concatenate(W_f)
        tables[key] = filament_field(RR, ZZ, R_f, Z_f, W_f)
        if verbose:
            print(f"  {key}: {len(members)} elements, {len(R_f)} filaments, "
                  f"{per_element:.3f} A/element/kA, mean {tables[key].mean():.3e} Wb/rad/kA")
    elapsed = wall_time.perf_counter() - started
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"machine": machine, "n": n, "adaptive": adaptive, "hash": digest, "build_seconds": elapsed,
                           "unit": "Wb/rad per kA of shipped signal"})
        np.savez(cache_path, __meta__=np.array(meta), **tables)
    if verbose:
        print(f"  built {machine} tables in {elapsed:.2f} s")
    return tables


# --------------------------------------------------------------------------- #
# Plasma source tables
# --------------------------------------------------------------------------- #
def plasma_filament_table(grid, Rc: float, Zc: float) -> np.ndarray:
    """Response [Wb/rad per kA] of a single toroidal filament at ``(Rc, Zc)``."""
    RR, ZZ = _grid_mesh(*grid)
    return 1000.0 * green_function(RR, ZZ, Rc, Zc)


def plasma_blob_table(grid, Rc: float, Zc: float, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Response [Wb/rad per kA] of an isotropic Gaussian current blob carrying unit current.

    The blob is discretized on the grid cells within ``truncate*sigma``; each
    cell is integrated with 2x2 Gauss points so that no source ever coincides
    with a grid node.
    """
    grid_R, grid_Z = (np.asarray(g, np.float64) for g in grid)
    RR, ZZ = _grid_mesh(grid_R, grid_Z)
    dR, dZ = float(np.mean(np.diff(grid_R))), float(np.mean(np.diff(grid_Z)))
    weight = np.exp(-0.5 * ((RR - Rc) ** 2 + (ZZ - Zc) ** 2) / sigma ** 2)
    weight[(RR - Rc) ** 2 + (ZZ - Zc) ** 2 > (truncate * sigma) ** 2] = 0.0
    weight[RR <= 0] = 0.0
    weight /= weight.sum()
    active = weight > 0
    Rs, Zs, Ws = RR[active], ZZ[active], weight[active]
    R_f, Z_f, W_f = [], [], []
    for u in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
        for v in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
            R_f.append(Rs + u * dR / 2)
            Z_f.append(Zs + v * dZ / 2)
            W_f.append(Ws / 4.0)
    return 1000.0 * filament_field(RR, ZZ, np.concatenate(R_f), np.concatenate(Z_f), np.concatenate(W_f))


# --------------------------------------------------------------------------- #
# Field assembly
# --------------------------------------------------------------------------- #
@dataclass
class FieldEvaluator:
    """Stacked coil tables for sub-millisecond per-frame evaluation."""
    names: list[str]
    matrix: np.ndarray  # (C, nz*nr)
    shape: tuple[int, int]

    @classmethod
    def from_tables(cls, tables: dict[str, np.ndarray], names: list[str] | None = None):
        names = sorted(tables) if names is None else list(names)
        shape = tables[names[0]].shape
        return cls(names, np.stack([tables[k].ravel() for k in names]), shape)

    def coil_field(self, currents_kA: np.ndarray) -> np.ndarray:
        """``currents_kA`` shape ``(C,)`` or ``(T, C)`` -> ``(nz, nr)`` or ``(T, nz, nr)``; unsigned sum."""
        currents_kA = np.asarray(currents_kA, np.float64)
        out = currents_kA @ self.matrix
        return out.reshape(currents_kA.shape[:-1] + self.shape)

    def currents_from(self, currents: dict[str, Any]) -> np.ndarray:
        columns = [np.asarray(currents[k], np.float64) for k in self.names]
        return np.stack(np.broadcast_arrays(*columns), axis=-1)


def shot_ip_sign(Ip_kA, mask=None) -> float:
    """Shot-level plasma current orientation: sign of the |Ip|-weighted mean.

    Ip keeps one sign within a shot, so this is robust to frames where the
    interpolated Ip is near zero or noisy (e.g. residual time-axis errors).
    """
    Ip = np.asarray(Ip_kA, np.float64).ravel()
    if mask is not None:
        Ip = Ip[np.asarray(mask, bool).ravel()]
    weighted = float(np.sum(Ip * np.abs(Ip))) if len(Ip) else 0.0
    return 1.0 if weighted >= 0 else -1.0


def _orientation(Ip: np.ndarray, ip_sign) -> np.ndarray:
    if ip_sign is None:
        return np.where(Ip >= 0, 1.0, -1.0)
    return np.broadcast_to(np.asarray(ip_sign, np.float64), Ip.shape).copy()


def external_field(tables: dict[str, np.ndarray], currents_kA: dict[str, Any], Ip_kA,
                   plasma_table: np.ndarray, coil_multipliers: dict[str, float] | None = None,
                   ip_sign: float | None = None) -> np.ndarray:
    """Axis-maximum external field ``phi_ext = sign(Ip) sum_c T_c I_c + |Ip| T_plasma``.

    ``currents_kA`` maps signal names to scalars or ``(T,)`` arrays (signals not
    in ``tables`` are ignored); ``Ip_kA`` is a scalar or ``(T,)``; ``plasma_table``
    is ``(nz, nr)`` or ``(T, nz, nr)``. ``coil_multipliers`` applies calibration
    factors (e.g. ECOILA effective turns) to named tables. ``ip_sign`` fixes the
    orientation (use ``shot_ip_sign``) instead of the per-frame sign of ``Ip``.
    Returns ``(nz, nr)`` for scalar inputs, else ``(T, nz, nr)``.
    """
    names = [k for k in sorted(tables) if k in currents_kA]
    if not names:
        raise ValueError("No current signal matches a response table")
    scaled = {k: tables[k] * (coil_multipliers or {}).get(k, 1.0) for k in names}
    evaluator = FieldEvaluator.from_tables(scaled, names)
    currents = evaluator.currents_from({k: currents_kA[k] for k in names})
    Ip = np.asarray(Ip_kA, np.float64)
    coil = evaluator.coil_field(currents)
    sign = _orientation(Ip, ip_sign)
    plasma_table = np.asarray(plasma_table, np.float64)
    if coil.ndim == 2:
        return float(sign) * coil + abs(float(Ip)) * plasma_table
    return sign[:, None, None] * coil + np.abs(Ip)[:, None, None] * plasma_table


def decompose_external_field(tables, currents_kA, Ip_kA, plasma_table, groups: dict[str, list[str]],
                             coil_multipliers=None, ip_sign: float | None = None) -> dict[str, np.ndarray]:
    """Signed per-group coil contributions plus the plasma term, same convention as ``external_field``."""
    Ip = np.asarray(Ip_kA, np.float64)
    sign = _orientation(Ip, ip_sign)
    out = {}
    for group, names in groups.items():
        present = [k for k in names if k in tables and k in currents_kA]
        if not present:
            continue
        scaled = {k: tables[k] * (coil_multipliers or {}).get(k, 1.0) for k in present}
        ev = FieldEvaluator.from_tables(scaled, present)
        coil = ev.coil_field(ev.currents_from({k: currents_kA[k] for k in present}))
        out[group] = coil * (sign if coil.ndim == 2 else sign[:, None, None])
    out["plasma"] = np.abs(Ip)[..., None, None] * np.asarray(plasma_table) if Ip.ndim else abs(float(Ip)) * plasma_table
    return out


# --------------------------------------------------------------------------- #
# Thomson splat
# --------------------------------------------------------------------------- #
def thomson_splat(grid, chord_R, chord_Z, values, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear splat of per-chord values (e.g. p_e in Pa) onto the grid.

    Returns ``(splat, hits)``: ``splat`` accumulates ``value * bilinear weight``
    and ``hits`` the bilinear weights themselves, so ``splat / hits`` is a
    weight-normalized local average where ``hits > 0``. Chords outside the
    grid or with ``mask == False`` are ignored.
    """
    grid_R, grid_Z = (np.asarray(g, np.float64) for g in grid)
    chord_R, chord_Z, values = (np.asarray(x, np.float64).ravel() for x in (chord_R, chord_Z, values))
    mask = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool).ravel()
    keep = mask & np.isfinite(values) & (chord_R >= grid_R[0]) & (chord_R <= grid_R[-1]) \
        & (chord_Z >= grid_Z[0]) & (chord_Z <= grid_Z[-1])
    splat = np.zeros((len(grid_Z), len(grid_R)))
    hits = np.zeros_like(splat)
    if not keep.any():
        return splat, hits
    dR, dZ = float(np.mean(np.diff(grid_R))), float(np.mean(np.diff(grid_Z)))
    fr = (chord_R[keep] - grid_R[0]) / dR
    fz = (chord_Z[keep] - grid_Z[0]) / dZ
    ir = np.clip(np.floor(fr).astype(int), 0, len(grid_R) - 2)
    iz = np.clip(np.floor(fz).astype(int), 0, len(grid_Z) - 2)
    tr, tz = fr - ir, fz - iz
    v = values[keep]
    for dz, dr, w in ((0, 0, (1 - tz) * (1 - tr)), (0, 1, (1 - tz) * tr), (1, 0, tz * (1 - tr)), (1, 1, tz * tr)):
        np.add.at(splat, (iz + dz, ir + dr), w * v)
        np.add.at(hits, (iz + dz, ir + dr), w)
    return splat, hits


# --------------------------------------------------------------------------- #
# Shot loading
# --------------------------------------------------------------------------- #
def read_shot_row(path: Path, with_labels: bool = False) -> dict:
    """Read the one-row parquet with an explicit input allowlist (labels only when asked)."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    allowed = set(INPUT_COLUMNS) | {c for c in names if re.fullmatch(r"magnetics_(F[1-9][AB]|ECOILA|p[2-6][lu]_current|sol_current)", c)}
    if with_labels:
        allowed |= LABEL_COLUMNS
    columns = sorted(allowed.intersection(names))
    table = pq.read_table(path, columns=columns)
    if len(table) != 1:
        raise ValueError("Expected exactly one shot per parquet")
    row = {name: arrow_to_numpy(table[name][0]) for name in columns}
    if with_labels and row.get("source") == "MAST" and "efit_psirz" in row:
        raise ValueError("MAST labels must never be read by this library")
    return row


def geometric_current_names(row: dict) -> list[str]:
    return sorted(set(str(k) for k in row["coil_input_column"]))


def _thomson_chord_channels(row: dict, system: str, channels: int):
    """Chord (R, Z) for each data channel of one Thomson system, via the chord name index."""
    group = "core" if system == "core" else ("tangential" if row["source"] == "DIII-D" else "edge")
    coords = np.full((channels, 2), np.nan)
    for name, r, z in zip(row["thomson_chord_name"], row["thomson_chord_R"], row["thomson_chord_Z"]):
        match = re.fullmatch(rf"TS_{group}_(?:.*_)?(\d+)", str(name))
        if match and int(match.group(1)) < channels:
            coords[int(match.group(1))] = (r, z)
    return coords


def load_shot_inputs(path: Path | str, frames: np.ndarray | None = None, with_labels: bool = False,
                     thomson: bool = True) -> dict:
    """Load one shot's inputs at its ``efit_times`` for either machine.

    Returns a dict with ``machine, path, efit_times (T,), frames, grid_R, grid_Z,
    currents {name: (T,) kA}, current_masks {name: (T,) bool}, Ip_kA (T,),
    Ip_mask (T,)``, and when ``thomson`` is set ``chord_R, chord_Z (K,), pe (T, K) Pa,
    pe_mask (T, K)`` (core then edge channels concatenated). With
    ``with_labels`` on DIII-D, ``phi_target (T, nz, nr) = -efit_psirz`` (axis-max).

    Currents are linearly interpolated after dropping non-finite samples per
    signal; the mask is False outside a signal's observed interval or when the
    signal is entirely missing. DIII-D Ip uses the v1.1.0 time-axis fix; the
    MAST Thomson channel-0 drop is applied.
    """
    path = Path(path)
    row = read_shot_row(path, with_labels=with_labels)
    machine = str(row["source"])
    times = np.asarray(row["efit_times"], np.float64).ravel()
    frames = np.arange(len(times)) if frames is None else np.asarray(frames, int)
    query = times[frames]
    out = {"machine": machine, "path": str(path), "efit_times": query, "frames": frames,
           "grid_R": np.asarray(row["efit_grid_R"], np.float64),
           "grid_Z": np.asarray(row["efit_grid_Z"], np.float64), "row": row}
    shared_time = row["magnetics_time"]
    currents, masks = {}, {}
    for key in geometric_current_names(row):
        value, mask = interpolate_signal(shared_time, row.get(key), query)
        currents[key] = value.astype(np.float64)
        masks[key] = mask.astype(bool)
    ip_times = shared_time
    if machine == "DIII-D":
        ip_times = fix_d3d_ip_times(row)
    ip, ip_mask = interpolate_signal(ip_times, row[D3D_IP_KEY], query)
    out.update(currents=currents, current_masks=masks, Ip_kA=ip.astype(np.float64), Ip_mask=ip_mask.astype(bool))
    # Shot-level orientation from the whole Ip trace (not only the queried frames).
    trace = np.asarray(row[D3D_IP_KEY], np.float64)
    out["Ip_sign"] = shot_ip_sign(trace[np.isfinite(trace)])
    if thomson:
        chord_R, chord_Z, pe, pe_mask = [], [], [], []
        for system in ("core", "edge"):
            Te = np.asarray(row[f"thomson_{system}_Te"], np.float64)
            ne = np.asarray(row[f"thomson_{system}_ne"], np.float64)
            ts = np.asarray(row[f"thomson_{system}_times"], np.float64)
            if Te.ndim == 2 and Te.shape[0] != len(ts) and Te.shape[1] == len(ts):
                Te, ne = Te.T, ne.T
            if machine == "MAST" and system == "core":
                _, Te, ne = align_mast_thomson_core(row)
            channels = Te.shape[1]
            coords = _thomson_chord_channels(row, system, channels)
            te_i = np.zeros((len(query), channels))
            ne_i = np.zeros_like(te_i)
            ok = np.zeros(te_i.shape, bool)
            for j in range(channels):
                te, m1 = interpolate_signal(ts, np.where(Te[:, j] > 0, Te[:, j], np.nan), query)
                de, m2 = interpolate_signal(ts, np.where(ne[:, j] > 0, ne[:, j], np.nan), query)
                te_i[:, j], ne_i[:, j] = te, de
                ok[:, j] = m1.astype(bool) & m2.astype(bool) & (te > 0) & (de > 0) & np.isfinite(coords[j]).all()
            chord_R.append(coords[:, 0])
            chord_Z.append(coords[:, 1])
            pe.append(E_CHARGE * np.maximum(te_i, 0) * np.maximum(ne_i, 0))
            pe_mask.append(ok)
        out.update(chord_R=np.concatenate(chord_R), chord_Z=np.concatenate(chord_Z),
                   pe=np.concatenate(pe, axis=1), pe_mask=np.concatenate(pe_mask, axis=1))
    if with_labels and "efit_psirz" in row:
        psi = np.asarray(row["efit_psirz"], np.float64)[frames]
        out["phi_target"] = -psi if machine == "DIII-D" else psi
    return out


# --------------------------------------------------------------------------- #
# Machine helpers
# --------------------------------------------------------------------------- #
def load_envelope(machine: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scorer public envelope ``(grid_R, grid_Z, mask_coarse)`` for a machine (no label content)."""
    with np.load(ENVELOPE_FILES[machine]) as data:
        return data["grid_R"].astype(np.float64), data["grid_Z"].astype(np.float64), data["mask_coarse"].astype(bool)


def envelope_centroid(machine: str) -> tuple[float, float]:
    """Machine-fixed plasma centre guess: centroid of the scorer envelope mask."""
    grid_R, grid_Z, mask = load_envelope(machine)
    RR, ZZ = _grid_mesh(grid_R, grid_Z)
    return float(RR[mask].mean()), float(ZZ[mask].mean())


def load_calibration(path: Path = TABLE_DIR / "diii_d_calibration.json") -> dict:
    """DIII-D-only calibration constants (label-fitted on TRAIN shots); empty if absent."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def coil_groups(machine: str, names: list[str]) -> dict[str, list[str]]:
    """Named decomposition groups for reporting: shaping coils vs central solenoid."""
    if machine == "MAST":
        return {"pcoils": [k for k in names if re.fullmatch(r"magnetics_p[2-6][lu]_current", k)],
                "solenoid": [k for k in names if k == "magnetics_sol_current"]}
    return {"fcoils": [k for k in names if re.fullmatch(r"magnetics_F[1-9][AB]", k)],
            "solenoid": [k for k in names if k == "magnetics_ECOILA"]}


def midplane_vertical_field(phi: np.ndarray, grid_R, grid_Z) -> dict:
    """``B_Z = (1/R) dphi/dR`` on the ``Z=0`` row; reports null radii and decay index ``n = -(R/B_Z) dB_Z/dR``."""
    grid_R, grid_Z = (np.asarray(g, np.float64) for g in (grid_R, grid_Z))
    iz = int(np.argmin(np.abs(grid_Z)))
    row = np.asarray(phi, np.float64)[iz]
    bz = np.gradient(row, grid_R) / grid_R
    dbz = np.gradient(bz, grid_R)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = -grid_R * dbz / bz
    crossings = np.flatnonzero(np.sign(bz[:-1]) * np.sign(bz[1:]) < 0)
    nulls = [float(grid_R[i] - bz[i] * (grid_R[i + 1] - grid_R[i]) / (bz[i + 1] - bz[i])) for i in crossings]
    return {"Z_row": float(grid_Z[iz]), "B_Z": bz, "decay_index": index, "null_R": nulls}


def frame_stats(phi: np.ndarray, grid_R, grid_Z, centre: tuple[float, float]) -> dict:
    """Scalar summary of one field frame: grid mean/min/max, perimeter mean, value at ``centre``."""
    phi = np.asarray(phi, np.float64)
    grid_R, grid_Z = (np.asarray(g, np.float64) for g in (grid_R, grid_Z))
    perimeter = np.concatenate([phi[0], phi[-1], phi[1:-1, 0], phi[1:-1, -1]])
    ir = int(np.argmin(np.abs(grid_R - centre[0])))
    iz = int(np.argmin(np.abs(grid_Z - centre[1])))
    return {"mean": float(phi.mean()), "min": float(phi.min()), "max": float(phi.max()),
            "boundary_mean": float(perimeter.mean()), "centre_value": float(phi[iz, ir]),
            "peak_to_peak": float(phi.max() - phi.min())}


def _rows_to_dict(names, values):
    return {str(k): v for k, v in zip(names, values)}


@dataclass
class ShotFields:
    """Per-frame physics fields for one shot (convenience wrapper over the table API)."""
    machine: str
    efit_times: np.ndarray
    phi_ext: np.ndarray          # (T, nz, nr) axis-max external field
    phi_coils: np.ndarray        # (T, nz, nr) signed coil part
    phi_plasma: np.ndarray       # (T, nz, nr) plasma filament part
    pe_splat: np.ndarray         # (T, nz, nr) Thomson pressure splat
    pe_hits: np.ndarray          # (T, nz, nr) splat weights
    Ip_kA: np.ndarray
    meta: dict = field(default_factory=dict)


def shot_fields(path: Path | str, frames=None, n: int = 4, centre: tuple[float, float] | None = None,
                sigma: float | None = None, calibration: dict | None = None,
                tables: dict[str, np.ndarray] | None = None, plasma_table: np.ndarray | None = None) -> ShotFields:
    """One-call builder: load inputs, evaluate ``phi_ext`` (filament or blob plasma) and the p_e splat.

    Pass ``tables`` and ``plasma_table`` (from ``plasma_blob_table``, ~2-7 s to
    build) when processing many shots of one machine; only the coil tables
    are disk-cached.
    """
    inputs = load_shot_inputs(path, frames=frames)
    row = inputs["row"]
    machine = inputs["machine"]
    if tables is None:
        tables = coil_response_table(row, n=n)
    grid = (inputs["grid_R"], inputs["grid_Z"])
    centre = envelope_centroid(machine) if centre is None else centre
    if plasma_table is not None:
        plasma = np.asarray(plasma_table, np.float64)
    else:
        plasma = plasma_filament_table(grid, *centre) if sigma is None else plasma_blob_table(grid, *centre, sigma)
    multipliers = {}
    if machine == "DIII-D":
        calibration = load_calibration() if calibration is None else calibration
        if "ecoila_effective_turns" in calibration:
            multipliers["magnetics_ECOILA"] = float(calibration["ecoila_effective_turns"])
    Ip = inputs["Ip_kA"]
    sign = inputs["Ip_sign"]
    phi = external_field(tables, inputs["currents"], Ip, plasma, coil_multipliers=multipliers, ip_sign=sign)
    phi_plasma = np.abs(Ip)[:, None, None] * plasma
    splat = np.zeros_like(phi)
    hits = np.zeros_like(phi)
    for t in range(len(Ip)):
        splat[t], hits[t] = thomson_splat(grid, inputs["chord_R"], inputs["chord_Z"], inputs["pe"][t], inputs["pe_mask"][t])
    return ShotFields(machine, inputs["efit_times"], phi, phi - phi_plasma, phi_plasma, splat, hits, Ip,
                      {"centre": centre, "sigma": sigma, "multipliers": multipliers, "sign": sign})
