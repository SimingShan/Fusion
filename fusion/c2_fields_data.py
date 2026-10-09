"""Machine-agnostic per-frame input channels for the Challenge-2 flux CNN.

Every channel is dimensionless and built from public *inputs* only on each
machine's own 65x65 grid (rows = Z, cols = R):

    0 phi_ext / F        axis-maximum external field (coils + Gaussian plasma blob)
    1 phi_coils / F      signed coil part                     F = mu0 |Ip| L / 2
    2 phi_plasma / F     |Ip| * blob table (sigma 0.25 m at the envelope centroid)
    3 log1p(pe_avg/P0)  weight-normalised Thomson p_e splat   P0 = mu0 Ip^2 / (8 pi^2 L^2)
    4 hits               bilinear splat weights (chord density; 0 where no chord)
    5 envelope           scorer public envelope mask
    6 u_R                grid-normalised column coordinate in [-1, 1] (each machine's own grid)
    7 u_Z                grid-normalised row coordinate in [-1, 1]
    8 dR / L, 9 dZ / L   constant cell sizes (physical scale)

plus two conditioning vectors built from the same named per-frame entries
(``ENTRY_NAMES``): ``FIELD_COND_NAMES`` feeds the FiLM layers of the U-Net
and contains only quantities whose MAST distribution lies inside the DIII-D
training range (verified by scripts/challenge2/check_c2_conditioning_ranges.py,
results/challenge2/c2_cnn_conditioning_ranges.json); ``SCALAR_COND_NAMES``
feeds the q95/betaN heads only and additionally carries the Bt-derived
``bt_over_bp``/``log_bt_over_bp`` (q95 physically needs Bt; on the spherical
tokamak MAST the ratio is ~1.2 versus 4-7 on DIII-D, so it must never reach
the field). ``L`` is the major radius of the machine's envelope centroid
(DIII-D 1.5903 m, MAST 0.8397 m).

Feature-spec v2 (2026-09-21, after the adversarial review of the v1 build):
``tf_sign_rel`` (sign(I_TF) sign(Ip): -1 on every MAST shot, mixed on DIII-D)
was removed; ``(R-L)/L`` and ``Z/L`` (MAST |Z/L| reaches 2.38 vs 1.0 on DIII-D)
were replaced by the grid-normalised coordinates; the optional ``R/L`` channel
was checked and dropped because MAST's grid (R/L 0.07-2.4) extends far outside
DIII-D's (0.53-1.6). The numeric range audit further moved the pressure
scalars ``pe_mean``/``pe_peak`` (MAST p_e/P0 lies below the DIII-D q001) to the
scalar heads, removed ``pe_valid_frac`` (fraction of the padded chord capacity:
0.25-0.58 on DIII-D vs 0.6-1.0 on MAST, a machine artefact) and replaced the
absolute ``time_s`` (MAST 0-0.4 s lies entirely inside DIII-D's ramp-up window)
by the shot fraction ``time_frac`` in [0, 1]. The constant cell-size channels
``dR/L``/``dZ/L`` differ between machines by construction (0.017/0.031 DIII-D vs
0.036/0.074 MAST) and are kept as a documented exception (task instruction:
physical scale). ``feature_spec()`` is stored in run.json and every checkpoint;
loaders refuse a differing spec.

Deviation from the brief: channel 3 is the *weight-normalised* average
``splat / hits`` rather than the raw accumulated splat, because chord density
per cell differs between machines (MAST core 130 chords over 1.3 m vs DIII-D
44 chords over 0.7 m); the raw density information is kept in channel 4.

DIII-D frames come from the prepared feature caches (``X`` rows -> currents in
kA, log1p Thomson Te/ne per channel with chord coordinates); MAST frames come
from ``fusion.physics_fields.load_shot_inputs``. The DIII-D training target is
``phi = -efit_psirz`` (axis maximum), predicted as
``phi_hat = phi_ext + F * (resid + offset)``.

Label use: DIII-D train/val labels only through the caches. No MAST labels exist
locally and none are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from fusion import physics_fields as pf
from fusion.data import arrow_to_numpy, interpolate_signal

ROOT = Path(__file__).resolve().parents[1]
MU0 = 4.0e-7 * np.pi
E_CHARGE = 1.602176634e-19
IP_FLOOR_KA = 50.0            # floors |Ip| in the normalisation scales only (bogus ~0 kA frames)
SIGMA_BLOB = 0.25             # m, validated in results/challenge2/physics_fields_v1
A_EST_FRACTION = 0.35         # fixed machine-agnostic minor-radius proxy a_est = 0.35 L
TF_TURNS = {"DIII-D": 144, "MAST": 24}   # fusion/profile_scalars.py (published hardware)
TF_SIGNAL = {"DIII-D": "magnetics_bcoil", "MAST": "magnetics_tf_current"}
RAW_SPLIT = {"DIII-D": "diii_d_train", "MAST": "mast_public_test"}
TABLE_CACHE = ROOT / "data" / "c2_cnn_tables"
HITS_CLIP = 8.0

FEATURE_SPEC_VERSION = 3
CHANNEL_NAMES = ["phi_ext", "phi_coils", "phi_plasma", "pe_avg", "pe_hits", "envelope",
                 "u_R", "u_Z"]  # v3: dR_over_L/dZ_over_L dropped (constant per machine -> machine switch)
# Every per-frame scalar the featurizer computes (superset; diagnostics included for the range check).
ENTRY_NAMES = ["log_ip_MA", "ip_sign", "time_frac", "coil_norm", "ip_observed", "pe_mean", "pe_peak",
               "bt_over_bp", "log_bt_over_bp", "time_s", "pe_valid_frac", "tf_sign_rel"]
# FiLM (field) conditioning: MAST q01-q99 inside DIII-D train q001-q999 for every entry (see module docstring).
FIELD_COND_NAMES = ["log_ip_MA", "ip_sign", "time_frac", "coil_norm", "ip_observed"]
# q95/betaN head conditioning: field entries plus pressure scalars and the Bt-derived ratio (deliberate MAST
# extrapolation on the low-pressure / low-Bt side, heads only).
SCALAR_COND_NAMES = FIELD_COND_NAMES + ["pe_mean", "pe_peak", "bt_over_bp", "log_bt_over_bp"]
# Diagnostics computed but used nowhere: time_s (absolute seconds), pe_valid_frac (padding-capacity artefact),
# tf_sign_rel (machine switch).
# Coordinate diagnostics evaluated by the range check but NOT used as channels.
COORDINATE_DIAGNOSTICS = ["x_R", "x_Z", "R_over_L"]
N_CHANNELS = len(CHANNEL_NAMES)
N_FIELD_COND = len(FIELD_COND_NAMES)
N_SCALAR_COND = len(SCALAR_COND_NAMES)


def feature_spec() -> dict:
    """Exact input contract of the model; stored in run.json/checkpoints and compared on load."""
    return {"version": FEATURE_SPEC_VERSION, "channel_names": list(CHANNEL_NAMES),
            "field_cond_names": list(FIELD_COND_NAMES), "scalar_cond_names": list(SCALAR_COND_NAMES),
            "coordinates": "grid_normalised_[-1,1]", "time": "time_frac = (t - t_first) / (t_last - t_first)", "sigma_blob_m": SIGMA_BLOB, "a_est_fraction": A_EST_FRACTION,
            "hits_clip": HITS_CLIP, "ip_floor_kA": IP_FLOOR_KA, "pe_channel": "log1p(splat/hits/P0)",
            "target": "(phi - phi_ext) / F, phi = axis-maximum flux"}


def feature_spec_hash(spec: dict | None = None) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(feature_spec() if spec is None else spec, sort_keys=True).encode()).hexdigest()[:16]


def check_feature_spec(saved: dict | None, what: str = "checkpoint") -> None:
    """Raise unless ``saved`` equals the current feature_spec() (refuses v1 checkpoints without a spec)."""
    current = feature_spec()
    if saved is None:
        raise ValueError(f"{what} carries no feature_spec (pre-v{FEATURE_SPEC_VERSION} contract); refusing to load")
    if saved != current:
        diff = {k: (saved.get(k), current.get(k)) for k in sorted(set(saved) | set(current)) if saved.get(k) != current.get(k)}
        raise ValueError(f"{what} feature_spec differs from the current contract: {json.dumps(diff, default=str)}")

# DIII-D advanced cache layout (feature_names in data/advanced_v1/manifest.json)
D3D_THOMSON_CAPACITY = {"core": 64, "edge": 32}
D3D_K = sum(D3D_THOMSON_CAPACITY.values())


# --------------------------------------------------------------------------- #
# Machine setup
# --------------------------------------------------------------------------- #
@dataclass
class MachineSetup:
    machine: str
    grid_R: np.ndarray
    grid_Z: np.ndarray
    envelope: np.ndarray               # (65, 65) bool
    centre: tuple[float, float]        # envelope centroid (Rc, Zc)
    table_names: list[str]             # geometric current signals, sorted
    table_matrix: np.ndarray           # (C, 65*65) Wb/rad per kA of signal, multipliers applied
    blob: np.ndarray                   # (65, 65) Wb/rad per kA of plasma current
    multipliers: dict = field(default_factory=dict)
    table_hash: str = ""

    @property
    def L(self) -> float:
        return float(self.centre[0])

    @property
    def tf_turns(self) -> int:
        return TF_TURNS[self.machine]

    def summary(self) -> dict:
        return {"machine": self.machine, "L": self.L, "centre": list(self.centre), "sigma": SIGMA_BLOB,
                "table_names": self.table_names, "multipliers": self.multipliers, "table_hash": self.table_hash,
                "grid_R": [float(self.grid_R[0]), float(self.grid_R[-1])], "grid_Z": [float(self.grid_Z[0]), float(self.grid_Z[-1])],
                "tf_turns": self.tf_turns, "ip_floor_kA": IP_FLOOR_KA, "a_est_fraction": A_EST_FRACTION}


def reference_parquet(machine: str) -> Path:
    files = sorted((ROOT / "data/raw/data" / RAW_SPLIT[machine]).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet for {machine}")
    return files[0]


def build_machine_setup(machine: str, cache_dir: Path = TABLE_CACHE, row: dict | None = None) -> MachineSetup:
    """Coil response tables (disk cached by physics_fields), blob table (cached here), envelope."""
    if row is None:
        row = pf.read_shot_row(reference_parquet(machine))
    if str(row["source"]) != machine:
        raise ValueError(f"Reference row is {row['source']}, expected {machine}")
    grid_R, grid_Z, envelope = pf.load_envelope(machine)
    if not (np.allclose(row["efit_grid_R"], grid_R) and np.allclose(row["efit_grid_Z"], grid_Z)):
        raise ValueError(f"{machine}: raw grid differs from the scorer envelope grid")
    tables = pf.coil_response_table(row, n=4)
    names = sorted(tables)
    multipliers = {}
    if machine == "DIII-D":
        calibration = pf.load_calibration()
        if "ecoila_effective_turns" in calibration:
            multipliers["magnetics_ECOILA"] = float(calibration["ecoila_effective_turns"])
    matrix = np.stack([tables[k].ravel() * multipliers.get(k, 1.0) for k in names]).astype(np.float64)
    centre = pf.envelope_centroid(machine)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob_path = cache_dir / f"{pf.MACHINE_TAGS[machine]}_blob_sigma{SIGMA_BLOB:g}.npz"
    if blob_path.exists():
        with np.load(blob_path) as data:
            blob = data["blob"]
    else:
        blob = pf.plasma_blob_table((grid_R, grid_Z), centre[0], centre[1], SIGMA_BLOB)
        np.savez(blob_path, blob=blob, centre=np.asarray(centre), sigma=SIGMA_BLOB)
    return MachineSetup(machine, grid_R, grid_Z, envelope, centre, names, matrix, blob, multipliers,
                        pf.geometry_hash(row, 4))


# --------------------------------------------------------------------------- #
# Compact per-shot inputs
# --------------------------------------------------------------------------- #
@dataclass
class ShotInputs:
    """Everything needed to build channels for one shot; arrays are float32 unless noted."""
    machine: str
    shot_key: str
    frames: np.ndarray            # (T,) original frame indices
    times_ms: np.ndarray          # (T,)
    currents: np.ndarray          # (T, C) kA in setup.table_names order
    Ip_kA: np.ndarray             # (T,)
    Ip_observed: np.ndarray       # (T,) bool
    ip_sign: float
    tf_kA: np.ndarray             # (T,)
    chord_R: np.ndarray           # (K,)
    chord_Z: np.ndarray           # (K,)
    pe: np.ndarray                # (T, K) Pa
    pe_mask: np.ndarray           # (T, K) bool
    phi: np.ndarray | None = None       # (T, 65, 65) axis-max target (DIII-D: -psi)
    scalars: np.ndarray | None = None   # (T, 2) q95, betaN
    meta: dict = field(default_factory=dict)
    xrow: np.ndarray | None = None      # (T, n_features) raw cache features (DIII-D only; opt-in full-X conditioning)

    @property
    def n_frames(self) -> int:
        return len(self.times_ms)


def uniform_frames(n: int, cap: int) -> np.ndarray:
    if cap <= 0 or cap >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(int))


class D3DFeatureIndex:
    """Column indices into the advanced cache ``X`` rows."""

    def __init__(self, feature_names: list[str], table_names: list[str]):
        idx = {name: i for i, name in enumerate(feature_names)}
        self.n_features = len(feature_names)
        self.currents = [idx[k] for k in table_names]
        self.ip = idx["magnetics_plasma_current"]
        self.ip_observed = idx["magnetics_plasma_current_observed"]
        self.tf = idx[TF_SIGNAL["DIII-D"]]
        self.time = idx["time_seconds"]
        self.thomson = {}
        for system, capacity in D3D_THOMSON_CAPACITY.items():
            self.thomson[system] = {
                key: idx[f"thomson_{system}_{key}_channel_00"]
                for key in ("Te_log1p", "Te_observed", "ne_log1p", "ne_observed", "R", "Z", "geometry_observed")}
            for key, start in self.thomson[system].items():
                names = feature_names[start:start + capacity]
                if any(not n.startswith(f"thomson_{system}_{key}_channel_") for n in names):
                    raise ValueError(f"Non-contiguous Thomson block {system}/{key}")


def d3d_shot_inputs(npz_path: Path, index: D3DFeatureIndex, frames_cap: int = 0,
                    with_labels: bool = True) -> ShotInputs:
    """Decode one prepared DIII-D cache file into compact physical inputs."""
    with np.load(npz_path, allow_pickle=False) as shot:
        X = np.asarray(shot["X"], np.float64)
        if X.shape[1] != index.n_features:
            raise ValueError(f"{npz_path}: {X.shape[1]} features, expected {index.n_features}")
        n = len(X)
        sel = uniform_frames(n, frames_cap)
        X = X[sel]
        times = np.asarray(shot["times"], np.float64)[sel]
        frame_indices = (np.asarray(shot["frame_indices"]) if "frame_indices" in shot else np.arange(n))[sel]
        phi = scalars = None
        if with_labels and "psi" in shot:
            phi = (-np.asarray(shot["psi"], np.float32)[sel]).astype(np.float32)
            scalars = np.asarray(shot["scalars"], np.float32)[sel]
        shot_key = str(shot["shot_id"]) if "shot_id" in shot else npz_path.stem
    currents = X[:, index.currents]
    ip = X[:, index.ip]
    ip_obs = X[:, index.ip_observed] > 0.5
    sign = pf.shot_ip_sign(ip)
    tf = X[:, index.tf]
    chord_R, chord_Z, pe, mask = [], [], [], []
    for system, capacity in D3D_THOMSON_CAPACITY.items():
        b = index.thomson[system]
        te = np.expm1(np.maximum(X[:, b["Te_log1p"]:b["Te_log1p"] + capacity], 0.0))
        ne = np.expm1(np.maximum(X[:, b["ne_log1p"]:b["ne_log1p"] + capacity], 0.0))
        ok = ((X[:, b["Te_observed"]:b["Te_observed"] + capacity] > 0.5)
              & (X[:, b["ne_observed"]:b["ne_observed"] + capacity] > 0.5)
              & (te > 0) & (ne > 0)
              & (X[:, b["geometry_observed"]:b["geometry_observed"] + capacity] > 0.5))
        chord_R.append(X[0, b["R"]:b["R"] + capacity])
        chord_Z.append(X[0, b["Z"]:b["Z"] + capacity])
        pe.append(E_CHARGE * te * ne)
        mask.append(ok)
    return ShotInputs("DIII-D", shot_key, frame_indices, times, currents.astype(np.float32), ip.astype(np.float32),
                      ip_obs, float(sign), tf.astype(np.float32),
                      np.concatenate(chord_R).astype(np.float32), np.concatenate(chord_Z).astype(np.float32),
                      np.concatenate(pe, axis=1).astype(np.float32), np.concatenate(mask, axis=1),
                      phi, scalars, {"path": str(npz_path)}, xrow=X.astype(np.float32))


def read_tf_current(path: Path, machine: str, query_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Toroidal-field coil current (kA) on the shared magnetics time axis at the query times."""
    import pyarrow.parquet as pq
    key = TF_SIGNAL[machine]
    names = pq.ParquetFile(path).schema_arrow.names
    if key not in names:
        return np.zeros(len(query_ms), np.float32), np.zeros(len(query_ms), bool)
    table = pq.read_table(path, columns=["magnetics_time", key])
    value, observed = interpolate_signal(arrow_to_numpy(table["magnetics_time"][0]), arrow_to_numpy(table[key][0]), query_ms)
    return value.astype(np.float32), observed.astype(bool)


def mast_shot_inputs(parquet_path: Path, setup: MachineSetup, frames: np.ndarray | None = None) -> ShotInputs:
    """Compact inputs for one MAST public-test shot (all native frames by default)."""
    inputs = pf.load_shot_inputs(parquet_path, frames=frames, with_labels=False, thomson=True)
    if inputs["machine"] != "MAST":
        raise ValueError(f"{parquet_path} is not a MAST shot")
    if not (np.allclose(inputs["grid_R"], setup.grid_R) and np.allclose(inputs["grid_Z"], setup.grid_Z)):
        raise ValueError(f"{parquet_path}: grid differs from the MAST envelope grid")
    if pf.geometry_hash(inputs["row"], 4) != setup.table_hash:
        raise ValueError(f"{parquet_path}: conductor geometry differs from the cached MAST table")
    T = len(inputs["efit_times"])
    currents = np.zeros((T, len(setup.table_names)), np.float32)
    unobserved = {}
    for j, name in enumerate(setup.table_names):
        if name in inputs["currents"]:
            currents[:, j] = inputs["currents"][name]
            unobserved[name] = int((~inputs["current_masks"][name]).sum())
        else:
            unobserved[name] = T
    tf, tf_obs = read_tf_current(parquet_path, "MAST", inputs["efit_times"])
    return ShotInputs("MAST", parquet_path.stem, np.asarray(inputs["frames"]), inputs["efit_times"], currents,
                      inputs["Ip_kA"].astype(np.float32), inputs["Ip_mask"].astype(bool), float(inputs["Ip_sign"]), tf,
                      inputs["chord_R"].astype(np.float32), inputs["chord_Z"].astype(np.float32),
                      inputs["pe"].astype(np.float32), inputs["pe_mask"].astype(bool), None, None,
                      {"path": str(parquet_path), "unobserved_current_frames": unobserved,
                       "tf_unobserved_frames": int((~tf_obs).sum())})


# --------------------------------------------------------------------------- #
# Batched storage
# --------------------------------------------------------------------------- #
class FrameStore:
    """Concatenated compact inputs for many shots of one machine (numpy, then torch)."""

    def __init__(self, shots: list[ShotInputs], K: int | None = None, with_labels: bool = True):
        if not shots:
            raise ValueError("No shots")
        self.machine = shots[0].machine
        K = max(s.pe.shape[1] for s in shots) if K is None else K
        self.K = K
        n = sum(s.n_frames for s in shots)
        C = shots[0].currents.shape[1]
        self.currents = np.zeros((n, C), np.float32)
        self.ip = np.zeros(n, np.float32)
        self.ip_observed = np.zeros(n, bool)
        self.tf = np.zeros(n, np.float32)
        self.time_s = np.zeros(n, np.float32)
        self.time_frac = np.zeros(n, np.float32)
        self.pe = np.zeros((n, K), np.float32)
        self.pe_mask = np.zeros((n, K), bool)
        self.shot = np.zeros(n, np.int64)
        self.frame = np.zeros(n, np.int64)
        self.chord_R = np.full((len(shots), K), np.nan, np.float32)
        self.chord_Z = np.full((len(shots), K), np.nan, np.float32)
        self.ip_sign = np.zeros(len(shots), np.float32)
        self.shot_keys = [s.shot_key for s in shots]
        self.offsets = np.zeros(len(shots) + 1, np.int64)
        self.phi = np.zeros((n, 65, 65), np.float32) if with_labels else None
        self.scalars = np.full((n, 2), np.nan, np.float32) if with_labels else None
        start = 0
        for i, s in enumerate(shots):
            end = start + s.n_frames
            k = s.pe.shape[1]
            self.currents[start:end] = s.currents
            self.ip[start:end] = s.Ip_kA
            self.ip_observed[start:end] = s.Ip_observed
            self.tf[start:end] = s.tf_kA
            self.time_s[start:end] = (s.times_ms - s.times_ms[0]) / 1000.0
            self.time_frac[start:end] = (s.times_ms - s.times_ms[0]) / max(float(s.times_ms[-1] - s.times_ms[0]), 1.0e-3)
            self.pe[start:end, :k] = s.pe
            self.pe_mask[start:end, :k] = s.pe_mask
            self.shot[start:end] = i
            self.frame[start:end] = s.frames
            self.chord_R[i, :k] = s.chord_R
            self.chord_Z[i, :k] = s.chord_Z
            self.ip_sign[i] = s.ip_sign
            if with_labels:
                if s.phi is None:
                    raise ValueError(f"{s.shot_key}: labels requested but absent")
                self.phi[start:end] = s.phi
                self.scalars[start:end] = s.scalars
            self.offsets[i + 1] = end
            start = end
        self.x = (np.concatenate([s.xrow for s in shots]).astype(np.float32)
                  if all(s.xrow is not None for s in shots) else None)
        self.n = n

    # time_frac uses the first/last loaded frame of the shot: the uniform frame subsample always keeps both ends,
    # and prediction uses every native frame, so train and inference agree on both machines.

    def frame_mask(self, min_ip_kA: float) -> np.ndarray:
        return np.abs(self.ip) >= min_ip_kA

    def to_torch(self, device, with_labels: bool = True) -> dict:
        import torch
        out = {"currents": torch.as_tensor(self.currents, device=device),
               "ip": torch.as_tensor(self.ip, device=device),
               "ip_observed": torch.as_tensor(self.ip_observed, device=device),
               "tf": torch.as_tensor(self.tf, device=device),
               "time_s": torch.as_tensor(self.time_s, device=device),
               "time_frac": torch.as_tensor(self.time_frac, device=device),
               "pe": torch.as_tensor(self.pe, device=device),
               "pe_mask": torch.as_tensor(self.pe_mask, device=device),
               "shot": torch.as_tensor(self.shot, device=device),
               "chord_R": torch.as_tensor(np.nan_to_num(self.chord_R, nan=-1.0e3), device=device),
               "chord_Z": torch.as_tensor(np.nan_to_num(self.chord_Z, nan=-1.0e3), device=device),
               "ip_sign": torch.as_tensor(self.ip_sign, device=device)}
        if self.x is not None:
            out["x"] = torch.as_tensor(self.x, device=device)
        if with_labels and self.phi is not None:
            out["phi"] = torch.as_tensor(self.phi, device=device)
            out["scalars"] = torch.as_tensor(self.scalars, device=device)
        return out


def batch_from_store(store: dict, indices) -> dict:
    """Gather one batch (dict of tensors) incl. shot-level chord geometry."""
    shot = store["shot"][indices]
    batch = {"currents": store["currents"][indices], "ip": store["ip"][indices], "ip_observed": store["ip_observed"][indices],
             "tf": store["tf"][indices], "time_s": store["time_s"][indices], "time_frac": store["time_frac"][indices],
             "pe": store["pe"][indices],
             "pe_mask": store["pe_mask"][indices], "chord_R": store["chord_R"][shot], "chord_Z": store["chord_Z"][shot],
             "ip_sign": store["ip_sign"][shot]}
    if "x" in store:
        batch["x"] = store["x"][indices]
    if "phi" in store:
        batch["phi"] = store["phi"][indices]
        batch["scalars"] = store["scalars"][indices]
    return batch


# --------------------------------------------------------------------------- #
# Torch featurizer
# --------------------------------------------------------------------------- #
class FrameFeaturizer:
    """Builds channels, conditioning, phi_ext and F for a batch (torch, any device)."""

    def __init__(self, setup: MachineSetup, device, dtype=None):
        import torch
        dtype = torch.float32 if dtype is None else dtype
        self.setup = setup
        self.device = device
        self.L = setup.L
        self.tf_turns = float(setup.tf_turns)
        R, Z = setup.grid_R, setup.grid_Z
        self.R0, self.Z0 = float(R[0]), float(Z[0])
        self.dR, self.dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
        self.nR, self.nZ = len(R), len(Z)
        self.table = torch.as_tensor(setup.table_matrix.T, dtype=dtype, device=device)  # (4225, C)
        self.blob = torch.as_tensor(setup.blob, dtype=dtype, device=device)             # (65, 65)
        coords = self.coordinate_maps()
        const = np.stack([setup.envelope.astype(np.float32), coords["u_R"], coords["u_Z"]])
        self.const = torch.as_tensor(const, dtype=dtype, device=device)                 # (3, 65, 65)
        self.envelope = torch.as_tensor(setup.envelope, device=device)

    def coordinate_maps(self) -> dict:
        """Constant (65, 65) float32 maps: the used grid-normalised u_R/u_Z plus the range-check diagnostics."""
        R, Z = self.setup.grid_R, self.setup.grid_Z
        RR, ZZ = np.meshgrid(R, Z)
        u_r = np.linspace(-1.0, 1.0, self.nR, dtype=np.float32)
        u_z = np.linspace(-1.0, 1.0, self.nZ, dtype=np.float32)
        UR, UZ = np.meshgrid(u_r, u_z)
        return {"u_R": UR.astype(np.float32), "u_Z": UZ.astype(np.float32),
                "x_R": ((RR - self.L) / self.L).astype(np.float32), "x_Z": (ZZ / self.L).astype(np.float32),
                "R_over_L": np.clip(RR / self.L, 0.0, 3.0).astype(np.float32)}

    def scales(self, ip):
        import torch
        ip_eff = torch.clamp(ip.abs(), min=IP_FLOOR_KA) * 1000.0                        # A
        F = MU0 * ip_eff * self.L / 2.0
        P0 = MU0 * ip_eff ** 2 / (8.0 * np.pi ** 2 * self.L ** 2)
        return ip_eff, F, P0

    def splat(self, chord_R, chord_Z, values, mask):
        import torch
        B, K = values.shape
        fr = (chord_R - self.R0) / self.dR
        fz = (chord_Z - self.Z0) / self.dZ
        inside = mask & torch.isfinite(values) & (fr >= 0) & (fr <= self.nR - 1) & (fz >= 0) & (fz <= self.nZ - 1)
        fr = torch.nan_to_num(fr, nan=0.0).clamp(0, self.nR - 1)
        fz = torch.nan_to_num(fz, nan=0.0).clamp(0, self.nZ - 1)
        ir = fr.floor().clamp(0, self.nR - 2).long()
        iz = fz.floor().clamp(0, self.nZ - 2).long()
        tr, tz = fr - ir, fz - iz
        v = torch.where(inside, values, torch.zeros_like(values))
        m = inside.to(values.dtype)
        base = (torch.arange(B, device=values.device) * (self.nZ * self.nR))[:, None]
        splat = torch.zeros(B * self.nZ * self.nR, dtype=values.dtype, device=values.device)
        hits = torch.zeros_like(splat)
        for dz, dr, w in ((0, 0, (1 - tz) * (1 - tr)), (0, 1, (1 - tz) * tr), (1, 0, tz * (1 - tr)), (1, 1, tz * tr)):
            idx = (base + (iz + dz) * self.nR + (ir + dr)).reshape(-1)
            splat.index_add_(0, idx, (w * m * v).reshape(-1))
            hits.index_add_(0, idx, (w * m).reshape(-1))
        return splat.view(B, self.nZ, self.nR), hits.view(B, self.nZ, self.nR), inside

    def __call__(self, batch: dict) -> dict:
        import torch
        ip = batch["ip"]
        B = ip.shape[0]
        sign = batch["ip_sign"]
        ip_eff, F, P0 = self.scales(ip)
        coil = ((batch["currents"] * sign[:, None]) @ self.table.T).view(B, self.nZ, self.nR)
        plasma = ip.abs()[:, None, None] * self.blob[None]
        phi_ext = coil + plasma
        splat, hits, inside = self.splat(batch["chord_R"], batch["chord_Z"], batch["pe"], batch["pe_mask"])
        pe_avg = torch.where(hits > 1.0e-3, splat / hits.clamp_min(1.0e-3), torch.zeros_like(splat)) / P0[:, None, None]
        pe_avg = torch.log1p(pe_avg.clamp_min(0.0))   # beta-like, log-compressed (heavy-tailed on low-Ip frames)
        Fb = F[:, None, None]
        channels = torch.cat([(phi_ext / Fb)[:, None], (coil / Fb)[:, None], (plasma / Fb)[:, None],
                              pe_avg[:, None], hits.clamp(max=HITS_CLIP)[:, None],
                              self.const[None].expand(B, -1, -1, -1)], dim=1)
        coil_norm = (coil / Fb).flatten(1).pow(2).mean(1).sqrt()
        n_valid = inside.sum(1).clamp_min(1)
        pe_mean = torch.log1p(((torch.where(inside, batch["pe"], torch.zeros_like(batch["pe"])).sum(1) / n_valid) / P0).clamp_min(0.0))
        pe_valid = inside.float().mean(1)
        pe_peak = torch.log1p((torch.where(inside, batch["pe"], torch.zeros_like(batch["pe"])).amax(1) / P0).clamp_min(0.0))
        bt_over_bp = A_EST_FRACTION * self.tf_turns * batch["tf"].abs() * 1000.0 / ip_eff
        entries = {"log_ip_MA": torch.log(ip_eff / 1.0e6), "ip_sign": sign, "time_frac": batch["time_frac"],
                   "coil_norm": coil_norm, "ip_observed": batch["ip_observed"].float(),
                   "pe_mean": pe_mean, "pe_peak": pe_peak, "bt_over_bp": bt_over_bp,
                   "log_bt_over_bp": torch.log(bt_over_bp.clamp_min(1.0e-3)),
                   "time_s": batch["time_s"], "pe_valid_frac": pe_valid, "tf_sign_rel": torch.sign(batch["tf"]) * sign}
        assert list(entries) == ENTRY_NAMES
        cond = torch.stack([entries[k] for k in FIELD_COND_NAMES], dim=1)
        if getattr(self, "x_mean", None) is not None:
            # Opt-in DIII-D-only conditioning on the full standardised cache feature row (members' input contract).
            xs = ((torch.nan_to_num(batch["x"]) - self.x_mean) / self.x_std).clamp(-12.0, 12.0)
            cond = torch.cat([cond, xs], dim=1)
        scalar_cond = torch.stack([entries[k] for k in SCALAR_COND_NAMES], dim=1)
        out = {"channels": channels, "cond": cond, "scalar_cond": scalar_cond, "entries": entries,
               "phi_ext": phi_ext, "F": F}
        if "phi" in batch:
            out["target"] = (batch["phi"] - phi_ext) / Fb
            out["scalars"] = batch["scalars"]
        return out


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def read_manifest(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def split_entries(data_dir: Path, split: str) -> list[dict]:
    manifest = read_manifest(data_dir)
    return manifest.get("splits", manifest)[split]


def load_d3d_split(data_dir: Path, split: str, index: D3DFeatureIndex, frames_cap: int = 0,
                   max_shots: int = 0, shot_indices: set[int] | None = None, positions: list[int] | None = None,
                   with_labels: bool = True, verbose: bool = True) -> tuple[list[ShotInputs], list[int]]:
    """Load prepared DIII-D shots; returns (shots, manifest positions)."""
    data_dir = Path(data_dir)
    entries = split_entries(data_dir, split)
    chosen = list(range(len(entries))) if positions is None else list(positions)
    if shot_indices is not None:
        chosen = [i for i in chosen if int(entries[i]["shot_index"]) in shot_indices]
    if max_shots > 0:
        chosen = chosen[:max_shots]
    shots = []
    for count, i in enumerate(chosen):
        shots.append(d3d_shot_inputs(data_dir / entries[i]["path"], index, frames_cap, with_labels))
        if verbose and (count + 1) % 500 == 0:
            print(f"loaded {split}: {count + 1}/{len(chosen)} shots", flush=True)
    return shots, chosen


def mast_parquet_paths(raw_dir: Path = ROOT / "data/raw") -> list[Path]:
    """Official MAST shot order: lexicographically sorted pinned manifest paths (package_submission.py)."""
    manifest = json.loads((Path(raw_dir) / "download_manifest.json").read_text())
    paths = sorted(item["path"] for item in manifest["files"]
                   if item["path"].startswith("data/mast_public_test/") and item["path"].endswith(".parquet"))
    return [Path(raw_dir) / p for p in paths]
