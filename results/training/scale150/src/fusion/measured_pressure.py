"""Pressure constraints from supplied Thomson observations, never EFIT labels."""
import re
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import nnls

from fusion.data import interpolate_signal
from fusion.mast_profiles import align_mast_thomson_core
from fusion.mast_physics import E_CHARGE, _surface, _bilinear
from fusion.smooth_profiles import integral


def measurements(row, time):
    positions, pressures, observed, systems = [], [], [], []
    for system in ("core", "edge"):
        ts = np.asarray(row[f"thomson_{system}_times"], float)
        te, ne = [np.asarray(row[f"thomson_{system}_{name}"], float) for name in ("Te", "ne")]
        if te.shape[0] != len(ts):
            te, ne = te.T, ne.T
        aligned_r = None
        if row["source"] == "MAST" and system == "core":
            aligned_r, te, ne = align_mast_thomson_core(row)
        count = te.shape[1]
        coords, known = np.zeros((count, 2)), np.zeros(count, bool)
        group = "core" if system == "core" else "tangential"
        for name, r, z in zip(row["thomson_chord_name"], row["thomson_chord_R"], row["thomson_chord_Z"]):
            match = re.fullmatch(rf"TS_{group}_(?:.*_)?(\d+)", str(name))
            if match and int(match.group(1)) < count:
                index = int(match.group(1))
                coords[index], known[index] = (r, z), True
        if aligned_r is not None:
            coords[:, 0] = aligned_r
            known |= np.isfinite(aligned_r)
        values, masks = [], []
        for source in (te, ne):
            v, m = [], []
            for channel in range(count):
                value, mask = interpolate_signal(ts, np.where(source[:, channel] >= 0, source[:, channel], np.nan), np.array([time]))
                v.append(float(value[0])); m.append(bool(mask[0]))
            values.append(np.asarray(v)); masks.append(np.asarray(m))
        pressure = E_CHARGE*values[0]*values[1]
        valid = known & (values[0] > 0) & (values[1] > 0) & np.isfinite(pressure)
        filtered = median_filter(np.where(valid, pressure, 0), size=3, mode="nearest")
        valid &= filtered > 0
        positions.extend(coords[valid].tolist())
        pressures.extend(filtered[valid].tolist())
        observed.extend((masks[0] & masks[1])[valid].tolist())
        systems.extend([system]*int(valid.sum()))
    return {"positions": np.asarray(positions, float).reshape(-1, 2),
            "pressure": np.asarray(pressures, float), "observed": np.asarray(observed, bool), "systems": systems}


class PressureConstraint:
    def __init__(self, observations, prior_axis_pressure, prior_shape, total_to_electron=2., regularization=.03):
        self.observations = observations
        self.factor = float(total_to_electron)
        self.regularization = float(regularization)
        self.prior_axis = float(prior_axis_pressure)
        self.prior_shape = np.asarray(prior_shape, float)
        if self.factor < 1 or self.regularization < 0 or self.prior_axis <= 0:
            raise ValueError("Invalid pressure constraint configuration")

    def fit(self, phi, solver, envelope, surface=None):
        surface = _surface(phi, solver, envelope) if surface is None else surface
        if surface is None:
            raise ValueError("no_closed_surface_for_pressure_fit")
        _, _, edge, delta, _ = surface
        coords, pressure = self.observations["positions"], self.observations["pressure"]
        valid = ((coords[:, 0] >= solver.R[0]) & (coords[:, 0] <= solver.R[-1]) &
                 (coords[:, 1] >= solver.Z[0]) & (coords[:, 1] <= solver.Z[-1]))
        coords, pressure = coords[valid], pressure[valid]
        if len(pressure) < 4:
            return self.prior_axis, self.prior_shape, {"pressure_fit_used": False, "pressure_channels": len(pressure)}
        values = _bilinear(phi, solver.R, solver.Z, coords[:, 0], coords[:, 1])
        s = np.clip((values-edge)/delta, 0, 1)
        if np.count_nonzero(s > .02) < 4:
            return self.prior_axis, self.prior_shape, {"pressure_fit_used": False, "pressure_channels": len(pressure)}
        reference = max(float(np.quantile(pressure, .9)), 1.)
        B, y = integral(s), pressure/reference
        prior = self.prior_axis/self.factor*self.prior_shape/(integral(1.)@self.prior_shape)/reference
        scale = np.maximum(np.linalg.norm(B, axis=0), 1e-5)
        root_penalty = np.sqrt(self.regularization)
        coef = nnls(np.vstack((B/scale, root_penalty*np.eye(4))),
                    np.r_[y, root_penalty*prior*scale], maxiter=500)[0]/scale*reference
        axis_pressure = self.factor*float(integral(1.)@coef)
        if axis_pressure <= 0 or not np.isfinite(coef).all():
            raise ValueError("invalid_observed_pressure_fit")
        predicted = integral(s)@coef
        diagnostics = {"pressure_fit_used": True, "pressure_channels": len(pressure),
            "pressure_relative_rmse": float(np.linalg.norm(predicted-pressure)/max(np.linalg.norm(pressure), 1e-20)),
            "positive_pressure_outside_surface_fraction": float(np.mean(s <= 0)),
            "total_to_electron_pressure": self.factor,
            "pressure_endpoint_extrapolation": bool(np.any(~self.observations["observed"]))}
        return axis_pressure, coef, diagnostics
