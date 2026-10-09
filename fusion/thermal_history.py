"""Fixed-lag Thomson input features; no equilibrium targets are accessed."""
from __future__ import annotations

import numpy as np

from fusion.advanced_data import THOMSON_CAPACITY, _profiles
from fusion.data import interpolate_signal

OFFSETS_MS = (-1000, -500, -200, 100)


def feature_names():
    return [f"thermal_{system}_{measure}_{field}_{channel:02d}_{lag:+d}ms"
            for lag in OFFSETS_MS for system, count in THOMSON_CAPACITY.items()
            for measure in ("Te", "ne", "pressure") for field in ("log1p_scaled", "observed")
            for channel in range(count)]


def extract(row, query):
    query = np.asarray(query, dtype=np.float64).ravel()
    if not np.isfinite(query).all():
        raise ValueError("Nonfinite query times")
    shifted = query[:, None] + np.asarray(OFFSETS_MS)[None]
    by_system = {}
    for system, count in THOMSON_CAPACITY.items():
        values, masks = [], []
        for measure, scale in (("Te", 1e3), ("ne", 1e19)):
            profiles, times = _profiles(row, system, measure)
            v = np.zeros((len(query), len(OFFSETS_MS), count), dtype=np.float32)
            m = np.zeros_like(v)
            for channel in range(profiles.shape[1]):
                signal, observed = interpolate_signal(times, profiles[:, channel], shifted.ravel())
                # Endpoint extrapolation is excluded from this additional path.
                v[:, :, channel] = np.where(observed, np.maximum(signal, 0)/scale, 0).reshape(shifted.shape)
                m[:, :, channel] = observed.reshape(shifted.shape)
            values.append(v); masks.append(m)
        by_system[system] = [(np.log1p(v), m) for v, m in zip(values, masks)]
        observed = masks[0]*masks[1]
        by_system[system].append((np.log1p(values[0]*values[1]*observed), observed))
    parts = [value[:, lag, :] for lag in range(len(OFFSETS_MS))
             for system in THOMSON_CAPACITY for pair in by_system[system] for value in pair]
    output = np.concatenate(parts, axis=1)
    if output.shape != (len(query), len(feature_names())) or not np.isfinite(output).all():
        raise ValueError("Invalid history feature matrix")
    return output


def analytic_check():
    times = np.array([0., 500., 1000.])
    row = {"thomson_core_times": times, "thomson_core_Te": np.column_stack((1000+times, np.full(3, np.nan))),
           "thomson_core_ne": np.column_stack((1e19+times*1e16, np.full(3, np.nan)))}
    query = np.array([500.])
    result = extract(row, query)
    names = feature_names()
    for lag in OFFSETS_MS:
        t = query[0]+lag
        valid = 0 <= t <= 1000
        te, ne = (1+t/1000, 1+t/1000) if valid else (0., 0.)
        for measure, expected in (("Te", te), ("ne", ne), ("pressure", te*ne)):
            value = names.index(f"thermal_core_{measure}_log1p_scaled_00_{lag:+d}ms")
            mask = names.index(f"thermal_core_{measure}_observed_00_{lag:+d}ms")
            np.testing.assert_allclose(result[0, value], np.log1p(expected), rtol=1e-6, atol=1e-7)
            assert result[0, mask] == float(valid)
            missing = names.index(f"thermal_core_{measure}_log1p_scaled_01_{lag:+d}ms")
            assert result[0, missing] == 0
    poisoned = dict(row, efit_psirz=np.full((3, 65, 65), 1e30), efit_q95=np.full(3, -1e30),
                    efit_beta_n=np.full(3, np.nan), shot_id="unused_identity")
    np.testing.assert_array_equal(extract(poisoned, query), result)
    return {"passed": True, "analytic_interpolation": True, "physical_scaling": True,
            "extrapolated_and_missing_values_zero": True, "target_and_identity_independence": True,
            "features": len(names)}
