#!/usr/bin/env python3
"""Check measured-pressure recovery against an independent analytic profile."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.measured_pressure import PressureConstraint, measurements
from fusion.mast_profiles import read_row


def main():
    R, Z = np.linspace(.7, 2.7, 129), np.linspace(-1, 1, 129)
    RR, ZZ = np.meshgrid(R, Z)
    r0, radius, delta = 1.7, .4, .2
    phi = delta*(1-((RR-r0)**2+ZZ**2)/radius**2)
    index = np.arange(44, 85)
    coords = np.column_stack((R[index], np.zeros(len(index))))
    s = 1-((coords[:, 0]-r0)/radius)**2
    pressure = 5000*s**2
    observations = {"positions": coords, "pressure": pressure, "observed": np.ones(len(s), bool)}
    fit = PressureConstraint(observations, 3e4, np.array([.1,.2,.3,.4]), regularization=0)
    solver = SimpleNamespace(R=R, Z=Z)
    surface = None, None, 0., delta, None
    value, shape, report = fit.fit(phi, solver, None, surface)
    np.testing.assert_allclose(value, 1e4, rtol=1e-10)
    assert report["pressure_relative_rmse"] < 1e-10
    shifted, _, _ = fit.fit(phi+2.5, solver, None, (None, None, 2.5, delta, None))
    np.testing.assert_allclose(shifted, value, rtol=1e-10)
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("A prediction feature accessed a target")
        def __getitem__(self, key):
            raise AssertionError("A prediction feature accessed a target")
    inventory = json.loads((ROOT / "data/raw/download_manifest.json").read_text())
    channels = {}
    for prefix in ("data/diii_d_train/", "data/mast_public_test/"):
        path = next(x["path"] for x in inventory["files"] if x["path"].startswith(prefix) and x["path"].endswith(".parquet"))
        row = read_row(ROOT / "data/raw" / path, labels=False)
        time = float(row["efit_times"][len(row["efit_times"])//2])
        original = measurements(row, time)
        row.update({key: Forbidden() for key in ("efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_rmaxis", "efit_zmaxis")})
        poisoned = measurements(row, time)
        for key in ("positions", "pressure", "observed"):
            np.testing.assert_array_equal(original[key], poisoned[key])
        assert original["systems"] == poisoned["systems"]
        channels[row["source"]] = len(original["pressure"])
        assert channels[row["source"]] >= 4
    print(json.dumps({"passed": True, "recovered_total_axis_pressure": value,
                      "relative_fit_error": report["pressure_relative_rmse"], "gauge_invariance": True,
                      "target_poison_invariance": True, "observed_channels": channels}))


if __name__ == "__main__":
    main()
