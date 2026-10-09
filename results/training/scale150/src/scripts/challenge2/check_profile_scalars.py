#!/usr/bin/env python3
"""Independent circular-equilibrium check of the new scalar functional."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.profile_scalars import closure_scalars, vacuum_toroidal_function
from fusion.mast_physics import MU0


def main():
    R, Z = np.linspace(.7, 2.7, 129), np.linspace(-1, 1, 129)
    RR, ZZ = np.meshgrid(R, Z)
    r0, a, delta, paxis, F = 1.7, .4, .2, 1e4, 3.
    rho = ((RR-r0)**2+ZZ**2)/a**2
    phi, inside = delta*(1-rho), rho <= 1
    solver = SimpleNamespace(R=R, Z=Z, RR=RR, ZZ=ZZ, dR=R[1]-R[0], dZ=Z[1]-Z[0])
    theta = np.linspace(0, 2*np.pi, 2048, endpoint=False)
    contour = np.column_stack((r0+a*np.cos(theta), a*np.sin(theta)))
    surface = contour, [r0, 0, 64., 64.], 0., delta, inside
    # This current cancels FF' exactly in the chosen constant-profile closure.
    ip = paxis/delta*np.sum(RR*inside)*solver.dR*solver.dZ
    observed = closure_scalars(phi, solver, inside, ip, paxis, np.ones(4), np.ones(4), F, surface)
    analytic_q = F*a*a/(2*delta*np.sqrt(r0*r0-.95*a*a))
    analytic_beta = 100*MU0*paxis/(F/r0)**2*a*(F/r0)/(ip/1e6)
    errors = {"q95_relative_error": abs(observed["q95"]/analytic_q-1),
              "betaN_relative_error": abs(observed["betaN"]/analytic_beta-1)}
    assert errors["q95_relative_error"] < .005
    assert errors["betaN_relative_error"] < .02
    # A constant additive flux offset must leave physical scalars invariant.
    shifted = closure_scalars(phi+2.5, solver, inside, ip, paxis, np.ones(4), np.ones(4), F,
                              (contour, surface[1], 2.5, delta, inside))
    np.testing.assert_allclose([observed["q95"], observed["betaN"]], [shifted["q95"], shifted["betaN"]], rtol=1e-10)
    print(json.dumps({"passed": True, **errors, "flux_gauge_invariance": True}))


if __name__ == "__main__":
    main()
