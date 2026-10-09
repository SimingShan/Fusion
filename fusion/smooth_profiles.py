"""Edge-vanishing profile basis with consistent pressure integration."""
import math
import numpy as np

from fusion.mast_profiles import bernstein, DEGREE
from fusion.mast_physics import _surface


def basis(s):
    s = np.clip(np.asarray(s, float), 0, 1)
    return bernstein(s)*s[..., None]


def integral(s):
    s = np.clip(np.asarray(s, float), 0, 1)
    values = []
    for k in range(DEGREE+1):
        total = np.zeros_like(s)
        for j in range(DEGREE-k+1):
            total += math.comb(DEGREE, k)*math.comb(DEGREE-k, j)*(-1)**j*s**(k+j+2)/(k+j+2)
        values.append(total)
    return np.stack(values, axis=-1)


def current(phi, solver, envelope, ip, paxis, ashape, cshape, surface=None):
    surface = _surface(phi, solver, envelope) if surface is None else surface
    if surface is None:
        raise ValueError("no_closed_plasma_surface")
    contour, _, edge, delta, inside = surface
    s = np.clip((phi-edge)/delta, 0, 1)
    B = basis(s)
    pressure_norm = integral(1.)@ashape
    if pressure_norm <= 0 or np.sum(cshape) <= 0:
        raise ValueError("invalid_profile_normalization")
    pprime = paxis/delta * (B@ashape)/pressure_norm
    inner = float(contour[:, 0].min())
    pressure_current = (solver.RR-inner**2/solver.RR)*pprime*inside
    density_shape = (B@cshape)/np.sum(cshape)/solver.RR*inside
    amplitude = (ip-pressure_current.sum()*solver.dR*solver.dZ)/(density_shape.sum()*solver.dR*solver.dZ)
    if amplitude < 0:
        raise ValueError("pressure_requires_negative_current_amplitude")
    return pressure_current + amplitude*density_shape
