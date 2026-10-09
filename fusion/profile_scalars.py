"""Scalar predictions from a specified pressure/current closure and measured TF."""
import numpy as np
from skimage.measure import find_contours

from fusion.data import interpolate_signal
from fusion.mast_profiles import bernstein, bernstein_integral, N_BASIS
from fusion.mast_physics import MU0, _surface, _bilinear, is_closed, encloses_point, grid_to_physical

# Published hardware, not fitted to challenge labels:
# DIII-D: https://fusion.gat.com/pubs-ext/SOFT02/A24059.pdf
# MAST: https://etheses.whiterose.ac.uk/id/eprint/14209/1/516649.pdf
TF_TURNS = {"DIII-D": 144, "MAST": 24}


def vacuum_toroidal_function(row, time):
    machine = row["source"]
    key = "magnetics_bcoil" if machine == "DIII-D" else "magnetics_tf_current"
    current, observed = interpolate_signal(row["magnetics_time"], row[key], np.array([time]))
    value = abs(MU0 / (2*np.pi) * TF_TURNS[machine] * float(current[0]) * 1000)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("No positive finite measured toroidal field")
    return value, bool(observed[0])


def closure_scalars(phi, solver, envelope, ip, paxis, ashape, cshape, Fvac, surface=None,
                    basis_function=bernstein, integral_function=bernstein_integral):
    surface = _surface(phi, solver, envelope) if surface is None else surface
    if surface is None:
        raise ValueError("No closed surface for scalar reconstruction")
    contour, axis, edge, delta, inside = surface
    if delta <= 0 or ip <= 0 or Fvac <= 0:
        raise ValueError("Invalid flux, current, or field convention")
    ashape, cshape = np.asarray(ashape), np.asarray(cshape)
    anorm, cnorm = integral_function(1.)@ashape, cshape.sum()
    if anorm <= 0 or cnorm <= 0:
        raise ValueError("Invalid pressure/current profile normalization")
    s = np.clip((phi-edge)/delta, 0, 1)
    B = basis_function(s)
    inner = float(contour[:, 0].min())
    pprime = paxis/delta * (B@ashape)/anorm
    pressure_current = (solver.RR-inner**2/solver.RR)*pprime*inside
    current_shape = (B@cshape)/cnorm/solver.RR*inside
    amplitude = (ip-pressure_current.sum()*solver.dR*solver.dZ)/(current_shape.sum()*solver.dR*solver.dZ)
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Pressure/current closure is inadmissible")

    def F_squared(level):
        integral = integral_function(level)
        return Fvac**2 + 2*MU0*(delta*amplitude*(integral@cshape)/cnorm
                                - inner**2*paxis*(integral@ashape)/anorm)

    Fmin = float(np.min(F_squared(np.linspace(0, 1, 1001))))
    if Fmin <= 0:
        raise ValueError("Inferred toroidal field squared is nonpositive")
    q95 = None
    dz, dr = np.gradient(phi, solver.Z, solver.R)
    bp = np.hypot(dz, dr)/solver.RR
    for curve in find_contours(phi, edge+.05*delta):
        if is_closed(curve) and encloses_point(curve, axis[2], axis[3]):
            physical = grid_to_physical(curve, solver.R, solver.Z)
            local_bp = _bilinear(bp, solver.R, solver.Z, physical[:, 0], physical[:, 1])
            if np.any(local_bp <= 0):
                raise ValueError("Degenerate field on the safety-factor contour")
            integrand = 1/(physical[:, 0]**2*local_bp)
            dl = np.linalg.norm(np.roll(physical, -1, axis=0)-physical, axis=1)
            q95 = float(np.sqrt(F_squared(.05))/(2*np.pi)*np.sum(.5*(integrand+np.roll(integrand, -1))*dl))
            break
    if q95 is None:
        raise ValueError("No closed 95-percent safety-factor contour")
    pressure = paxis*(integral_function(s)@ashape)/anorm
    mean_pressure = np.sum(pressure[inside]*solver.RR[inside])/np.sum(solver.RR[inside])
    Rgeom = .5*(contour[:, 0].max()+contour[:, 0].min())
    minor = .5*np.ptp(contour[:, 0])
    Bt = Fvac/Rgeom
    betaN = float(100*2*MU0*mean_pressure/Bt**2*minor*Bt/(ip/1e6))
    return {"q95": q95, "betaN": betaN, "F_squared_min": Fmin, "Fvac": Fvac,
            "pressure_mean_Pa": float(mean_pressure), "scalar_valid": bool(np.isfinite([q95, betaN]).all())}
