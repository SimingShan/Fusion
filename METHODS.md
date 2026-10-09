# Methods: DIII-D equilibrium reconstruction from actuator and Thomson signals (Challenge 1)

## Task and inputs
Each frame (an EFIT time) needs the 65x65 poloidal flux map psi(R, Z), q95 and betaN. Inputs are restricted to the challenge's measured
signals: the 18 F-coil currents, the E-coil and toroidal-field coil currents, the plasma current (with the starter-kit time fix), Thomson
core and edge electron temperature/density profiles with their chord geometry, and time/grid metadata. No magnetic probes, flux loops or
`magnetics_dsep` are available on test, so shape and position must be inferred from the actuators and the kinetic profiles.

Per frame we build 708 inputs: current values at the frame and at lags of -4 s to +100 ms (with observed flags), Thomson channel values
(log Te, log ne, positions, coverage) and binned summaries, and coil geometry. Two feature blocks proved to carry new information:
- **vessel-current features** (126): exponentially filtered time derivatives (tau = 2, 5, 10, 20, 50 ms) and a 2 ms slope of each of the
  21 currents from the raw 0.05 ms record; they proxy eddy currents in the vessel and the vertical-control response;
- **future-actuator features** (126): the same filters run backwards in time over the next 2-50 ms of the same shot. The plasma
  control system reacts to the plasma's present position and shape, so the coil response that follows a frame encodes that frame's
  state. These features help most in end-of-shot vertical drifts. They use only inputs within the same shot, never labels.

## Model
Each member is an MLP (width 256, depth 3, residual) that predicts the first 128 POD coefficients of psi (basis fitted on training flux)
plus q95 and betaN. The loss combines coefficient error (weighted by the POD quadratic form), field-gradient error, the scalar heads and a
fixed-contour internal-inductance term. Training has three stages: AdamW cold start (600 epochs), a refinement (30-60 epochs, amplitude
frame weights, li loss), and a short Muon stage (30 epochs) at a Muon/AdamW learning-rate ratio that was tuned per family (30 for history
members, 200 for no-history members). Two families are trained on all labelled shots:
- **LVF** (17 seeds): history + vessel + future features;
- **LAV** (26 seeds): the no-history recipe + vessel features.

## Ensemble and post-processing
The submission is the equal-weight mean of the 43 members. On frames with almost no usable Thomson data, only the history members
are averaged, because the no-history members degrade there. The blend's POD modes 129-512 are filled by a closure network that maps the
leading 64 coefficients to the higher modes (trained on training flux). Finally, the magnetic axis is moved inside the separatrix only
(weight clip((psi - psi_b)/(psi_axis - psi_b), 0, 1)), vertically, half-way toward the axis predicted by six direct regressors trained on the
same inputs; the separatrix is untouched and the shift is skipped on Thomson-poor frames.

## Results
Development (held-out shots, official scorer, train-only analogues of the members): S = 0.99684 on the native-frame fold and 0.99450
on the legacy fold for a 43-member structure, the same as our 57-training dev ensemble. Public leaderboard: 0.996271 for this 43-member
all-labelled entry (0.996276 for our largest development blend). Measured contributions on the public leaderboard: vessel features
-0.00032 D_LCFS and +0.00017 consistency; future features and the axis shift +0.00006 to +0.00008 S.

## Compute
Training is about 12-14 GPU-minutes per member when four members share one GPU. The full entry (43 members, regressors and closure)
takes roughly 10 h on an H100, or an estimated 11-14 h on an A100, which is within the 24 GPU-hour limit. CPU preparation takes
about 4-6 h.

## What did not help
Things we tested on the official scorer that did not help: larger output rank, U-Net and transformer members, Green's-function/GS
physics decoders and GS losses, geometry/kappa/axis losses, post-hoc stretches or li remaps toward direct predictions, temporal
smoothing, and additional input families (Thomson dynamics, long-horizon future, current-diffusion simulations, fast fluctuation power).
Beyond about 30-40 members, more members do not help, because the members share their errors.
