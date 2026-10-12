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
- **HLVF** (24 seeds): history + vessel + future features, with seven extra outputs that predict the seven consistency scalars of the
  label maps (computed with the scorer's own code), trained jointly at a small weight (0.05 per entry; the q95/betaN loss unchanged).
  The extra heads leave the flux map unchanged within seed noise (5 paired seeds on both development folds), but their mean is a
  better estimate of the axis height and the elongation than the separately trained direct regressors;
- **HLAV** (40 seeds): the no-history recipe + vessel features, with the same seven heads.

## Ensemble and post-processing
The submission is the equal-weight mean of the 64 members. On frames with almost no usable Thomson data, only the history members
are averaged, because the no-history members degrade there. The blend's POD modes 129-512 are filled by a closure network that maps the
leading 64 coefficients to the higher modes (trained on training flux). Finally, the magnetic axis is moved inside the separatrix only
(weight clip((psi - psi_b)/(psi_axis - psi_b), 0, 1)), vertically, 75% of the way toward the mean of the 64 members' head axis height and the
axis height predicted by six direct regressors trained on the same inputs; the separatrix is untouched and the shift is skipped on
Thomson-poor frames.

Next, the elongation is corrected. Each member's own map gives its own kappa (the scorer's definition); the mean of these member kappas
is a better kappa estimate than the kappa of the averaged map, because averaging maps blurs the boundary shape. The target is the mean
of this member-mean kappa, the 64 members' head kappa and the direct regressors' kappa, applied only on frames where all three agree on the
direction of the change (about 54% of frames; Thomson-poor frames never). The map is compressed or expanded radially about the magnetic axis, inside the plasma
(weight 1 inside the separatrix, smoothly 0 by psi_N = 1.4) and only near the heights of the separatrix's inboard and outboard extremes (Gaussian
weight in Z, 0.25 m, each side on its own side of the axis): kappa is read from the four extreme points, so compressing the whole
cross-section (the earlier variant) moved the volume for nothing; localising the compression keeps the kappa gain and most of the volume, with the factor found per frame by root-finding so that the kappa read off
the map moves half-way to the target; the realised value of the written float32 map is re-checked, because the separatrix extraction
can jump near the X-point, and frames without an admissible solution are left unchanged. Changing the width rather than the height
matters: vertical stretches move the X-point and worsen the separatrix distance, radial compression does not.

Last, the internal inductance is adjusted without moving the separatrix or the axis. Inside the separatrix the normalised flux
s = (psi_axis - psi)/(psi_axis - psi_b) is remapped as h(s) = s + a s (1 - s)^2, which keeps the axis value, the separatrix level and
the edge gradient (h(1) = 1, h'(1) = 1). For each frame, `a` is found by root-finding so that li computed from the remapped map
(the scorer's li(2) definition: volume-averaged Bp^2 over the squared mean boundary Bp) moves 35% of the way from the map's own li
toward the direct regressors' li, measured from the li before the elongation step. Thomson-poor frames are skipped. The adjustment changes the map by about 0.25% of its spread.
Earlier open-loop corrections (fixed remap or stretch amplitudes) did not help because the realised change tracked the intended one
poorly; solving for the official li per frame removes that error.

## Results
Development (held-out shots, official scorer, train-only analogues of the members): S = 0.99684 on the native-frame fold and 0.99450
on the legacy fold for a 43-member structure. Public leaderboard (all-labelled members): 0.996271 and 0.996275 for two independent
43-member draws, 0.996283 for both draws together (86 members), and 0.996345 for the 86-member blend with the li adjustment (li R^2
0.9675 -> 0.9698; development: +0.000025 native, +0.000011 legacy at strength 0.35). The submitted 64-member structure scored 0.996330
without and 0.996379 with the elongation step (kappa R^2 0.9800 -> 0.9819, volume -0.0005; development: +0.000039 native, +0.000049
legacy). Replacing the 24 head-less history members by the HLVF members and using their heads in the axis and elongation targets
scored 0.996400 (+0.000021: kappa R^2 +0.00025, li +0.00017, Z_axis +0.00004, q95 +0.00007; development with train-only analogues:
axis shift +0.000013 native / +0.00011 legacy, elongation +0.000004 / +0.000003). Giving the no-history members the heads as well
(HLAV; head targets from all 64 members) scored 0.996423 (+0.000023: Z_axis +0.00021, li +0.00021, volume +0.00019, q95 +0.00008,
betaN +0.00004, kappa -0.00009). Localising the elongation step at the width extremes (submitted configuration) scored 0.996439
(+0.000016: volume R^2 +0.00041, kappa +0.00015, lower triangularity +0.00003; development: native +0.000077 vs +0.000064, legacy
+0.000064 vs +0.000060 for the whole-plasma step). Measured contributions on the public leaderboard:
vessel features -0.00032 D_LCFS and +0.00017 consistency; future features and the axis shift +0.00006 to +0.00008 S; li adjustment
+0.00006 S; elongation adjustment +0.00005 S (+0.00002 more when localised); consistency-scalar heads +0.00004 S (history members +0.00002, no-history members +0.00002).

## Compute
Training is about 15 A100-minutes per member when four members share one GPU. A complete from-scratch run of the 64-member entry
(members, regressors and closure) took 16.7 hours on one A100 80GB PCIe, within the 24 GPU-hour limit (the heads add seven outputs to a
1.2M-parameter network and no measurable time); a 43-member run took 8.9 H100-hours. CPU preparation takes about 4-6 h; the post-processing (closure, axis shift, per-member kappa of
all 64 members, elongation and li adjustments) about 5 h on 10 cores.

## What did not help
Things we tested on the official scorer that did not help: larger output rank, U-Net and transformer members, Green's-function/GS
physics decoders and GS losses, geometry/kappa/axis losses, open-loop post-hoc stretches, vertical (instead of radial) closed-loop
stretches, closed-loop triangularity corrections (ill-conditioned: the separatrix level moves instead of the vertex), member reweighting
or trimming, temporal
smoothing, li heads on the members' hidden features, consistency-scalar heads at a large loss weight (they then degrade the flux map),
physics q95 from the predicted map's flux surfaces, a stronger li adjustment (0.5 instead of 0.35: public -0.000007), local
warps of the separatrix corners toward member-mean X-point positions (the extraction amplifies edits near the X-point), learned per-frame combinations of the scalar estimates, and additional input families (Thomson dynamics, long-horizon future,
current-diffusion simulations, fast fluctuation power). More members help only slowly (43 -> 86: +0.000008 public), because the
members share their errors; 64 members fit the GPU budget with margin.
