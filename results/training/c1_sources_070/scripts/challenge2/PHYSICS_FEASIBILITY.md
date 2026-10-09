# Challenge 2: first MAST physics prototype

**The first viable direction is a fixed-coil free-boundary Grad–Shafranov solver,
with learned dimensionless profile priors from DIII-D and public synthetic
equilibria. The current prototype is useful for physics checks, but is not ready
for a full submission.** No MAST labels, starter demo labels, hidden references,
participant predictions, or audit labels were read.

## Contract and input conventions

Challenge 2 ranks the uncapped ratio `S_MAST / S_DIII-D` when `S_DIII-D >= 0.85`.
Both machines must be present. Each machine uses the same composite:
`0.55 max(0,R²ψ) + 0.15 max(0,mean(R²q95,R²betaN)) + 0.10(1-min(1,D_LCFS))
+ 0.20 mean(max(0,R²derived))`. There are seven derived quantities: axis R/Z,
elongation, upper/lower triangularity, volume, and li. A single global flux sign
is selected per machine; amplitude and additive gauge are not normalized.
The current authoritative implementation is scorer 3.2.0 in the unmodified
[starter scorer](../../starter/fusion_scoring/common.py).
The [official rules](https://fusion-equilibrium-challenge.sophelio.io/rules/)
specify whole-shot splits, external-resource disclosure, and hidden-label protection.

There are 1,206 MAST public-input shots and no MAST training split. Output is
native `(T,65,65)` flux in Z-major/R-minor order, plus `(T,)` q95 and betaN, at
every original `efit_times`. R is approximately 0.06–2.0 m and Z −2–2 m; both are
physical coordinates from each row. Flux is Wb/rad. MAST's stored axis is a
maximum, whereas DIII-D's is a minimum. Times are milliseconds. Only inputs
may be interpolated. All supplied target frames are diverted.
See the [official data description](https://fusion-equilibrium-challenge.sophelio.io/data/).

The row contains 812 individual PF conductor rectangles: 656 solenoid turns;
20 turns each P2L/U; 8 each P3L/U; 23 each P4L/U and P5L/U; 4 each P6L/U.
Each P signal is already **total kA-turn**, so an element receives `1000 I / N`
amperes. The solenoid is **feed kA**, so every solenoid element receives `1000 I`.
The supplied current signs are retained. Geometry joins on `coil_input_column`.
TF and EFPS have no supplied PF rectangle. DIII-D's lumped F-coil currents are
also kA-turn; its ECOILA convention is explicitly unresolved in the starter.
These details come from the version-pinned [starter signal dictionary](../../starter/README.md).

MAST's core Thomson chord is horizontal, with calibrated R and Z=0. Drop ghost
data channel zero using `align_mast_thomson_core` before joining 131 data columns
to 130 radii. Te is eV, ne is m⁻³, so `p_e = e ne Te` in pascals. Early-campaign
currents contain NaNs on a union time axis: finite samples must be masked for
each signal separately. The prototype records when nearest finite TS endpoint
values are extrapolated beyond that channel's observed time interval.

## Implemented calculation

[fusion/mast_physics.py](../../fusion/mast_physics.py) reads an explicit input
allowlist. It integrates each rectangular conductor using 2×2 Gauss quadrature
and the circular-filament Green function for `ψ=R Aφ`. This sets a free-space
gauge with ψ→0 at R→0 and infinity. That physical gauge has **not** been shown
to match the hidden MAST EFIT gauge.

The plasma field solves

```text
Δ*ψ_plasma = −μ0 R jφ,       Δ* = ∂RR − R⁻¹∂R + ∂ZZ,
ψ_total = ψ_coils + ψ_plasma.
```

The native-grid sparse solve uses a free-space Green integral of the plasma
source on the computational boundary. Coils stay fixed. The initial elliptical
current support comes from the measured TS pressure center/width, with an
explicit elongation prior 1.8. The nonlinear closure is

```text
s = (ψ − ψ_edge)/(ψ_axis − ψ_edge), inside the extracted plasma surface,
p(s) = p_axis s^(α+1),
jφ = [R p_axis(α+1)/(ψ_axis−ψ_edge) + A/R] s^α,
A chosen so ∫jφ dR dZ = measured Ip.
```

The first experiment fixes α=1 and `p_axis=2×90th-percentile measured p_e`,
explicitly assuming equal ion/electron pressure and no fast-ion pressure.
The scorer's public envelope assists contour extraction; its output is **not
independent proof of a diverted X-point separatrix**. Counter-current closure
requirements and missing surfaces are reported rather than clipped into success.
The q95 and betaN calculations are approximate consequences of this assumed
pressure/FF′ closure. TF uses 24 effective turns, supported by the original MAST
central-rod description in this [York thesis](https://etheses.whiterose.ac.uk/id/eprint/14209/1/516649.pdf).

## Measured checks and failures

[Analytic checks](../../results/challenge2/physics_prototype_v2/analytic_checks.json)
compare the Green function against independently integrated Biot–Savart:
relative error **1.50×10⁻¹⁶**; reciprocity error zero. The synthetic sparse GS
residual is **8.39×10⁻¹⁴**, the independent boundary integral agrees, and the
600 kA source integrates exactly to its specified current.

[Public-input experiment](../../results/challenge2/physics_prototype_v2/summary.json):
four deterministic shots `[0,401,803,1205]`, three native frames per shot, 200
Picard steps at relaxation 0.2. **5/12 frames converge** to relative profile
residual below 0.002; **10/12 retain closed surfaces**; 2 lose their plasma
surface. Every returned flux is finite float32 on the native grid. Linear solve
residuals remain around 10⁻¹³ and Ip stays exact. These are numerical/physics
diagnostics, **not MAST accuracy estimates**. The observed drift motivates a
Newton–Krylov solver and temporal continuation before any full 1,206-shot run.
See [flux illustration](../../results/challenge2/physics_prototype_v2/prototype_flux.png).

[DIII-D convention check](../../results/challenge2/d3d_physics_conventions_v2/summary.json)
uses eight deterministic **training-only** shots with alternating coefficient
fit/check subsets. Differentiating the training target and integrating the
source inside its training LCFS gives median `Ip_from_flux/Ip_measured=0.998153`.
This independently supports the per-radian amplitude convention; no extra 2π
factor is needed. This diagnostic deliberately uses target-derived plasma
sources and is **not a deployable model or a validation score**.

The initial 0.096 Wb/rad exterior discrepancy was largely caused by mixing
opposite current orientations. The target is expressed with a fixed maximum
axis, so signed coil currents must first be normalized by `sign(Ip)`. Root's
[controlled orientation and solenoid-rectangle check](../../results/challenge2/d3d_physics_conventions_oriented_v3/summary.json)
reduces median exterior RMSE to **0.009135 Wb/rad** (coefficient-check subset
0.00833), with global PF response factor **0.93453**. One training input has an
Ip clock/value anomaly, so the median magnitude agreement is not universal.
The unresolved ECOILA/ECOILB response and required frame gauge offsets remain
uncalibrated; none of these fitted constants is transferred to MAST.

The [complete MAST input inventory](../../results/challenge2/input_inventory_v1/summary.json)
confirms all **69,075 native frames** across 1,206 shots have observed positive
Ip (81.314–1188.767 kA), one physical grid and one conductor geometry. Thus the
prototype's positive-Ip orientation applies throughout the released MAST fold.

## Nonlinear solver update

The [fixed-coil comparison](../../results/challenge2/solver_comparison_v1/README.md)
improves the same 12-frame convergence count from **5 to 9** using either
Anderson or Newton–Krylov. Both preserve closed surfaces and finite scalar
outputs on all 12 frames, although only 5 pass all independent structural checks
at the initial 0.002 residual tolerance. Temporal continuation converges on all
eight native frames in the short segment and substantially reduces evaluation cost.

The tighter [native continuation proof](../../results/challenge2/solver_strict_continuation_v1/README.md)
then converges on **8/8 frames at 1e−4**, with independent saddle/separatrix,
current-integral, positivity and scalar checks. The looser tolerance sometimes
admitted materially different fields and should not be used for final acceptance.
This establishes a numerical solver on one short segment, not experimental
accuracy, general full-shot coverage or unique equilibrium identification.

## Recommended next model

| Approach | Feasibility and role |
|---|---|
| DIII-D output-basis or coil-name remapping | Poor first choice: DIII-D POD lives on a different physical domain; coil count/geometry and TS orientation change. Dimensionless rescaling alone cannot fix those differences. |
| Physics-informed GS reconstruction | Best first path: analytic actual-coil field, measured Ip, calibrated TS coordinates, pressure/FF′ closure priors, and a fixed-coil Newton–Krylov solve. Learn unknown closures and residual response, not known coil geometry. |
| Synthetic equilibria/operator transfer | Best scalable follow-up: solve diverse **actual MAST geometry** equilibria from public inputs and physically bounded profile priors, then train an operator with the vacuum field and nondimensional TS/Ip inputs. Synthetic labels must be disclosed and remain distinct from experimental labels. |

Use `R/a`, `Z/a`, `ψ/(μ0 Ip a)`, `I_coil_turns/Ip`, and pressure normalized by
`μ0 Ip²/a²` to represent cross-machine physics. First infer dimensionless p′/FF′
profile priors from DIII-D training targets and allowed TS/current inputs. A
forward solve then uses actual target-machine geometry. The proposed synthetic
operator should learn source/profile corrections or solver updates, with
boundary, current integral and GS residual checks. Scalar heads must account
for F(ψ) and total pressure rather than treating ψ alone as sufficient.

Static solvers have been validated against EFIT++ when the equilibrium
constraints are known, but MAST-U geometry must not be substituted for MAST:
[solver validation paper](https://arxiv.org/abs/2407.12432),
[FreeGSNKE static solver](https://docs.freegsnke.com/api/freegsnke.gsstaticsolver).
Multiple valid equilibria can exist for one free-boundary problem, as shown in
this [MAST-U nonuniqueness study](https://arxiv.org/abs/2503.05674).

Identifiability remains limited: Ip supplies one current integral, TS supplies
electron pressure on a chord, and the missing ion/fast-ion pressure, FF′ profile,
passive conductor currents and absolute flux reference are not uniquely fixed.
A small GS residual alone cannot resolve those missing constraints. Promote a
candidate only after actual-separatrix checks, fixed-coil convergence across
the public-input regimes, current closure, grid refinement, TS pressure
consistency, complete native output coverage and reproducible provenance.

Reproduce with `.venv/bin/python scripts/challenge2/run_mast_prototype.py
--shots 4 --frames 3 --iterations 200 --output <new-directory>` under one allocated
CPU; use `check_d3d_physics_conventions.py --shots 8 --output <new-directory>` for
the independent training-only convention diagnostic. No upload code is included.
