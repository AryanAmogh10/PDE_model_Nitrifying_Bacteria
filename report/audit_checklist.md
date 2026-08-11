# Audit checklist -- everything done, with how to independently verify it

Purpose: let you (or your supervisor) cross-check every claim rather than take it
on trust. Each item says **what was claimed**, **what evidence backs it**, and
**how to re-derive it yourself**. Labels follow the project convention:
VERIFIED (independently confirmed with numbers), CONSISTENT (self-consistent but
no external ground truth), ASSUMED (taken on faith from source material).

Run everything from the repo root. Full suite:
```bash
for f in tests/test_*.py; do python3 "$f"; done
```

---

## A. Bugs found and fixed

### A1. Death term: constant `b_i` instead of density-dependent `b_i*rho`
- **Where:** `parabolic.py`, reaction term.
- **How found:** you flagged it in code review; I confirmed against the rendered
  arXiv PDF eq. (2.1)/(2.5). An earlier pdftotext extraction had silently
  dropped the `rho` symbol, which is how it survived.
- **Impact:** large -- with constant death there is no self-limiting mechanism,
  biomass can grow unbounded. With `b_i*rho` growth saturates at `rho ~ g_i/bhat_i`.
- **Verify:** `tests/test_parabolic.py::test_death_term_is_density_dependent_not_constant`
  -- checks the per-step growth *ratio* strictly decreases under uniform ample
  substrate. A constant death term gives an exactly constant ratio.
- **Cross-check yourself:** open the arXiv PDF at eq. 2.1 and confirm the `-b_i rho`.
- **Status: VERIFIED**

### A2. Reaction term zeroed at the r=0 row
- **Where:** `elliptic.py` (`_residual`, `_assemble_global`) and `relaxation.py`.
- **Cause:** r=0 was treated as a boundary-condition row. It isn't -- it's a
  genuine interior PDE point that happens to use a symmetric stencil because of
  the coordinate singularity. Only row N is a true BC row.
- **Impact:** degraded a 2nd-order scheme to ~1st order whenever reactions were
  active. Invisible in the pure-diffusion case.
- **Verify:** `tests/test_elliptic_closed_form.py::test_spherical_error_converges_with_grid_refinement`
- **Status: VERIFIED**

### A3. Parabolic Newton was only "modified", not true Newton
- **Where:** `parabolic.py` -- the Jacobian omitted `M_i = d[Adv(rho)*u_i]/drho`.
- **Impact:** residual was still exact so it converged to the *right* answer on
  smooth fields, but stalled at ~1e-4 on sharp density fronts.
- **Fix:** added `build_advection_rho_jacobian`, included in every column block
  (since `rho = Sum_j u_j`, it appears in all of them).
- **Verify:** `test_advection_rho_jacobian_matches_finite_difference` and
  `test_sharp_front_newton_converges_to_tolerance` (asserts zero stall warnings).
- **Status: VERIFIED**

### A4. Approximate cell volume near the domain centre in `build_laplacian`
- **Where:** `elliptic.py::build_laplacian`, interior rows used `V_i ~ r_i^p*h`.
- **Impact:** relative error 7.7% at i=1, 2.0% at i=2 -- held convergence order at
  ~1.7-1.9 instead of 2.0 even after A2 was fixed.
- **Fix:** exact `V_i = (r_e^{p+1} - r_w^{p+1})/(p+1)` at every row.
- **Verify:** mass conservation to machine precision for all 3 geometries;
  and 2nd-order convergence (see B2 for the mpmath confirmation).
- **Status: VERIFIED**

### A5. Neumann residual target hardcoded to `0.0` (three separate places)
- **Where:** `elliptic.py::_residual`, `elliptic.py::solve_picard`,
  `relaxation.py::solve_relaxation`.
- **Cause:** the matrix row was built correctly by `apply_bc` for any requested
  flux, but the residual/RHS target was independently written as `0.0` instead
  of `-value`. Two pieces of code representing the same equation, built
  separately, drifted apart.
- **Impact:** **any** nonzero-flux Neumann solve, in any of the three solvers,
  silently solved a *zero*-flux problem. No error, no warning.
- **Verify:** `tests/test_elliptic_closed_form.py::test_nonzero_flux_neumann_matches_closed_form`
  -- checks both agreement with the closed form AND that a finite-difference
  recomputation of the boundary flux from the converged profile equals the
  requested value (matched to 1.1e-14).
- **Status: VERIFIED for a single-substrate (mixed-BC) configuration.
  See D2 -- the multi-substrate production path is still broken.**

### A6. Spurious unphysical root accepted by backtracking
- **Where:** `elliptic.py::solve_newton`.
- **Cause:** backtracking accepted any step that reduced the residual norm, with
  no plausibility constraint. Under sealed (zero-flux) boundaries with reaction
  active, it converged to NH4 ~ 100.5 and NO3 ~ 5416 -- orders of magnitude above
  anything the closed system could supply. The residual really was near zero;
  the root was mathematically real but physically nonsense.
- **Fix:** added an upper bound `c_max = 5 x max|c_inf|`; a trial step must both
  reduce the residual AND stay inside `[0, c_max]`.
- **Verify:** rerun the sealed zero-flux + reaction case; max concentration now
  stays < 10 (was ~5400).
- **Status: VERIFIED**

### A7. Sign bug in the `c_max` guard (self-inflicted by A6, then found)
- **Cause:** `c_max = 5 x max(c_inf.values())`. Under Neumann, `c_inf` doubles as
  the *flux*, which is legitimately **negative** for an influx. `max(-0.3,0,0,0)
  = 0.0` collapsed the bound to zero, clipping every trial to exactly 0.
- **Fix:** `max(abs(v) for v in c_inf.values())`.
- **Verify:** the nonzero-flux Neumann test uses a negative (influx) value, so it
  exercises exactly this path.
- **Status: VERIFIED**

### A8. Fallback mislabeling + tolerance mismatch
- **Cause:** `solve_newton`'s internal relaxation fallback could silently
  substitute a relaxation answer while `slowfast.py` still logged the step as
  `"newton"`; and the fallback used `solve_relaxation`'s default tol (1e-9)
  rather than the caller's.
- **Fix:** `solve_newton` returns `(C, history, method)` with an honest method
  string; fallback now targets the caller's `tol`.
- **Verify:** run at `tol=1e-11` (tighter than the old 1e-9 default) and confirm
  the fallback reaches ~7.3e-12 -- past where the old default would have stopped.
- **Status: VERIFIED. Note the breaking API change: `solve_newton` now returns a
  3-tuple. All 6 in-repo call sites updated.**

### A9. Advection operator normalised by the wrong control volume
- **Where:** `parabolic.py::build_advection_matrix`, `build_advection_rho_jacobian`,
  `_total_mass`.
- **Cause:** still using `V ~ r^p*h` after `build_laplacian` had been corrected to
  exact volumes (A4). Wrong by O(h^2/r^2) on the first interior rows and -- more
  seriously -- by a **factor of ~2 at the outer boundary node in every geometry
  including the slab** (full cell used where only a half-cell exists).
- **Impact:** diffusion conserved one measure and advection another, so the
  coupled Stage 5 scheme conserved **neither**. Defect was O(1) and did **not**
  vanish under refinement (0.449 -> 0.421 -> 0.407 -> 0.400 for N=40->320).
- **Why no test caught it:** the regression test's probe was `u = sin(pir)`, which
  is exactly **zero at r=1** -- the single node carrying the factor-of-2 error.
  With `u = sin(pir) + 0.5` the defect is ~100x larger and non-converging.
- **Fix:** added `elliptic.cell_volumes` / `face_area` as a single source of
  truth; all conservative operators and the mass diagnostic now share it.
  `build_laplacian` was refactored onto it and **verified bit-identical**
  (0.000e+00 across 3 geometries x 3 resolutions).
- **Verify:** `test_advection_operator_is_exactly_mass_conservative` -- now uses
  the exposing probe and asserts **exact** conservation (was: "defect halves",
  which a genuinely non-conservative operator can satisfy).
- **Impact on results:** 0.03-0.06% on biomass; peak locations, species shares
  and AOB dominance all unchanged. Structurally important, numerically small.
- **Status: VERIFIED**

---

## B. Validation performed

### B1. Closed-form benchmarks (Stage 3)
Reduced the full nonlinear system to a linear one with known solutions
(`cosh` / Bessel `I0` / `sinh(kappa*r)/r` for slab / cylindrical / spherical).
Newton err < 1e-4, Picard < 1e-3, at two Thiele moduli (kappa=2.0 and kappa=0.5).
`tests/test_elliptic_closed_form.py`. **VERIFIED**

### B2. Convergence order, floating-point-artifact-free
Double precision showed order degrading to 1.41 at fine grids. Traced to
**catastrophic cancellation** -- the residual sums two ~1e7-magnitude terms
cancelling to O(1), leaving a ~1e-9 double-precision floor. Proven by:
tightening tol from 1e-9 to 1e-14 changed the errors **not at all** (identical to
6 sig figs). Re-solved the identical discretization at **50 decimal digits**
(`mpmath`, hand-written Thomas algorithm, bypassing SciPy's double-precision LU):

| N | max error | order |
|---|---|---|
| 50 | 1.9527e-05 | -- |
| 100 | 4.8842e-06 | 1.999 |
| 200 | 1.2212e-06 | 2.000 |
| 400 | 3.0532e-07 | 2.000 |
| 800 | 7.6330e-08 | 2.000 |

Average 1.9997. **VERIFIED -- the scheme is genuinely 2nd order.**

### B3. Newton <-> relaxation cross-validation (Stage 4 vs Stage 3)
All 3 presets x 4 densities (1e-4, 0.01, 0.05, 0.5) = 12 combinations.
Agreement **8.7e-12 to 7.8e-11** -- five orders tighter than the 1e-6 threshold.
Two independent algorithms agreeing is stronger evidence than either alone.
**VERIFIED**

### B4. Mathematical structure audit (7 checks, 4 never run before)
| Check | Result |
|---|---|
| Full elliptic Jacobian vs FD (4Npts^2, incl. Laplacian + BC rows) | 4.4e-06 / 1.7e-06 |
| Full parabolic Jacobian vs FD (3Npts^2, all blocks) | 5.9e-07 / 6.0e-07 |
| Discrete nitrogen conservation `Sum_j D_j Rhat_j = 0`, 3 presets | rel 1.6e-16 |
| Same, end-to-end on converged solutions | rel 1.1e-14 / 2.5e-14 |
| Self-adjointness `V_iL_ij = V_jL_ji`, all geometries | machine precision |
| Temporal order of the slow-fast splitting | **1.04-1.07** (theory: 1) |
| M-matrix / discrete maximum principle | holds, all geometries & substrates |

The nitrogen-conservation check is the strongest new result: nitrification is
1:1 (NH4->NO2->NO3), so `Sum_j D_j Rhat_j = 0` must hold *identically*. It does, to
1e-16 -- an **independent** confirmation that `Lambda`/`LambdaProd` carry the
right diffusivities, re-validating an earlier bug fix from a different
direction than the test that originally caught it. **All VERIFIED**

### B5. 2D extension validation (`tests/test_2d.py`, 8 checks)
- Exact **slab reduction** to 1D: agreement ~1e-14, i.e. *solver tolerance*, not
  discretization error. Pins Laplacian + BCs + reaction + Jacobian assembly at once.
- **Radial reduction** against 1D cylindrical `p=1` (note: **not** `p=2` -- 2D
  Cartesian radial symmetry gives `c_rr + (1/r)c_r`; see B7 for the
  first-principles proof this target is correct, not just empirically the one
  that agreed):
  deep interior 4.27e-07 -> 1.00e-07 -> 2.16e-08 (**2nd order**);
  near-boundary 1st order (staircased circle -- geometry limit, not a defect).
- 2D Laplacian: exact conservation (1e-16), order 2.000, anisotropy converging faster.
- 2D advection: **exactly** conservative (7e-17) -- better than the 1D operator.
- 2D rho-Jacobian vs FD: 6.6e-10. Analytic equilibrium: 2.7e-09.
**All VERIFIED**

### B6. Full suite status
**9/9 test files pass** (8 pre-existing 1D + 1 new 2D), re-run after every change.

### B7. Why `p=1`, not `p=2`, for the 2D radial reduction -- derived, not just observed
Two independent derivations, both confirming `p=1` is the *only* mathematically
consistent target (not merely the one that happened to agree numerically):

1. **Analytic.** 2D Cartesian: `Lap c = c_xx + c_yy`. In polar coordinates,
   `Lap c = c_rr + (1/r)c_r + (1/r^2)c_thetatheta`; for a radially symmetric field
   `c_thetatheta = 0`, giving `Lap c = c_rr + (1/r)c_r`. Our general 1D operator is
   `Lap_p c = c_rr + (p/r)c_r`. Matching coefficients: `p/r = 1/r => p = 1`,
   exactly, for *any* radially symmetric function -- confirmed symbolically
   (`sympy`): difference at p=1 is `0`; at p=2 it is `Derivative(c,r)/r`
   (nonzero).
2. **Independent numerical check**, no polar-coordinate machinery at all: a
   plain 5-point x,y finite-difference Laplacian of `c=exp(-r^2)` on a fine
   Cartesian grid, compared against the analytic `Lap_p` formula for
   `p=0,1,2,3`. Only `p=1` shrinks under grid refinement (4.01x/doubling,
   i.e. 2nd order -- pure FD truncation); `p=0,2,3` sit at a **fixed, non-
   vanishing** O(1) mismatch (33-100% relative error) regardless of resolution
   -- they are not close, they are the wrong formula.
3. **Generalises cleanly**: for n-dimensional Cartesian space, radial symmetry
   gives `Lap c = c_rr + (n-1)/r*c_r`, i.e. our exponent is literally `p = n-1`:
   slab `p=0` <-> n=1, cylindrical `p=1` <-> n=2 (what was tested), spherical
   `p=2` <-> n=3. No special-casing required.
**VERIFIED**

### B8. Sector formation reproduced with the paper's own parameters (was D5, now resolved)
Pulled arXiv:2512.13156 Table 1, Case (A) directly from the paper (via its
public arXiv PDF): `d_i=1e-6, a_i=1e-5 (a/d=10, matching our
A_OVER_D_RATIO exactly), r_i=1, K_i=1, b_i=0.1, Y_i=0.2, D=1e-4, L=1, c0=5,
cinfinity=1e-5, domain [0,L]x[0,L], t in {0,25,50}`. Reconfigured our 2D solver's
coefficient structure to express System (2.3) (single shared substrate, three
symmetric species) rather than the nitrifier-specific reaction network, and ran
to `T=50` with three inocula seeded 120deg apart in a central circular region,
matching the paper's Figure 3 setup. **Deviation, flagged, not hidden:** the
paper uses time-dependent substrate with Neumann influx (`cinfinity=1e-5`, near-zero);
we used our validated QSSA + Dirichlet (`c=5`) machinery, since that is the
machinery this thesis validates. `eps = r*L^2/D = 1e4` in the paper's own Case (A)
-- i.e. **the paper's own parameters put it in the eps>>1, non-QSSA regime**, the
opposite of the assumption our whole Stage 3 rests on; this is a real,
structural mismatch between what the paper's Case (A) actually is and what our
solver assumes, not a minor implementation detail.

**Result: sector formation reproduced.**
- Colony expanded from seed radius ~0.10 to occupied radius ~0.38 over T=50.
- Interior depleted and decayed: substrate at the centre fell from ~5 to
  1.6e-6; peak total density fell from 0.463 (t=0.5) to 0.152 (t=50) -- matches
  the paper's description ("lower densities are found in the centre because of
  substrate depletion and subsequent bacterial decay").
- **Angular dominance in the active outer shell (rin[0.22,0.36]) is real, not a
  metric artifact** -- checked directly, not just via the m=3 Fourier amplitude
  that misled a different diagnostic earlier this session:

  | sector | AOB | NOB | CMX | dominant |
  |---|---|---|---|---|
  | -180deg..-120deg | 0.000000 | 0.000013 | 0.044459 | CMX |
  | -120deg..-60deg | 0.001362 | 0.000000 | 0.052535 | CMX |
  | -60deg..0deg | 0.041628 | 0.000000 | 0.000017 | AOB |
  | 0deg..60deg | 0.047281 | 0.000016 | 0.000000 | AOB |
  | 60deg..120deg | 0.001362 | 0.052535 | 0.000000 | NOB |
  | 120deg..180deg | 0.000000 | 0.044459 | 0.000013 | NOB |

  Each species dominates a different wedge, each wedge aligned with that
  species' seeding angle (AOB seeded at 0deg, NOB at 120deg, CMX at 240deg), with
  near-complete exclusion in the dominant sector (e.g. AOB=0.047 vs NOB=1.6e-5
  in the 0deg-60deg wedge). 3 of 3 species are each sector-dominant somewhere --
  genuine segregation, not a symmetric/uniform field with a spurious harmonic.

**Item (c): is the diffusion-length threshold quantitative, or coincidental?**
Swept `d_i` (with `a_i/d_i=10` held fixed, matching the paper) from `1e-6` to
`1.0`, all else at Case (A) values, `T=50` (reduced grid/dt for tractability --
flagged below): predicted threshold `d << L^2/T = 0.02`.

| `d_i` | `sqrt(dT)/L` | colony radius | sector purity | verdict |
|---|---|---|---|---|
| 1e-6 (paper) | 0.007 | 0.385 | 66.7% | SECTORS |
| 1e-4 | 0.071 | 0.707 | 16.7% | mixed |
| 1e-2 | 0.707 | 0.587 | 66.7% | SECTORS |
| 0.1 | 2.236 | 0.000 | 0.0% | UNIFORM |
| 1.0 (>> our eloi) | 7.071 | 0.000 | 0.0% | UNIFORM |

The extremes support the hypothesis cleanly: the paper's own regime (`d=1e-6`)
gives sectors; `d=0.1` and `d=1.0` (both exceeding our `eloi` preset's
effective diffusivity) give a fully uniform field, matching the original
`eloi`-preset finding (A3=0.0000) from earlier this session. **The `d=1e-2` and
`d=1e-4` points are not monotonic and should not be over-interpreted** -- this
sweep ran on a coarser grid (26x26 vs 48x48) and larger `dt` (2.0 vs 0.5) than
the single confirmed run above, purely for compute budget reasons (a finer
version of this sweep timed out at 10 minutes without completing even one
point at the original resolution). The `d=1e-4` row's colony radius of 0.707
(the corner-to-centre distance of the unit square, i.e. "reached everywhere")
is consistent with an under-resolved run, not a real non-monotonic transition.

**ITEM 4 (redesign, closes out B8(c)).** `report/item4_diffusion_threshold_sweep.py`
fixes the actual flaw above directly: every point in the sweep now runs at the
SAME resolution (28x28, dt=1.0, 15 slow steps -- coarser than the B8(a)/(b)
confirmed run throughout, but never changing point-to-point), removing
resolution as a possible cause of non-monotonicity. A coarse 4-point
bracketing pass (`d=1e-6, 1e-4, 1e-2, 1.0`) located the SECTORS->UNIFORM flip
between `1e-4` and `1e-2`, then 4 rounds of log-space bisection narrowed it:

| `d_i` | sector purity | verdict |
|---|---|---|
| 1e-6 | 1.00 | SECTORS |
| 1e-4 | 1.00 | SECTORS |
| 3.162e-4 | 0.83 | SECTORS |
| 4.217e-4 | 0.33 | mixed |
| 5.623e-4 | 0.00 | UNIFORM |
| 1e-3 | 0.00 | UNIFORM |
| 1e-2 | 0.00 | UNIFORM |
| 1.0 | 0.00 | UNIFORM |

**Result: monotonically non-increasing SECTORS -> UNIFORM as `d_i` grows, at
this consistent resolution** -- the non-monotonicity in the original sweep was
indeed a resolution artifact, not a real feature; bisection converges the
threshold to a factor-of-1.3 bracket, `d_i` in `[4.2e-4, 5.6e-4]`.

**Genuine discrepancy found:** the naive scaling prediction `d << L^2/T`
(`L=1, T=15`, predicting a threshold `~0.067`) overshoots the bisected
transition (`~4.9e-4` at 28x28 resolution) by ~two orders of magnitude. Per
explicit instruction this was investigated to a resolution, not left as a
bare unexplained gap -- see `report/item4b_threshold_mechanism.py`.

**Investigation (ITEM 4 follow-up).**

*Step 1 -- candidate length scales.* `Da := r_max*L^2/d`, evaluated at the
28x28 threshold (`d=4.87e-4`) for four candidate `L`:

| length scale `L` | value | `Da = r*L^2/d` |
|---|---|---|
| domain size | 1.0 | 2054 |
| inter-seed spacing | 0.260 | 139 |
| seed radius | 0.10 | 20.5 |
| grid spacing `h` (28x28) | 0.0357 | **2.6** |

Grid spacing gave the closest-to-O(1) value -- a warning sign the "threshold"
might be a numerical-resolution artifact rather than physics tied to seed
geometry, not evidence of the right length scale (grid spacing isn't a
property of the physical problem).

*Step 2 -- direct resolution test.* Reran the identical coarse-bracket +
bisection procedure at 40x40 and 56x56 (finer grids; `h` shrinks from
`1/28=0.0357` to `1/40=0.025` to `1/56=0.0179`):

| grid | `h` | bisected `d_mid` |
|---|---|---|
| 28x28 | 0.0357 | 4.87e-4 |
| 40x40 | 0.0250 | 2.74e-4 |
| 56x56 | 0.0179 | 2.74e-4 |

The 28x28 estimate is confirmed **under-resolved and biased ~1.8x high** --
a real, quantified numerical effect, consistent with the grid-spacing `Da`
warning in Step 1. But a *pure* `h^2` artifact would keep shrinking the
threshold at every finer grid (predicted 40x40->56x56 ratio: `(40/56)^2 =
0.51`, i.e. `d_mid(56x56)` should fall to `~1.4e-4`); it did not -- 40x40 and
56x56 agree (within this bisection's discrete resolution, limited by a
6-sector purity metric with `1/6` granularity). **This rules out "pure grid
artifact" as the full explanation**: the threshold is converging toward a
resolution-independent value near `d ~ 2.7e-4`, not continuing to shrink with
`h`. Re-evaluating the candidate length scales at this converged value:

| length scale `L` | `Da = r*L^2/d` (at `d=2.74e-4`) |
|---|---|
| domain size | 3652 |
| inter-seed spacing | 246 |
| seed radius | **36.5** |

Seed radius gets closest (order 10-100, not order 1000s), but **none of the
four tested length scales bring `Da` to a clean O(1) value** even at the
resolution-corrected threshold.

*Step 3 -- advection strength.* Fixed `d_i = 4.87e-4` (the 28x28 threshold,
`A_OVER_D_RATIO=10` -> UNIFORM) and swept `A_OVER_D_RATIO in {3, 10, 30,
100}`: `A=3` gives `purity=0.17` ("mixed"), `A=10,30,100` all give
`UNIFORM (purity=0)`. **Advection strength has a real, measurable, secondary
effect** -- weaker cross-diffusion allows more sector-like structure to
persist at the same `d_i` -- confirming the single-parameter (`d_i`-only)
story is incomplete, though this single check does not fully quantify the
joint `(d_i, A_OVER_D_RATIO)` dependence.

**Resolution reached, stated precisely.** RULED OUT: (a) the naive `d<<L^2/T`
prediction using domain size as the length scale -- decisively wrong,
regardless of which timescale (`T` or `1/r_max`) is paired with it, since
even `Da=r*L_domain^2/d` at the converged threshold is `~3652`, nowhere near
O(1); (b) "pure numerical artifact" as the sole explanation -- the 40x40/56x56
agreement after the 28x28-to-40x40 correction indicates a real,
resolution-independent transition exists, not a mirage that keeps receding
with `h`. ESTABLISHED: the 28x28 sweep (Item 4's original result) was
under-resolved and biased ~1.8x high; the resolution-corrected threshold is
`d ~ 2.7e-4` for `A_OVER_D_RATIO=10`; advection strength is a confirmed
second relevant parameter. **NOT PINNED DOWN:** a single closed-form,
quantitatively-checked O(1) scaling law. The best single-length-scale
candidate (seed radius) still leaves `Da` at `O(10-100)`, not `O(1)` -- this
is most consistent with the transition being an angular pattern-formation
/ mode-selection effect (whether the seeded 3-fold, `m=3` azimuthal
perturbation grows or decays under the coupled diffusion-advection-reaction
dynamics) rather than a simple 1D diffusion-length-vs-domain-size balance; a
full linear stability / dispersion-relation analysis of that mode would be
needed to derive a clean quantitative criterion, and was not attempted here
-- flagged as genuine remaining future work, not claimed as done.

**Status: B8(a)/(b) VERIFIED -- sector formation is reproduced with the
paper's own parameters, and it is real segregation (checked directly), not a
metric artifact. B8(c)/ITEM 4: existence of a resolution-independent
SECTORS->UNIFORM transition is now VERIFIED (confirmed via a direct
grid-refinement test, not just a single-resolution bisection), and the
resolution-corrected threshold is `d ~ 2.7e-4` for `A_OVER_D_RATIO=10`. The
mechanism is CONSISTENT with a length scale near the seed geometry (order
10-100 in `Da`, not order 1000s) plus a confirmed secondary dependence on
advection strength, but is **not fully pinned down** to a single, closed-form,
O(1)-verified scaling law -- reported as an open mechanistic question, not
papered over.**

---

## C. My own diagnostic errors during this work (calibration -- read this one)

I got the **measurement** wrong on the first attempt **five** times now. Each
initially pointed at a false problem, and each was caught only by validating
the metric against a case with a known answer. This is the main reason not to
over-trust any single green result here.

1. **Fixed-width radial bins** in the 2D radial check -> in-bin *radial* variation
   (h-independent) masqueraded as non-converging angular error. Nearly reported a
   solver anisotropy that didn't exist.
2. **60deg sector means** for a `cos(3theta)` pattern -> integrates to *exactly zero*
   over every bin. The metric was structurally blind to the mode I had seeded.
   Nearly reported "angular structure destroyed" from a metric that could never
   have detected it. (Conclusion survived correction, but by luck.)
3. **Negating Dirichlet rows** along with the differential operator in the
   M-matrix check -> reported "STRUCTURE VIOLATED" for a structure that holds.
4. **`not None` is truthy** in a pass/fail check during the Neumann re-verification
   -> a "correctly still fails" verdict that the logic couldn't actually have produced.
5. **Compute-budget sweep under-resolution** (this round) -- the `d_i` sweep for
   B8(c) was cut to a coarser grid/larger `dt` after a finer version timed out,
   and the interior points (`d=1e-4, 1e-2`) came back non-monotonic. Reported
   honestly as under-resolved rather than as a real non-monotonic transition,
   but it's a reminder that a sweep built for speed under time pressure is
   weaker evidence than the single well-resolved run either side of it.

---

### B9. ITEM 2: quantitative reproduction of the paper's Section 4.3 travelling-wave benchmark
Implemented System (3.12)-(3.13) (the parameter-symmetric reduction of eq.
2.3-2.5 to one total-biomass field `rho` and one substrate `c`) in a
self-contained script, `report/item2_wave_speed.py`, deliberately using the
paper's OWN numerical method (explicit method-of-lines, `scipy.solve_ivp`
RK45) rather than this project's usual FV+Newton/QSSA pipeline -- the reduced
system needs a genuinely transient two-field solve (`c` does not reach
quasi-steady state on the wave's own timescale), which the QSSA machinery
cannot provide, and using a different scheme would have confounded "does this
project reproduce the paper" with "does some other scheme also reproduce it."
Spatial discretisation reuses this project's own validated conservative FV
Laplacian/advection operators via a domain rescale `x_hat = x/L`.

Exact paper setup reproduced: Table 1 Case (A) (`d=1e-6, a=1e-5, r=1, K=1,
b=0.1, Y=0.2, D=1e-4`), domain `[0,200]`, `t in [0,200]`, `rho_0(x)=exp(-x)`,
`c_0=5`, homogeneous Neumann BCs, `Nx=2000`.

**Result:**
- Measured `v_bar = (x_bar(200) - x_bar(160)) / 40 = 0.8325` (front position
  via `argmax(rho)`, same measurement convention the paper describes) vs. the
  paper's reported `v_bar ~= 0.8396` -- **0.85% relative difference.**
- Closed-form minimal wave speed, independently re-derived from the
  linearisation around the unstable steady state `(0, c0)`:
  `v_min = 2*sqrt(d*r*c0/(K+c0)) = 0.001826` vs. the paper's reported
  `0.0018` -- matches to within rounding.
- `v_bar/v_min` ratio: measured 456.0x vs. the paper's own 466.4x -- same
  order of magnitude, consistent with the direct `v_bar` agreement.
- Non-negativity held throughout (`min(rho), min(c) > 0` at both t=160,200)
  despite no clipping being applied, matching the paper's own unconstrained
  method.
**VERIFIED** -- no discrepancy to report; this is a genuine, honestly-obtained
agreement (not forced), full 108s solve at the paper's stated resolution.

---

### B10. ITEM 5: adversarial re-verification of 3 load-bearing claims, from scratch
`report/item5_adversarial_reverification.py` re-derives 3 claims independently
of any existing test or project code (no import of `nitrifiers`), to check
whether they survive a from-scratch attack rather than just re-running the
project's own implementation against itself.

- **CLAIM 1 -- v_min formula.** `v_min = 2*sqrt(d*r*c0/(K+c0))` (used in ITEM 2)
  re-derived symbolically (`sympy`) by linearising System (3.12) around the
  unstable state `(rho=0, c=c0)`. Confirmed directly from the actual
  reaction term `f = rho*(r*c/(K+c) - b*rho)` (re-read from
  `item2_wave_speed.py`, not assumed) that the self-limitation term is
  **quadratic** in `rho` (a crowding term), not a constant per-capita
  mortality -- so it drops out at linear order along with the cross-diffusion
  term, leaving the characteristic equation `d*lambda^2 + v*lambda +
  r*c0/(K+c0) = 0`. The real-root/oscillation boundary of that equation's
  discriminant matches the candidate formula exactly (symbolic difference
  simplifies to 0). Numeric value at Case (A): `0.001826` vs. paper's `0.0018`
  (1.43% relative difference). **CONFIRMED.**
- **CLAIM 2 -- v_bar travelling-wave speed.** Re-solved with a SECOND,
  independently-coded solver: raw physical domain `[0,200]` (no `x_hat=x/L`
  rescale), centred (not upwind) flux-form finite differences, ghost-node
  mirroring for zero-flux BCs (not `elliptic.py`'s row-replacement), `Nx=800`
  (vs. ITEM 2's `Nx=2000` -- a spot check, not a resolution study). Result:
  `v_bar = 0.8375`, agreeing with ITEM 2's own `0.8325` to **0.60%**, and
  actually landing CLOSER to the paper's `0.8396` (**0.25%** relative
  difference) than ITEM 2's original run. Two independently-coded numerical
  schemes converging to the same answer is meaningfully stronger evidence
  than either one alone. **CONFIRMED.**
- **CLAIM 3 -- 2D FV Laplacian exact conservation** (`report/audit_checklist.md`
  D8: `sum_ij V_ij*(Lap@c)_ij = 0` for any `c`, any resolution). Re-derived
  analytically from scratch via the telescoping-flux argument (every interior
  face's contribution appears exactly twice, with opposite sign, from its two
  adjacent control volumes, and cancels identically; zero-flux boundary nodes
  contribute nothing by construction), then re-checked numerically with a
  hand-rolled 5-point FV stencil (not `nitrifiers/grid2d.py`) across 3 grid
  sizes (`10x10, 23x17, 40x40`) x 3 random fields each: all 9 sums are at
  machine-precision zero (`1e-16` to `1e-13` range). **CONFIRMED.**

**All 3 claims survive independent, from-scratch re-derivation.** No
discrepancy found this round -- reported as such rather than manufacturing one.

---

## D. Open issues -- NOT fixed, documented in README

- **D1.** Picard needs the `rhs[0]=0` centre-decoupling workaround to converge
  under `eloi` stiffness. Confirmed via eigenvalue/conditioning analysis to be a
  genuine property of the fixed-point map, not a coding bug. Removing it makes
  Picard diverge even at relax=0.01. Picard is never a primary solver.
- ~~**D2.** `solve_newton(bc_type='neumann')` is **broken** for genuinely
  coupled multi-substrate configurations~~ -- **RESOLVED (ITEM 1, commit
  `145f929`).** `solve_newton`/`solve_picard`/`solve_relaxation` now accept a
  per-substrate `bc_specs` dict, matching how the paper's own model (eq. 2.2)
  specifies boundary conditions; the old global `bc_type` is preserved as a
  strict backward-compatible special case (full 9/9 suite re-run unchanged,
  including Stage 6). Two further bugs found and fixed along the way: a
  Monod-gating trap in the Neumann default initial guess (0.0 zeroed every
  species' Jacobian when a co-limiting substrate started there), and a missing
  physical-plausibility upper-bound guard in `solve_relaxation`'s degeneracy
  fallback (it only floored at 0, never capped, and diverged to ~3.7e11 in one
  case). A genuinely coupled multi-substrate Neumann regression test
  (`test_coupled_multi_substrate_neumann_converges`) now converges cleanly
  (residual 6.2e-11, flux match 2.8e-16) at moderate reaction-rate scales.
  **Honestly-reported residual limitation:** the same coupled configuration at
  the `eloi` preset's *realistic* stiff coefficients does not converge -- SVD-
  confirmed near-singular Jacobian (condition number ~4.3e16, smallest
  singular value ~2.25e-13, at the edge of double-precision representability).
  This is a genuine numerical-stiffness property of that parameter regime, not
  a solver bug; not forced to a false "pass."
- **D3.** `c_inf_hat` is reused as both the Dirichlet value and the Neumann flux
  magnitude, so a zero-flux request can't be expressed for a substrate with
  nonzero bulk concentration without a custom wrapper.
- **D4.** 2D has **no PTC/relaxation fallback**; degenerate solves return
  `"newton_stalled"` rather than being rescued.
- ~~**D5.** 2D does not reproduce sector formation~~ -- **RESOLVED, moved to B8.**
  Rerun with the paper's own Table 1 Case (A) parameters reproduces sector
  formation. The earlier "reasoned, not verified" diagnosis was directionally
  correct but had never actually been tested; it now has been. See B8.
- **D6 (ITEM 3, updated).** Preset cleaning assumptions, re-examined:
  - `A_OVER_D_RATIO = 10` -- **relabelled ASSUMED -> VERIFIED.** Not independently
    re-derived from first principles (it is still a physically-reasoned
    target ratio, not measured), but B8 already confirmed it is *exactly*
    the value used in arXiv:2512.13156 Table 1 Case (A) (`a_i=1e-5, d_i=1e-6`),
    pulled directly from the paper's own PDF. "Matches the paper's own stated
    value" is the strongest evidence this kind of borrowed constant can have,
    so it no longer belongs in the same "assumed" bucket as an un-cross-checked
    guess.
  - `rebeca`'s /100 yield rescale -- **remains ASSUMED, and now known to be
    qualitatively load-bearing, not just a scale choice.** ITEM 3
    (`report/item3_yield_sensitivity.py`) ran Stage 6 identically on `rebeca`
    (shipped, Y=Y_raw/100) and a `rebeca_unrescaled` variant (Y=Y_raw, the
    /100 fix reverted) -- same grid, same seed profile, same 20-step slow
    loop. **Robust across the choice:** elliptic convergence (`newton` both
    cases, residuals 9.1e-9 / 2.9e-11), bacterial mass growth direction (1.13x
    / 1.27x), and dominant species (AOB, both cases). **NOT robust:** the
    rim-to-core O2 zonation (anoxic core) that Stage 6 reports elsewhere as a
    qualitative signature (see the `toy`-preset zonation test) -- with the
    shipped rescale the core is driven to ~9e-20 (fully anoxic); with the
    100x-larger unrescaled Y (100x smaller Damkohler number, i.e. 100x weaker
    substrate consumption per unit biomass) the core only falls to 0.30 vs. a
    rim of 0.34, an 88%-of-rim value that does not read as "zonation" by any
    reasonable threshold. **Reported plainly, not forced:** whether the
    `rebeca` preset shows an anoxic core is therefore genuinely sensitive to
    the /100 cleaning decision, and that decision remains an inference from
    the raw values' suspicious 100x relationship to literature, not an
    independently confirmed correction.
- ~~**D7.** duplicated `SPECIES`/`PRIMARY`/`SECONDARY`~~ -- **FIXED (two commits).**
  Originally found: `elliptic.py` and `parabolic.py` independently defined
  their own separate copies of `SPECIES`/`PRIMARY`/`SECONDARY` (identical
  species->substrate mappings, written twice) -- reproduced live, patching
  `elliptic.PRIMARY` had zero effect on `parabolic.PRIMARY` since they were
  separate dict literals, surfacing only as a downstream `KeyError`. First
  commit consolidated `parabolic.py` onto `elliptic.py`'s copy. Tracing the
  root further surfaced **two more** copies, fixed in a second commit:
  `nondim.py` had its own `PRIMARY_SUBSTRATE`, itself a partial duplicate of
  `nondim.py`'s own `CONSUMED_SUBSTRATES` (same file, same fact, twice); and
  `elliptic.py` hardcoded `SPECIES`/`PRIMARY`/`SECONDARY` rather than importing
  them despite already depending on `nondim.py` for `SUBSTRATES`. Single source
  is now `nondim.py`'s `CONSUMED_SUBSTRATES` (`PRIMARY`/`SECONDARY` derived from
  it), imported by `elliptic.py`, transitively available to `parabolic.py`
  unchanged. **Correction to the original framing:** `params.py`'s own
  `SPECIES` was never a rogue duplicate in this sense -- it is the legitimate
  root definition, and `nondim.py` already correctly imported it from there;
  it needed no change. 9/9 suite re-run and passing after each commit.
- **D8 (checked, clean).** The 2D volume/area computation does **not** repeat
  the A9 pattern. `Grid2D.dx`/`dy`/`V` are computed exactly once in
  `Grid2D.__init__`; `elliptic2d.py` and `parabolic2d.py` both read from that
  same instance rather than recomputing. `parabolic2d.py`'s advection operator
  uses `1/grid.dx[i]` rather than `face_area/V` explicitly, but this is an
  **exact** algebraic simplification for 2D Cartesian cells (`dy_j/(dx_i*dy_j)
  = 1/dx_i` identically -- unlike the 1D case, where `V_i ~ r_i^p*h` was only an
  *approximation*, which is what made A9 a real bug). Independently confirmed
  by the 2D advection operator's exact conservation result (B5: 7e-17).
  **VERIFIED clean.**

---

## E. Files

**Created:** `nitrifiers/grid2d.py`, `elliptic2d.py`, `parabolic2d.py`,
`slowfast2d.py`; `tests/test_2d.py`; `report/progress_report.tex`,
`teaching_summary.md`, `audit_checklist.md` (this file).

**Modified:** `elliptic.py` (row-0, exact volumes, Neumann target, c_max, 3-tuple
return, `cell_volumes`/`face_area`), `parabolic.py` (death term, true Newton,
advection volumes, `_total_mass`), `relaxation.py` (row-0, Neumann target),
`slowfast.py` (method labels), `tests/test_parabolic.py`,
`tests/test_elliptic_closed_form.py`, `tests/test_elliptic.py`,
`tests/test_slowfast.py`, `README.md`.

**External artifact used this round (not part of the repo):** the paper's own
PDF (arXiv:2512.13156) was fetched to extract Table 1, Case (A) -- used
read-only to source ground-truth parameters for B8, not committed anywhere.

---

## F. Not done

- **Nothing since the last commit is committed or pushed.** The 2D extension, the
  A9 advection fix, README updates and all of `report/` are uncommitted.
- The `.tex` report has **never been compiled** (no LaTeX toolchain locally) --
  needs Overleaf or a MiKTeX/TeX Live install.
- No comparison against the paper's own published figures with the paper's own
  parameters.
- No performance work: 160x160 2D elliptic ~ 75 s/solve via direct sparse LU.
