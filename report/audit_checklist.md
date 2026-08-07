# Audit checklist — everything done, with how to independently verify it

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

### A1. Death term: constant `b_i` instead of density-dependent `b_i·ρ`
- **Where:** `parabolic.py`, reaction term.
- **How found:** you flagged it in code review; I confirmed against the rendered
  arXiv PDF eq. (2.1)/(2.5). An earlier pdftotext extraction had silently
  dropped the `ρ` symbol, which is how it survived.
- **Impact:** large — with constant death there is no self-limiting mechanism,
  biomass can grow unbounded. With `b_i·ρ` growth saturates at `ρ ≈ g_i/b̂_i`.
- **Verify:** `tests/test_parabolic.py::test_death_term_is_density_dependent_not_constant`
  — checks the per-step growth *ratio* strictly decreases under uniform ample
  substrate. A constant death term gives an exactly constant ratio.
- **Cross-check yourself:** open the arXiv PDF at eq. 2.1 and confirm the `−b_i ρ`.
- **Status: VERIFIED**

### A2. Reaction term zeroed at the r=0 row
- **Where:** `elliptic.py` (`_residual`, `_assemble_global`) and `relaxation.py`.
- **Cause:** r=0 was treated as a boundary-condition row. It isn't — it's a
  genuine interior PDE point that happens to use a symmetric stencil because of
  the coordinate singularity. Only row N is a true BC row.
- **Impact:** degraded a 2nd-order scheme to ~1st order whenever reactions were
  active. Invisible in the pure-diffusion case.
- **Verify:** `tests/test_elliptic_closed_form.py::test_spherical_error_converges_with_grid_refinement`
- **Status: VERIFIED**

### A3. Parabolic Newton was only "modified", not true Newton
- **Where:** `parabolic.py` — the Jacobian omitted `M_i = ∂[Adv(ρ)·u_i]/∂ρ`.
- **Impact:** residual was still exact so it converged to the *right* answer on
  smooth fields, but stalled at ~1e-4 on sharp density fronts.
- **Fix:** added `build_advection_rho_jacobian`, included in every column block
  (since `ρ = Σ_j u_j`, it appears in all of them).
- **Verify:** `test_advection_rho_jacobian_matches_finite_difference` and
  `test_sharp_front_newton_converges_to_tolerance` (asserts zero stall warnings).
- **Status: VERIFIED**

### A4. Approximate cell volume near the domain centre in `build_laplacian`
- **Where:** `elliptic.py::build_laplacian`, interior rows used `V_i ≈ r_i^p·h`.
- **Impact:** relative error 7.7% at i=1, 2.0% at i=2 — held convergence order at
  ~1.7–1.9 instead of 2.0 even after A2 was fixed.
- **Fix:** exact `V_i = (r_e^{p+1} − r_w^{p+1})/(p+1)` at every row.
- **Verify:** mass conservation to machine precision for all 3 geometries;
  and 2nd-order convergence (see B2 for the mpmath confirmation).
- **Status: VERIFIED**

### A5. Neumann residual target hardcoded to `0.0` (three separate places)
- **Where:** `elliptic.py::_residual`, `elliptic.py::solve_picard`,
  `relaxation.py::solve_relaxation`.
- **Cause:** the matrix row was built correctly by `apply_bc` for any requested
  flux, but the residual/RHS target was independently written as `0.0` instead
  of `−value`. Two pieces of code representing the same equation, built
  separately, drifted apart.
- **Impact:** **any** nonzero-flux Neumann solve, in any of the three solvers,
  silently solved a *zero*-flux problem. No error, no warning.
- **Verify:** `tests/test_elliptic_closed_form.py::test_nonzero_flux_neumann_matches_closed_form`
  — checks both agreement with the closed form AND that a finite-difference
  recomputation of the boundary flux from the converged profile equals the
  requested value (matched to 1.1e-14).
- **Status: VERIFIED for a single-substrate (mixed-BC) configuration.
  See D2 — the multi-substrate production path is still broken.**

### A6. Spurious unphysical root accepted by backtracking
- **Where:** `elliptic.py::solve_newton`.
- **Cause:** backtracking accepted any step that reduced the residual norm, with
  no plausibility constraint. Under sealed (zero-flux) boundaries with reaction
  active, it converged to NH4 ≈ 100.5 and NO3 ≈ 5416 — orders of magnitude above
  anything the closed system could supply. The residual really was near zero;
  the root was mathematically real but physically nonsense.
- **Fix:** added an upper bound `c_max = 5 × max|c_inf|`; a trial step must both
  reduce the residual AND stay inside `[0, c_max]`.
- **Verify:** rerun the sealed zero-flux + reaction case; max concentration now
  stays < 10 (was ~5400).
- **Status: VERIFIED**

### A7. Sign bug in the `c_max` guard (self-inflicted by A6, then found)
- **Cause:** `c_max = 5 × max(c_inf.values())`. Under Neumann, `c_inf` doubles as
  the *flux*, which is legitimately **negative** for an influx. `max(−0.3,0,0,0)
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
  the fallback reaches ~7.3e-12 — past where the old default would have stopped.
- **Status: VERIFIED. Note the breaking API change: `solve_newton` now returns a
  3-tuple. All 6 in-repo call sites updated.**

### A9. Advection operator normalised by the wrong control volume
- **Where:** `parabolic.py::build_advection_matrix`, `build_advection_rho_jacobian`,
  `_total_mass`.
- **Cause:** still using `V ≈ r^p·h` after `build_laplacian` had been corrected to
  exact volumes (A4). Wrong by O(h²/r²) on the first interior rows and — more
  seriously — by a **factor of ~2 at the outer boundary node in every geometry
  including the slab** (full cell used where only a half-cell exists).
- **Impact:** diffusion conserved one measure and advection another, so the
  coupled Stage 5 scheme conserved **neither**. Defect was O(1) and did **not**
  vanish under refinement (0.449 → 0.421 → 0.407 → 0.400 for N=40→320).
- **Why no test caught it:** the regression test's probe was `u = sin(πr)`, which
  is exactly **zero at r=1** — the single node carrying the factor-of-2 error.
  With `u = sin(πr) + 0.5` the defect is ~100× larger and non-converging.
- **Fix:** added `elliptic.cell_volumes` / `face_area` as a single source of
  truth; all conservative operators and the mass diagnostic now share it.
  `build_laplacian` was refactored onto it and **verified bit-identical**
  (0.000e+00 across 3 geometries × 3 resolutions).
- **Verify:** `test_advection_operator_is_exactly_mass_conservative` — now uses
  the exposing probe and asserts **exact** conservation (was: "defect halves",
  which a genuinely non-conservative operator can satisfy).
- **Impact on results:** 0.03–0.06% on biomass; peak locations, species shares
  and AOB dominance all unchanged. Structurally important, numerically small.
- **Status: VERIFIED**

---

## B. Validation performed

### B1. Closed-form benchmarks (Stage 3)
Reduced the full nonlinear system to a linear one with known solutions
(`cosh` / Bessel `I₀` / `sinh(κr)/r` for slab / cylindrical / spherical).
Newton err < 1e-4, Picard < 1e-3, at two Thiele moduli (κ=2.0 and κ=0.5).
`tests/test_elliptic_closed_form.py`. **VERIFIED**

### B2. Convergence order, floating-point-artifact-free
Double precision showed order degrading to 1.41 at fine grids. Traced to
**catastrophic cancellation** — the residual sums two ~1e7-magnitude terms
cancelling to O(1), leaving a ~1e-9 double-precision floor. Proven by:
tightening tol from 1e-9 to 1e-14 changed the errors **not at all** (identical to
6 sig figs). Re-solved the identical discretization at **50 decimal digits**
(`mpmath`, hand-written Thomas algorithm, bypassing SciPy's double-precision LU):

| N | max error | order |
|---|---|---|
| 50 | 1.9527e-05 | — |
| 100 | 4.8842e-06 | 1.999 |
| 200 | 1.2212e-06 | 2.000 |
| 400 | 3.0532e-07 | 2.000 |
| 800 | 7.6330e-08 | 2.000 |

Average 1.9997. **VERIFIED — the scheme is genuinely 2nd order.**

### B3. Newton ↔ relaxation cross-validation (Stage 4 vs Stage 3)
All 3 presets × 4 densities (1e-4, 0.01, 0.05, 0.5) = 12 combinations.
Agreement **8.7e-12 to 7.8e-11** — five orders tighter than the 1e-6 threshold.
Two independent algorithms agreeing is stronger evidence than either alone.
**VERIFIED**

### B4. Mathematical structure audit (7 checks, 4 never run before)
| Check | Result |
|---|---|
| Full elliptic Jacobian vs FD (4Npts², incl. Laplacian + BC rows) | 4.4e-06 / 1.7e-06 |
| Full parabolic Jacobian vs FD (3Npts², all blocks) | 5.9e-07 / 6.0e-07 |
| Discrete nitrogen conservation `Σ_j D_j R̂_j = 0`, 3 presets | rel 1.6e-16 |
| Same, end-to-end on converged solutions | rel 1.1e-14 / 2.5e-14 |
| Self-adjointness `V_iL_ij = V_jL_ji`, all geometries | machine precision |
| Temporal order of the slow-fast splitting | **1.04–1.07** (theory: 1) |
| M-matrix / discrete maximum principle | holds, all geometries & substrates |

The nitrogen-conservation check is the strongest new result: nitrification is
1:1 (NH4→NO2→NO3), so `Σ_j D_j R̂_j = 0` must hold *identically*. It does, to
1e-16 — an **independent** confirmation that `Lambda`/`LambdaProd` carry the
right diffusivities, re-validating an earlier bug fix from a different
direction than the test that originally caught it. **All VERIFIED**

### B5. 2D extension validation (`tests/test_2d.py`, 8 checks)
- Exact **slab reduction** to 1D: agreement ~1e-14, i.e. *solver tolerance*, not
  discretization error. Pins Laplacian + BCs + reaction + Jacobian assembly at once.
- **Radial reduction** against 1D cylindrical `p=1` (note: **not** `p=2` — 2D
  Cartesian radial symmetry gives `c_rr + (1/r)c_r`):
  deep interior 4.27e-07 → 1.00e-07 → 2.16e-08 (**2nd order**);
  near-boundary 1st order (staircased circle — geometry limit, not a defect).
- 2D Laplacian: exact conservation (1e-16), order 2.000, anisotropy converging faster.
- 2D advection: **exactly** conservative (7e-17) — better than the 1D operator.
- 2D ρ-Jacobian vs FD: 6.6e-10. Analytic equilibrium: 2.7e-09.
**All VERIFIED**

### B6. Full suite status
**9/9 test files pass** (8 pre-existing 1D + 1 new 2D), re-run after every change.

---

## C. My own diagnostic errors during this work (calibration — read this one)

I got the **measurement** wrong on the first attempt **four** times. Each initially
pointed at a false problem, and each was caught only by validating the metric
against a case with a known answer. This is the main reason not to over-trust any
single green result here.

1. **Fixed-width radial bins** in the 2D radial check → in-bin *radial* variation
   (h-independent) masqueraded as non-converging angular error. Nearly reported a
   solver anisotropy that didn't exist.
2. **60° sector means** for a `cos(3θ)` pattern → integrates to *exactly zero*
   over every bin. The metric was structurally blind to the mode I had seeded.
   Nearly reported "angular structure destroyed" from a metric that could never
   have detected it. (Conclusion survived correction, but by luck.)
3. **Negating Dirichlet rows** along with the differential operator in the
   M-matrix check → reported "STRUCTURE VIOLATED" for a structure that holds.
4. **`not None` is truthy** in a pass/fail check during the Neumann re-verification
   → a "correctly still fails" verdict that the logic couldn't actually have produced.

---

## D. Open issues — NOT fixed, documented in README

- **D1.** Picard needs the `rhs[0]=0` centre-decoupling workaround to converge
  under `eloi` stiffness. Confirmed via eigenvalue/conditioning analysis to be a
  genuine property of the fixed-point map, not a coding bug. Removing it makes
  Picard diverge even at relax=0.01. Picard is never a primary solver.
- **D2.** `solve_newton(bc_type='neumann')` is **broken** for genuinely coupled
  multi-substrate configurations (>1 substrate under nonzero flux, which the
  single global `bc_type` forces). Diverges to residual ~2e9 or hangs. A5 fixed
  the *target formula*; this is a separate conditioning problem. **Does not
  affect Stage 6, which is Dirichlet-only.**
- **D3.** `c_inf_hat` is reused as both the Dirichlet value and the Neumann flux
  magnitude, so a zero-flux request can't be expressed for a substrate with
  nonzero bulk concentration without a custom wrapper.
- **D4.** 2D has **no PTC/relaxation fallback**; degenerate solves return
  `"newton_stalled"` rather than being rescued.
- **D5.** 2D does **not** reproduce the paper's Fig 3/4 sector formation.
  Diagnosed as setup/parameter regime (no angular forcing in a symmetric domain;
  `D̂≈4.2` gives diffusion length > domain; sectors there need a sharp front
  expanding into empty space). **This diagnosis is reasoned, not verified against
  the paper's actual parameters.**
- **D6.** Preset cleaning assumptions remain **ASSUMED**: the `rebeca` ÷100 yield
  rescale, and `A_OVER_D_RATIO = 10`.

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

---

## F. Not done

- **Nothing since the last commit is committed or pushed.** The 2D extension, the
  A9 advection fix, README updates and all of `report/` are uncommitted.
- The `.tex` report has **never been compiled** (no LaTeX toolchain locally) —
  needs Overleaf or a MiKTeX/TeX Live install.
- No comparison against the paper's own published figures with the paper's own
  parameters.
- No performance work: 160×160 2D elliptic ≈ 75 s/solve via direct sparse LU.
