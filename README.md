# PDE Model for Nitrifying Bacteria in Biofilms

A numerical PDE solver for a slow-fast reaction-diffusion-advection system
modelling three competing nitrifying bacterial species (AOB, NOB, comammox
CMX) growing in a biofilm/granule, based on the model in
[arXiv:2512.13156](https://arxiv.org/abs/2512.13156) (Freingruber,
Gonzalez-Cabaleiro, Yoldas).

## Model

Three bacterial densities `u_AOB, u_NOB, u_CMX` (slow, parabolic) are coupled
to four substrate concentrations `c_NH4, c_NO2, c_NO3, c_O2` (fast,
quasi-steady/elliptic) on a 1D radial (spherical/cylindrical) or slab domain:

```
d/dt u_i = div(d_i grad(u_i) + a_i u_i grad(rho)) + u_i f_i(u_i, c)      (slow)
0        = D_j Delta(c_j) + c_j g_j(u, c)                                 (fast, quasi-steady)

rho = u_AOB + u_NOB + u_CMX
f_i(u_i, c) = r_i * M(c_p; K_ip) * M(c_s; K_is) - b_i * rho     (Monod growth, density-dependent death)
```

The scale separation between fast substrate diffusion and slow bacterial
growth (`epsilon = r_max * L^2 / D_j << 1`, verified per parameter preset)
justifies treating the substrate system as quasi-steady at each slow time
step.

## Pipeline (six stages)

| Stage | Module | Role |
|---|---|---|
| 1 | `nitrifiers/params.py` | Dimensional parameter presets (`toy`, `rebeca`, `eloi`) |
| 2 | `nitrifiers/nondim.py` | Non-dimensionalization; builds `Lambda`, `Khat`, etc. |
| 3 | `nitrifiers/elliptic.py` | Quasi-steady substrate solver (Newton + Picard) |
| 4 | `nitrifiers/relaxation.py` | Pseudo-transient continuation fallback/cross-check for Stage 3 |
| 5 | `nitrifiers/parabolic.py` | Fully-implicit Newton solver for bacterial density evolution |
| 6 | `nitrifiers/slowfast.py` | Couples Stages 3-5 into the full slow-time loop |

### Parameter presets

- **`toy`** — round numbers, for solver development and unit tests.
- **`rebeca`** — rough estimate from an e-mail (`parameters_nitrifiers_clean.json`).
- **`eloi`** — thermodynamics/literature-derived kinetics (Martinez-Rabert et al.).

All cleaning/unit decisions on the raw parameter sources are documented
inline in `params.py`.

## Numerics

- Finite-volume discretization on a 1D radial/slab grid, with **exact**
  control-volume weighting at every node (including the domain center,
  `r=0`) — verified 2nd-order accurate via closed-form (cosh/Bessel/sinh)
  benchmarks and cross-checked against a 50-digit `mpmath` reference to rule
  out floating-point artifacts at fine grids.
- **Stage 3 (elliptic)**: full Newton with backtracking line search and
  non-negativity projection is the primary solver. A Picard/fixed-point
  solver is also implemented but does **not** converge for the reaction
  stiffness of the `eloi` preset without the `rhs[0]=0` center-decoupling
  workaround still present in the code — it is kept for reference/weaker
  regimes, not relied on.
- **`solve_newton` Neumann support**: both boundary condition types are
  implemented (`bc_type="dirichlet"` / `"neumann"`); the Neumann row's
  residual target correctly reflects the requested flux value (verified
  against the closed-form linear-reaction Neumann solutions,
  `c(r) = A*cosh/I0/sinh(kappa*r)` per geometry, with `A` fixed by the flux —
  see `test_nonzero_flux_neumann_matches_closed_form`). **Caveat:** this is
  verified only via a mixed-BC construction (Dirichlet for O2/NO2/NO3, Neumann
  for NH4); the standard entry point with `bc_type="neumann"` applied to *all
  four coupled substrates simultaneously* remains unreliable (diverges or
  hangs on realistic coupled configurations) — a separate, still-open
  conditioning issue, not something fixed here. Stage 6 (`slowfast.py`) always
  uses Dirichlet and is unaffected.
- **`solve_newton` robustness guards**: two internal safeguards, both
  triggerable independently and both reported honestly via the returned
  `method` string (`"newton"`, `"newton_inner_relax_fallback"`, or — one level
  up, from `slowfast.py::solve_c_given_u` — `"outer_relaxation_backstop"`):
  (1) a physical-plausibility upper bound on concentrations, rejecting
  backtracking steps that reduce the residual but land on a spurious,
  unphysical root; (2) an automatic fallback to `relaxation.solve_relaxation`
  (targeting the *caller's own* `tol`, not a hardcoded default) when the
  Newton Jacobian is genuinely singular — e.g. a substrate correctly driven to
  exactly 0 under a sealed (zero-flux) boundary. Neither guard affects normal
  Dirichlet operation (confirmed: zero regressions across the full test
  suite).
- **Stage 4 (relaxation)**: pseudo-transient continuation, used as an
  independent cross-check of Stage 3 (agreement to ~1e-11 across all three
  presets and a range of densities).
- **Stage 5 (parabolic)**: fully-implicit backward Euler with the complete
  Newton Jacobian, including the cross-diffusion `d(Adv(rho))/d(rho)` term
  (needed for clean convergence on sharp density fronts).
- **Shared volume measure**: every conservative operator (`build_laplacian`,
  `build_advection_matrix`, `build_advection_rho_jacobian`) and the mass
  diagnostic `_total_mass` are normalised by the *same* exact control volumes,
  `elliptic.cell_volumes`. This matters: a finite-volume operator normalised by
  one measure but integrated against another conserves nothing. The advection
  operator previously used the approximate `V ~= r^p*h` — wrong by ~2x at the
  outer boundary node in **every** geometry — so diffusion and advection
  conserved different measures and the coupled scheme conserved neither, with
  an O(1) defect that did not vanish under refinement. Both operators are now
  exactly conservative to machine precision (~1e-16).

### 2D Cartesian extension

`grid2d.py` / `elliptic2d.py` / `parabolic2d.py` / `slowfast2d.py` extend the
same validated finite-volume + quasi-steady machinery to a 2D Cartesian domain,
reusing `reaction_and_jacobian` verbatim (it is elementwise and already
dimension-agnostic). Validated in `tests/test_2d.py` by two reductions to the
1D solvers: an **exact** slab reduction (agreement to ~1e-14, i.e. solver
tolerance rather than discretisation error) and a **radial** reduction against
the cylindrical `p=1` solver — note 2D Cartesian radial symmetry gives
`c_rr + (1/r)c_r`, the `p=1` operator, *not* spherical `p=2`. The deep interior
converges to the 1D reference at 2nd order (2.2e-8 at N=160); the near-boundary
band is 1st order because a circle cannot be represented exactly on a Cartesian
grid (staircasing), which is a geometry limitation, not a solver defect.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, NumPy, SciPy.

## Usage

```python
from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.elliptic import Grid
from nitrifiers.slowfast import run_slow_loop
import numpy as np

grid = Grid(N=60, geometry="radial", p=2)  # spherical granule
r = grid.r
U0 = {sp: 0.02 * np.exp(-((r - 0.5) / 0.15) ** 2) for sp in ("AOB", "NOB", "CMX")}

U, C, history, _ = run_slow_loop("eloi", grid, U0, n_slow_steps=100, dt_slow=0.05)
```

## Tests

```bash
pytest tests/
```

Test coverage includes: closed-form ground-truth convergence tests for the
elliptic solver (slab/cylindrical/spherical geometries, both Dirichlet and
nonzero-flux Neumann), mass-conservation checks, Newton-vs-relaxation
cross-validation across presets/densities, analytic-equilibrium and
Jacobian-vs-finite-difference checks for the parabolic solver, and anoxic-core
zonation reproduction.

## Known limitations

- The Picard solver in `elliptic.py` requires a center-row RHS workaround to
  converge and is not a drop-in replacement for Newton under the `eloi`
  preset's reaction stiffness; this is a property of the fixed-point map,
  not the spatial discretization.
- `solve_newton`/`solve_picard`/`solve_relaxation` now take a per-substrate
  `bc_specs = {sub: (bc_type, value)}` dict, so each substrate can be
  independently Dirichlet or (possibly nonzero-flux) Neumann; the old global
  `bc_type` parameter is still accepted and expanded into the equivalent
  per-substrate spec. A genuinely-coupled multi-substrate Neumann case is
  regression-tested and converges cleanly at moderate reaction-rate scales.
  **Remaining, honestly-characterized limitation:** that same coupled
  configuration does not converge at the `eloi` preset's realistic stiff
  coefficients — confirmed via direct SVD to be a genuinely near-singular
  Jacobian there (condition number ~4.3e16), not a solver defect.
- `solve_newton`'s return signature is `(C, history, method)` — a 3-tuple, not
  2. All in-repo call sites are updated; any new caller must unpack three
  values.
- The 2D solver has **no relaxation/PTC fallback** (there is no 2D PTC solver);
  a degenerate solve returns `method="newton_stalled"` and the achieved
  residual rather than being silently rescued. Callers should inspect
  `elliptic_residual` rather than assuming every step converged.
- The 2D solver **does** reproduce sector formation when run with the source
  paper's own Table 1 Case (A) parameters (`d_i=1e-6, a_i=1e-5`, three inocula
  seeded 120 degrees apart): colony expands, interior depletes and decays, and
  each species dominates a distinct angular wedge aligned with its seed angle
  (checked directly per-sector, not via a Fourier-mode proxy that can miss the
  seeded pattern). Deviation flagged, not hidden: the paper uses a
  time-dependent substrate with near-zero Neumann influx; this reproduction
  uses the project's validated QSSA + Dirichlet machinery instead — the
  paper's own Case (A) is actually outside the epsilon-small regime that QSSA
  assumes (`eps = r*L^2/D = 1e4` there), a structural mismatch worth noting on
  its own. The sector/uniform transition is a real, resolution-independent
  effect: a bisection sweep (`report/item4_diffusion_threshold_sweep.py`) at
  28x28 resolution located it at `d_i ~ 4.9e-4`, but a direct grid-refinement
  check (`report/item4b_threshold_mechanism.py`, 40x40 and 56x56) showed that
  estimate was under-resolved and biased ~1.8x high — the resolution-corrected
  threshold is `d_i ~ 2.7e-4` (40x40/56x56 agree). The naive `d << L^2/T`
  scaling estimate (using domain size) is decisively wrong regardless of
  timescale choice (`Da = r*L_domain^2/d ~ 3650` at the corrected threshold,
  nowhere near O(1)); the best single length-scale candidate tested (seed
  radius) gets to `Da ~ 37` (order 10-100, not order 1000s) but not to a
  clean O(1). Advection strength (`A_OVER_D_RATIO`) has a confirmed secondary
  effect on the threshold location. **Mechanism not fully pinned down**: most
  consistent with an angular pattern-formation/mode-selection effect (whether
  the seeded `m=3` perturbation grows or decays) rather than a simple
  diffusion-length-vs-domain-size balance, but the linear-stability analysis
  that would nail this down was not attempted — see `report/audit_checklist.md`
  B8/ITEM 4 for the full investigation, including what was ruled out.
- Parameter presets carry documented cleaning assumptions (`params.py`
  docstring, note 1 and note 7). `A_OVER_D_RATIO = 10` is **VERIFIED**: it
  exactly matches the `a_i/d_i` ratio used in arXiv:2512.13156's own Table 1
  Case (A) toy simulations. The `rebeca` preset's ÷100 yield rescale remains
  **ASSUMED** (a plausible units-slip correction, not independently
  confirmed against the original source) and is **not merely cosmetic**: a
  sensitivity check (`report/item3_yield_sensitivity.py`) found that reverting
  it changes whether `rebeca`'s Stage 6 run develops an anoxic core (present
  with the shipped rescale, absent without it), while leaving solver
  convergence, mass-growth direction, and dominant species unchanged.

## Reproducing the validation results

`report/` holds every quantitative claim made about this codebase as a
runnable script plus its captured output, so a reviewer can regenerate any of
them independently rather than trusting the prose:

| Script | What it reproduces |
|---|---|
| `report/item2_wave_speed.py` | Paper Sec. 4.3 travelling-wave benchmark (measured `v_bar` vs. the paper's 0.8396, closed-form `v_min` vs. 0.0018) |
| `report/item3_yield_sensitivity.py` | Sensitivity of Stage 6's qualitative conclusions to the `rebeca` preset's ÷100 yield rescale |
| `report/item4_diffusion_threshold_sweep.py` | Bisection-refined diffusion-length threshold for 2D sector formation |
| `report/item4b_threshold_mechanism.py` | Grid-refinement check + advection-strength sweep investigating *why* the naive `d<<L^2/T` scaling missed the ITEM 4 threshold |
| `report/item5_adversarial_reverification.py` | 3 load-bearing claims (`v_min` formula, `v_bar`, 2D Laplacian conservation), each re-derived from scratch with no import of `nitrifiers` |

```bash
pip install -r requirements.txt
pytest tests/                                        # full regression suite, ~48 tests
python report/item2_wave_speed.py                     # ~2 min
python report/item3_yield_sensitivity.py               # ~seconds
python report/item4_diffusion_threshold_sweep.py        # ~30 s
python report/item4b_threshold_mechanism.py              # ~2 min (grid-refinement check)
python report/item5_adversarial_reverification.py       # ~1 min
```

`report/audit_checklist.md` is the running record of every claim's
verification status (`VERIFIED` / `CONSISTENT` / `ASSUMED`), including the
ones that turned out to be genuine, honestly-characterized discrepancies
rather than confirmations — see B8/ITEM 4 for an example where a specific
quantitative prediction was contradicted by ~2 orders of magnitude while the
underlying qualitative claim held up. `report/progress_report.tex` and
`report/teaching_summary.md` are longer-form narrative writeups of the same
material.

## Citing this work

See [`CITATION.cff`](CITATION.cff). If you use this code, please also cite
the source model paper below. For a specific reproducible snapshot, cite a
tagged release (`git tag`) rather than an untagged commit — see "Archival"
below. **No `LICENSE` file exists in this repository yet** — `CITATION.cff`
intentionally omits a `license` field rather than guessing one; add a license
file and update `CITATION.cff` before treating any release as reusable by
others under specific terms.

## Archival

This repository is tagged at points corresponding to completed validation
milestones (`git tag -l`) so that a specific, reproducible state of the code
can be referenced independently of ongoing development on `main`. Archiving a
tagged release on [Zenodo](https://zenodo.org/) (which mints a permanent DOI
for a GitHub release) is **optional** and has not been done for this
repository — if a DOI is needed (e.g. for a thesis/paper citation), enabling
the Zenodo GitHub integration on a tagged release is the standard route; it
is not required for the code or its validation record to be usable as-is.

## Reference

Freingruber, Gonzalez-Cabaleiro, Yoldas. *A slow-fast PDE model for
nitrifying bacteria in biofilms.* [arXiv:2512.13156](https://arxiv.org/abs/2512.13156).
