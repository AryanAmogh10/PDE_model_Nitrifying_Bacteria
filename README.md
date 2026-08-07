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
- **`solve_newton(bc_type="neumann")` is unreliable for genuinely-coupled
  multi-substrate configurations** (i.e. more than one substrate under
  nonzero flux at once, which the single global `bc_type` parameter forces).
  Directly tested: diverges (residual ~2e9 after the internal relaxation
  fallback exhausts its step budget) or hangs, depending on the coupling
  strength. Only a single-substrate-nonzero-flux configuration (via a mixed
  Dirichlet/Neumann construction) is verified working. This does not affect
  Stage 6, which always uses Dirichlet.
- `solve_newton`'s return signature is `(C, history, method)` — a 3-tuple, not
  2. All in-repo call sites are updated; any new caller must unpack three
  values.
- The 2D solver has **no relaxation/PTC fallback** (there is no 2D PTC solver);
  a degenerate solve returns `method="newton_stalled"` and the achieved
  residual rather than being silently rescued. Callers should inspect
  `elliptic_residual` rather than assuming every step converged.
- The 2D solver does **not** reproduce the sector formation of the source
  paper's Fig. 3/4. Diagnosed as a setup/parameter-regime consequence, not a
  solver defect: the domain and boundary conditions are angularly symmetric so
  angular modes have no forcing and only decay, and `Dhat ~ 4.2` gives a
  diffusion length exceeding the domain. Sector formation there arises from a
  sharp front expanding into empty space, a condition this setup never creates.
  This has not been checked against the paper's own Fig. 3/4 parameters.
- Parameter presets carry several documented, but unverified-against-source,
  cleaning assumptions (e.g. the `rebeca` yield rescale, `A_OVER_D_RATIO`);
  see the docstring in `params.py` for the full list.

## Reference

Freingruber, Gonzalez-Cabaleiro, Yoldas. *A slow-fast PDE model for
nitrifying bacteria in biofilms.* [arXiv:2512.13156](https://arxiv.org/abs/2512.13156).
