# PDE Model for Nitrifying Bacteria in Biofilms

A numerical PDE solver for a slow-fast reaction-diffusion-advection system
modelling three competing nitrifying bacterial species (AOB, NOB, comammox
CMX) growing in a biofilm/granule, based on the model in
[arXiv:2512.13156](https://arxiv.org/abs/2512.13156) (Freingruber,
Gonzalez-Cabaleiro, Yoldas).

## Background

Nitrification is the two-step oxidation of ammonia to nitrite and then
nitrate, carried out by distinct microbial guilds in engineered systems like
wastewater treatment plants and in natural soils. Historically this was
attributed to two separate functional groups: ammonia-oxidising bacteria
(AOB, ammonia -> nitrite) and nitrite-oxidising bacteria (NOB, nitrite ->
nitrate). The more recent discovery of comammox (CMX) organisms -- a single
species capable of the complete ammonia-to-nitrate pathway on its own --
reframed nitrification as a three-way competition for overlapping substrate
niches rather than a fixed two-step division of labour, and motivated
mathematical models (including the one this codebase implements) that ask
how these three guilds spatially segregate and compete within a biofilm or
granule when they share diffusing substrates.

Individual-based models (IBMs) have previously reproduced the spatial
segregation ("columned stratification") this competition produces, but IBMs
are computationally expensive and don't easily yield continuum-level
analysis. The source paper derives a continuum PDE alternative -- coupling
slow bacterial density growth/movement to fast, near-instantaneously
equilibrating substrate diffusion -- and shows it reproduces the same
qualitative segregation patterns. This repository is a from-scratch
reimplementation and numerical verification of that PDE model, built for a
thesis project: it exists both to independently confirm the paper's claims
and to explore how the model's behaviour depends on assumptions (boundary
conditions, initial biomass configuration, parameter regime) that the paper
itself doesn't exhaustively test.

## Model

Three bacterial densities `u_AOB, u_NOB, u_CMX` (slow, parabolic) are coupled
to four substrate concentrations `c_NH4, c_NO2, c_NO3, c_O2` (fast,
quasi-steady/elliptic):

```
d/dt u_i = div(d_i grad(u_i) + a_i u_i grad(rho)) + u_i f_i(u_i, c)      (slow)
0        = D_j Delta(c_j) + c_j g_j(u, c)                                 (fast, quasi-steady)

rho = u_AOB + u_NOB + u_CMX
f_i(u_i, c) = r_i * M(c_p; K_ip) * M(c_s; K_is) - b_i * rho     (Monod growth, density-dependent death)
```

The scale separation between fast substrate diffusion and slow bacterial
growth (`epsilon = r_max * L^2 / D_j << 1`, checked per parameter preset)
justifies treating the substrate system as quasi-steady at each slow time
step: rather than integrating the substrate PDE forward in time (which would
force a prohibitively small time step), it's solved as a pure boundary-value
problem at every slow step, using whatever the current bacterial density
happens to be.

Two spatial versions of this same idea are implemented: a 1D radial/slab
solver (`nitrifiers/one_dimensional/`) and a 2D Cartesian solver
(`nitrifiers/two_dimensional/`). Both reuse the same reaction/chemistry code
and the same overall Newton-solver design; what differs is the grid and the
boundary-condition bookkeeping.

## Package layout

```
nitrifiers/
    params.py               dimensional parameter presets
    nondim.py                non-dimensionalisation, builds solver coefficients
    one_dimensional/
        elliptic.py           substrate solver (Newton + Picard)
        relaxation.py          pseudo-transient-continuation fallback/cross-check
        parabolic.py            bacterial density solver
        slowfast.py              couples the above into the full slow-time loop
    two_dimensional/
        grid2d.py               2D Cartesian grid + boundary machinery
        elliptic2d.py            2D substrate solver
        parabolic2d.py            2D bacterial density solver
        slowfast2d.py              2D slow-time loop
    coupled/
        substrate_step2d.py     time-dependent substrate step (epsilon kept)
        coupled2d.py             fully coupled loop, substrates + bacteria share dt
experiments/
    <case>_<regime>/            self-contained replication folders (slow-fast)
    coupled/
        competition_<regime>/    coupled solver: eps scan (dirichlet) + full runs
        commensalism_<regime>/    coupled solver, commensalism chain
```

`params.py` and `nondim.py` sit at the top level because both the 1D and 2D
solvers use them; everything dimension-specific lives in its own
subpackage. (Named `one_dimensional`/`two_dimensional` rather than `1d`/`2d`
because Python module names can't start with a digit.)

### Parameter presets (`params.py`)

- **`toy`** -- round numbers, for solver development and unit tests.
- **`rebeca`** -- rough estimate from `parameters_nitrifiers_clean.json`.
- **`eloi`** -- thermodynamics/literature-derived kinetics (Martinez-Rabert et al.).

`load_preset(name)` returns a preset's raw dimensional numbers (growth
rates, decay rates, Monod constants, yields, diffusivities). `nondim.py`
turns that into the dimensionless coefficients every solver actually
consumes -- in particular `elliptic_coefficients(preset_name)`, which builds
the `coeffs` dict (`Lambda`, `LambdaProd`, `Khat`, `rhat`, `bhat`, `Dhat`,
`Ahat`, `c_inf_hat`, `beta`) you'll see passed into every solver function
below.

## What each file does

### `nitrifiers/one_dimensional/elliptic.py` -- the substrate solver

- `monod(c, K)` / `dmonod(c, K)` -- the Monod saturation function `c/(K+c)`
  and its derivative.
- `Grid` -- the 1D mesh: node positions and a geometry exponent `p` (0 =
  slab, 1 = cylindrical, 2 = spherical).
- `face_area`, `cell_volumes` -- exact face areas and control-volume sizes
  for the finite-volume discretisation; every conservative operator in this
  file and in `parabolic.py` is normalised against the same `cell_volumes`.
- `build_laplacian(grid)` -- assembles the diffusion operator matrix.
- `apply_bc`, `normalize_bc_specs` -- wire a Dirichlet (fixed value),
  Neumann (fixed flux) or Robin (flux proportional to the gap between the
  surface and a bulk value, `-dc/dr = Bi*(c - c_inf)`) condition into the
  outer boundary row, per substrate. Robin's value is a `(Bi, c_inf)` pair.
- `_default_initial_guess` -- starting concentration guess for Newton
  (Neumann substrates start at `1.0`, not `0.0`, so the reaction term isn't
  dead at the first iterate; Robin starts at its own bulk `c_inf`).
- `reaction_and_jacobian(C, U, coeffs)` -- the chemistry: how much of each
  substrate is produced/consumed given current concentrations `C` and
  bacterial densities `U`, plus the derivative Newton needs.
- `_assemble_global`, `_residual` -- build the full coupled system matrix and
  residual vector for all four substrates at once.
- `solve_newton(...)` -- the main entry point. Damped Newton with a
  backtracking line search and a physical-plausibility bound (rejects steps
  that would push a concentration to an absurd value); falls back to
  `relaxation.solve_relaxation` if it genuinely can't find a step.
- `solve_picard(...)` -- a slower fixed-point alternative, kept as an
  independent cross-check rather than a production solver.

### `nitrifiers/one_dimensional/relaxation.py` -- the fallback solver

- `solve_relaxation(...)` -- pseudo-transient continuation: marches toward
  the steady state gradually (via a fictitious pseudo-time-step that grows
  geometrically) instead of jumping there in one Newton step. Slower, more
  robust; used automatically when `solve_newton` stalls.
- `compare_with_elliptic(...)` -- runs both solvers on the same problem and
  reports how far apart their answers are, for cross-validation.

### `nitrifiers/one_dimensional/parabolic.py` -- the bacterial density solver

- `build_advection_matrix`, `build_advection_rho_jacobian` -- the
  "bacteria move away from crowding" (cross-diffusion) operator and its
  derivative with respect to total density.
- `growth_rate_field` -- the growth-only part of a species' reaction rate,
  from the current substrate field.
- `_parabolic_residual` -- the residual for one implicit backward-Euler time
  step.
- `solve_parabolic(...)` -- advances all three species forward by however
  many steps you ask for, with a full Newton solve at each step (including
  the cross-diffusion Jacobian term, needed for real convergence on sharp
  density fronts).
- `_total_mass` -- volume-weighted total biomass, used for diagnostics and
  conservation checks.

### `nitrifiers/one_dimensional/slowfast.py` -- the 1D driver

- `solve_c_given_u(...)` -- solves for substrate given a fixed bacterial
  state, trying Newton first and falling back to relaxation as an outer
  safety net.
- `run_slow_loop(...)` -- the function you actually call for a full 1D
  simulation: alternates solving for substrate and advancing bacteria for
  `n_slow_steps`, returning the final state plus a per-step history and
  periodic snapshots.

### `nitrifiers/two_dimensional/grid2d.py` -- the 2D spatial scaffolding

- `Grid2D` -- the Cartesian mesh: node coordinates, cell volumes (half-cells
  at the four edges), and which flattened node indices sit on the boundary.
  `.ravel`/`.unravel` convert between a 2D field and the flat array the
  solvers operate on; `.radius(x0, y0)` gives a distance-from-a-point field
  (used to seed colonies); `.integrate(field)` is a volume-weighted sum.
- `build_laplacian_2d(grid)` -- the 2D diffusion operator (5-point stencil).
  Edges are zero-flux by default, simply because there's no neighbouring
  cell there.
- `apply_bc_2d(...)` -- overwrites boundary rows with a fixed value when you
  want Dirichlet instead of the default zero-flux. Neumann and Robin leave
  the rows alone; their coupling enters through the source term below.
- `boundary_flux_coefficients(grid)` -- what makes a **nonzero**-flux (or
  Robin) boundary possible: a per-node weight you multiply by the boundary
  flux and add in as a source term at the edge.

### `nitrifiers/two_dimensional/elliptic2d.py` -- the 2D substrate solver

- `normalize_bc_specs_2d(...)` -- per-substrate boundary type/value, same
  idea as the 1D version; Neumann flux and Robin transfer are applied
  uniformly across all four edges (not per-edge).
- `_residual_2d`, `_assemble_global_2d` -- build the residual and system
  matrix, folding in the flux source term where relevant. For Robin the
  flux `Bi*(c - c_inf)` depends on `c`, so its linear part is also added to
  the Jacobian diagonal at boundary nodes -- without that Newton would still
  converge, just slower, which is why that term is checked against a
  finite-difference Jacobian in the tests rather than trusted on sight.
- `solve_newton_2d(...)` -- the 2D substrate solver. Same Newton +
  backtracking + plausibility-bound design as 1D, but with **no relaxation
  fallback** (there's no 2D pseudo-transient solver yet) -- a stuck solve
  just comes back with `method="newton_stalled"` rather than being silently
  rescued.

### `nitrifiers/two_dimensional/parabolic2d.py` -- the 2D bacterial solver

- `_face_index_arrays` -- bookkeeping helper: for every internal grid face,
  which two nodes it connects and their volume-normalisation factors.
- `build_advection_matrix_2d`, `build_advection_rho_jacobian_2d` -- 2D
  versions of the crowding-driven movement operator and its derivative.
- `total_mass_2d` -- 2D version of `_total_mass`.
- `solve_parabolic_2d(...)` -- advances all three species forward on the 2D
  grid, same fully-implicit Newton scheme as the 1D solver.

### `nitrifiers/two_dimensional/slowfast2d.py` -- the 2D driver

- `solve_c_given_u_2d(...)` -- solves for substrate given a fixed 2D
  bacterial state (Newton only, no fallback layer).
- `run_slow_loop_2d(...)` -- the 2D equivalent of `run_slow_loop`: alternate
  substrate-solve and bacteria-advance for `n_slow_steps`.

### `nitrifiers/coupled/` -- the time-dependent alternative (epsilon kept)

The slow-fast solvers above drop `d(c)/dt` (epsilon -> 0). This package
keeps it and advances substrates and bacteria together with one shared
time step:

```
eps * d/dt c_j = Delta(c_j) + R_j(u, c)        eps = r_max * L^2 / D_j
```

- `substrate_step2d.py::step_substrate_2d(...)` -- one backward-Euler step
  of the substrate system, Newton-solved. It reuses `elliptic2d`'s residual
  and Jacobian and adds only the `eps/dt` mass term. There is deliberately
  **no plausibility bound** here: the mass term makes every step solvable,
  so a boundary influx larger than the consumption capacity accumulates
  substrate over time instead of leaving Newton chasing a steady state that
  does not exist -- the failure mode that forced the bound in the
  quasi-steady solver.
- `substrate_step2d.py::term_magnitudes(...)` -- volume-integrated size of
  the three terms `eps*c_t`, `Delta c`, `R`. Note `|R| / |Delta c|` is ~1
  near any quasi-steady state and says nothing about timescales; the
  informative ratio is `eps*|c_t| / |Delta c|`.
- `coupled2d.py::run_coupled_2d(...)` -- the loop (Lie splitting: substrate
  step with the current `u`, then the existing implicit bacterial step with
  the new `c`), with the same snapshot/diagnostic hooks as the slow-fast
  driver.

`eps` and `Lambda` are not independent knobs: both scale as `1/D`, so in
this project's units `Lambda = eps / Y`. Changing one without the other is
not a physically consistent parameter change.

## Numerics notes

- Every finite-volume operator uses **exact** control-volume weighting,
  including at the domain centre (`r=0` in 1D) and along the domain edges
  (2D) -- the 1D scheme is verified 2nd-order accurate via closed-form
  (cosh/Bessel/sinh) benchmarks, cross-checked against a 50-digit `mpmath`
  reference to rule out floating-point artifacts at fine grids.
- **1D Neumann support**: both `bc_type="dirichlet"` and `"neumann"` work,
  including nonzero flux, verified against closed-form linear-reaction
  solutions. A genuinely coupled multi-substrate nonzero-flux case (more than
  one substrate on flux boundaries at once) is regression-tested
  (`test_coupled_multi_substrate_neumann_converges`) and converges cleanly at
  moderate reaction-rate scales; at the `eloi` preset's realistic stiffness
  that same configuration does not converge, confirmed to be a genuinely
  near-singular Jacobian there (condition number in the `1e16`-`1e17` range,
  varies by exact stall state but always far past any well-conditioned
  threshold), not a solver defect -- see
  `test_eloi_stiffness_coupled_neumann_is_near_singular` in the same file,
  which makes this finding independently reproducible via `pytest` rather
  than a one-off interactive result.
- **2D Neumann support**: also works, including nonzero flux (uniform across
  all four edges), verified the same way -- exact global mass-balance check
  plus first-order convergence of the recovered boundary flux to the
  prescribed value under grid refinement. Not carried over from 1D: there is
  no 2D relaxation/PTC fallback, so a degenerate 2D solve reports
  `method="newton_stalled"` rather than being rescued.
- **Robin support (1D and 2D)**: `-dc/dn = Bi*(c - c_inf)`, a mass-transfer
  condition where the flux into the biofilm is proportional to how far the
  surface concentration sits below the bulk value. Specified per substrate as
  `("robin", (Bi, c_inf))`. It is written so that it degenerates exactly:
  `Bi=0` is bit-identical to zero-flux Neumann (the same discrete row), and
  as `Bi -> inf` the solution approaches Dirichlet `c=c_inf` with the gap
  shrinking as `1/Bi` (measured: `6.5e-3 -> 6.5e-5 -> 6.5e-7` for
  `Bi = 1e2, 1e4, 1e6`). Validation, 1D
  (`test_robin_matches_closed_form_and_degenerates_correctly`): closed-form
  linear-reaction solutions in all three geometries, error at the same
  first-order magnitude as the Neumann row and converging at rate 1.00 under
  refinement; the Robin relation recomputed from the converged profile holds
  to machine precision (`~3e-14`); Picard agrees as an independent solver.
  Validation, 2D (`test_2d_robin_jacobian_matches_finite_difference`,
  `test_2d_robin_degenerate_limits_and_mass_balance`): the analytic Jacobian
  matches a finite-difference Jacobian to `6e-11` relative -- this is the
  decisive check, since a wrong Robin diagonal term would still converge as
  a modified Newton -- and integrated Robin influx balances integrated
  consumption to `3e-15`. One practical note: a very large `Bi` is a stiff,
  penalty-like limit whose absolute residual floor scales with `Bi`, so at a
  tight tolerance the solver may honestly report `newton_stalled` while
  still being correct (residual `~1e-9` at `Bi=1e6`, down from `~1`). If
  you want the Dirichlet limit, use Dirichlet directly.
- **Shared volume measure**: every conservative operator in a given
  dimension (Laplacian, advection, mass diagnostic) is normalised by the
  *same* exact control volumes. This matters -- a finite-volume operator
  normalised by one measure but integrated against another doesn't conserve
  anything, even if each operator looks fine in isolation.
- `solve_newton` (1D and 2D) returns a 3-tuple `(C, history, method)`, not a
  pair -- `method` tells you honestly whether the result came from plain
  Newton, an inner relaxation fallback, or (one level up, in
  `slowfast.py::solve_c_given_u`) an outer relaxation backstop.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, NumPy, SciPy.

## Usage

1D:

```python
from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.one_dimensional.elliptic import Grid
from nitrifiers.one_dimensional.slowfast import run_slow_loop
import numpy as np

grid = Grid(N=60, geometry="radial", p=2)  # spherical granule
r = grid.r
U0 = {sp: 0.02 * np.exp(-((r - 0.5) / 0.15) ** 2) for sp in ("AOB", "NOB", "CMX")}

U, C, history, _ = run_slow_loop("eloi", grid, U0, n_slow_steps=100, dt_slow=0.05)
```

2D:

```python
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.two_dimensional.slowfast2d import run_slow_loop_2d
import numpy as np

grid = Grid2D(Nx=48, Ny=48, Lx=1.0, Ly=1.0)
d = grid.radius()
U0 = {sp: 0.02 * np.exp(-((d - 0.3) / 0.1) ** 2) for sp in ("AOB", "NOB", "CMX")}

U, C, history, _ = run_slow_loop_2d("eloi", grid, U0, n_slow_steps=100, dt_slow=0.05)
```

To try a scenario that isn't one of the three presets, build a `coeffs` dict
by hand (same keys `elliptic_coefficients` returns) and pass it straight to
`solve_newton_2d`/`solve_parabolic_2d` in your own loop instead of
`run_slow_loop_2d`.

Per-substrate boundary conditions, mixing the three types (works the same
way in 1D via `solve_newton`):

```python
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d

bc_specs = {
    "NH4": ("robin",     (2.0, 1.0)),   # -dc/dn = Bi*(c - c_inf), Bi=2, bulk c_inf=1
    "O2":  ("dirichlet", 0.1876),       # held at a fixed value
    "NO2": ("neumann",   0.0),          # sealed (zero flux)
    "NO3": ("neumann",   0.0),
}
C, history, method = solve_newton_2d(coeffs, U, grid, bc_specs=bc_specs)
```

## Tests

```bash
pytest tests/
```

Covers: closed-form ground-truth convergence for the 1D elliptic solver
(slab/cylindrical/spherical; Dirichlet, nonzero-flux Neumann, and Robin,
including Robin's exact degeneration to the other two),
mass-conservation checks in 1D and 2D, a finite-difference check of the 2D
Robin Jacobian, Newton-vs-relaxation cross-validation
across presets/densities, analytic-equilibrium and Jacobian-vs-finite-
difference checks for the parabolic solvers, 2D-to-1D reduction checks
(exact slab reduction, convergent cylindrical reduction), anoxic-core
zonation reproduction, a genuinely-coupled multi-substrate Neumann case at
moderate stiffness (converges), and the same case at `eloi`'s realistic
stiffness (confirmed non-convergent with a near-singular Jacobian, not a
solver bug).

## Known limitations

- The Picard solver in `elliptic.py` requires a center-row workaround to
  converge and is not a drop-in replacement for Newton under the `eloi`
  preset's reaction stiffness -- a property of the fixed-point map, not the
  spatial discretisation.
- The 1D and 2D solvers both accept a per-substrate `bc_specs = {sub:
  (bc_type, value)}` dict so each substrate can independently be Dirichlet,
  Neumann or Robin (value `(Bi, c_inf)`); the older global `bc_type`
  parameter still works for Dirichlet/Neumann and is expanded into the
  equivalent per-substrate spec. Robin has no global-`bc_type` form, since
  there is no `Bi` in the coefficient set to fall back on -- it must be given
  through `bc_specs`. Coupled multi-substrate
  Neumann does not converge at `eloi`-realistic stiffness (see Numerics
  notes above, and `tests/test_elliptic_closed_form.py::test_eloi_stiffness_coupled_neumann_is_near_singular`
  for the reproducible regression test) -- an open numerical-conditioning
  issue, not a missing feature.
- The 2D solver has no relaxation/PTC fallback; a stuck solve reports
  `method="newton_stalled"` with the achieved residual rather than being
  silently rescued.
- The 2D solver reproduces angular sector formation when run with the
  source paper's own Table 1 Case (A) parameters, but that reproduction uses
  this project's own Dirichlet/QSSA machinery rather than the paper's actual
  time-dependent, low-flux Neumann boundary -- and the paper's own Case (A)
  parameters place `epsilon = r*L^2/D` at `~1e4`, well outside the
  small-epsilon regime the whole quasi-steady approach assumes. The
  diffusion-length threshold governing the sector-vs-uniform transition is a
  real, resolution-independent effect, but its precise mechanism (most
  likely an angular pattern-selection effect) has not been pinned down with
  a linear-stability analysis.
- Parameter presets carry documented cleaning assumptions in `params.py`.
  `A_OVER_D_RATIO = 10` matches the `a_i/d_i` ratio used in the source
  paper's own Table 1 Case (A). The `rebeca` preset's /100 yield rescale is
  a plausible units-slip correction, not independently confirmed against the
  original source, and it isn't cosmetic: it determines whether that
  preset's slow-loop run develops an anoxic core at all, while leaving
  solver convergence, mass-growth direction, and species dominance
  unaffected.

## Project status

The 2D solver reproduces the source paper's Figure 3 (Case A: three-species
competition) qualitatively, once the initial condition is corrected to match
the paper's own setup: a packed-density (peak = 1.0, the model's
nondimensional reference scale), spatially speckled inoculum confined to a
central circular region -- not a smooth, dilute field. `dirichlet_plot.png`,
`neumann_plot.png`, and `robin_plot.png` at the repo root show this
side-by-side against the paper's actual Figure 3, for each of the three
boundary condition types this codebase supports.

The substrate solver itself remains a deliberate departure from the paper:
this codebase solves substrate as quasi-steady (elliptic) at every bacterial
time step, while the paper integrates it forward in time alongside the
bacteria. That choice, and the diffusivity/consumption-rate rescaling it
forces (see Known limitations below), is the main open gap between this
implementation and a literal reproduction of the paper's own numerics --
tracked here rather than silently absorbed into the results.

Two solver systems now exist side by side and are kept deliberately
separate. The slow-fast (quasi-steady, `epsilon -> 0`) system is unchanged.
The coupled system in `nitrifiers/coupled/` keeps `eps * dc/dt` and steps
substrates and bacteria together; it needs no plausibility bound. An
epsilon scan on 3-species competition (`experiments/coupled/competition_dirichlet/`)
measured the transient term against diffusion and the coupled solution
against QSSA: at the existing `eps = 5` the two agree to 0.09 %, the
crossover where the substrate's own dynamics matter is `eps ~ 25`
(`eps * l^2 ~ 1` for the colony scale `l ~ 0.2`), and at `eps = 100` they
differ by ~10 %. `eps = 25` is the value used for the full coupled run.

At a glance, what's independently verified against the paper versus what's
still an open, flagged deviation:

- **Verified against the paper**: initial-condition geometry and magnitude
  for Case (A) (checked directly against the paper's own Figure 3 image);
  `a_i/d_i` ratio; growth rate, half-saturation, and yield values from Table
  1 Case (A); qualitative sectorial/segregation pattern formation.
- **Deliberate, flagged deviations**: quasi-steady vs. time-dependent
  substrate; substrate diffusivity `D` (raised from the paper's `1e-4` to
  `0.2`, to keep the quasi-steady assumption's small-`epsilon` regime valid)
  and the resulting ~2000x change in the consumption-rate coefficient
  `Lambda`; boundary condition type used for the closest qualitative match
  (Dirichlet) versus the paper's literal time-dependent low-flux Neumann
  condition.
- **Why this distinction matters**: a quasi-steady Neumann substrate
  solve has no built-in mechanism to prevent concentration blow-up when
  prescribed boundary influx exceeds the domain's total consumption
  capacity -- the paper's own time-dependent equation doesn't have this
  failure mode, because excess substrate simply accumulates over time
  instead of requiring an instantaneous steady state to exist. This
  codebase's plausibility bound (see `solve_newton`/`solve_newton_2d` above)
  is the practical consequence of that design choice, not a numerical
  workaround for a bug.

## Citing this work

See [`CITATION.cff`](CITATION.cff). If you use this code, please also cite
the source model paper below. **No `LICENSE` file exists in this repository
yet** -- `CITATION.cff` intentionally omits a `license` field rather than
guessing one; add a license file and update `CITATION.cff` before treating
any release as reusable by others under specific terms.

## Reference

Freingruber, Gonzalez-Cabaleiro, Yoldas. *A slow-fast PDE model for
nitrifying bacteria in biofilms.* [arXiv:2512.13156](https://arxiv.org/abs/2512.13156).
