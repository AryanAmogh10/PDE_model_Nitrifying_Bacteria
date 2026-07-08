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
- **Stage 4 (relaxation)**: pseudo-transient continuation, used as an
  independent cross-check of Stage 3 (agreement to ~1e-11 across all three
  presets and a range of densities).
- **Stage 5 (parabolic)**: fully-implicit backward Euler with the complete
  Newton Jacobian, including the cross-diffusion `d(Adv(rho))/d(rho)` term
  (needed for clean convergence on sharp density fronts).

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
elliptic solver (slab/cylindrical/spherical geometries), mass-conservation
checks, Newton-vs-relaxation cross-validation across presets/densities,
analytic-equilibrium and Jacobian-vs-finite-difference checks for the
parabolic solver, and anoxic-core zonation reproduction.

## Known limitations

- The Picard solver in `elliptic.py` requires a center-row RHS workaround to
  converge and is not a drop-in replacement for Newton under the `eloi`
  preset's reaction stiffness; this is a property of the fixed-point map,
  not the spatial discretization.
- Parameter presets carry several documented, but unverified-against-source,
  cleaning assumptions (e.g. the `rebeca` yield rescale, `A_OVER_D_RATIO`);
  see the docstring in `params.py` for the full list.

## Reference

Freingruber, Gonzalez-Cabaleiro, Yoldas. *A slow-fast PDE model for
nitrifying bacteria in biofilms.* [arXiv:2512.13156](https://arxiv.org/abs/2512.13156).
