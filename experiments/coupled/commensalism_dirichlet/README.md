# Coupled (time-dependent substrate) -- commensalism, dirichlet regime

Same custom chain solver as `../../commensalism_dirichlet/` (the slow-fast
version; paper eq. 2.4:
u1 consumes c1, produces c2; u2 consumes c2, produces c3; u3 consumes c3),
same rough IC, same regime convention (regime applies to c1 only; c2, c3
stay sealed). Here the substrate Newton solve is one backward-Euler step
from the previous field, `eps*dc/dt = Lap c + R`, with no plausibility
bound -- the mass term makes every step solvable.

`eps` and `Lambda` move together (`Lambda = eps / Y`). Case (C)'s yield is
`Y = 10` (vs. competition's `Y = 0.2`), so at the same `eps = 25`,
`Lambda = 2.5` here -- a much weaker reaction. `c1(x, 0) = 5` (the paper's
`c_0`); `c2, c3` start at 0, matching "only the first substrate is
initially supplied."

## Choosing eps: run independently, not just carried over

`Lambda` scales as `1/Y`, and Case (C)'s `Y = 10` is 50x competition's
`Y = 0.2`, so the two systems do not share a single "crossover eps" a
priori -- `eps = 25` needed its own check here, not an assumption that
competition's scan transfers. `python run_coupled_commensalism.py scan`
runs that check (49x49 grid, T = 10):

| eps | Lambda | eps*c_t / Lap c | coupled vs QSSA | dt 0.1 vs 0.05 |
|-----|--------|-----------------|------------------|----------------|
| 1   | 0.10   | 0.31            | 0.44 %           | 0.02 %         |
| 5   | 0.50   | 0.37            | 0.59 %           | 0.03 %         |
| 20  | 2.00   | 0.51            | 1.05 %           | 0.03 %         |
| 25  | 2.50   | 0.54            | 1.18 %           | 0.03 %         |
| 50  | 5.00   | 0.68            | 1.55 %           | 0.03 %         |
| 100 | 10.00  | 0.86            | 1.31 %           | 0.04 %         |

Two things to read correctly here:

- The `eps*c_t / Lap c` column is already ~0.3 at `eps = 1`, far higher
  than competition showed at the same `eps` (~0.01). This is **not**
  evidence QSSA fails at low `eps` -- it is a startup-transient artifact:
  `c2` and `c3` are byproducts starting at exactly 0, so they have a
  genuine, `eps`-independent transient while the colony first establishes
  the production chain. It says nothing about the eps-dependent crossover.
- The metric that actually matters -- coupled solution vs. the quasi-steady
  (QSSA) solution -- stays under 2 % across the entire scanned range, and
  at `eps = 25` gives **1.18 %**, almost identical to competition's
  **1.06 %** at the same `eps`. So `eps = 25` is independently justified
  for commensalism by this scan, not merely reused from competition's.
  (Pushing `eps` far higher, toward where `Lambda` would reach
  competition's ~125, was not necessary given this agreement.)

```
python run_coupled_commensalism.py scan   # ~4 min
python run_coupled_commensalism.py full   # full 100x100 run, T = 50
python render_coupled_commensalism.py
```

## Files

- `run_coupled_commensalism.py` -- solves and writes `results/` (not
  committed): per-time-unit `.npy` snapshots, `boundary_mass_history.npy`,
  `term_magnitudes.json`, `run_info.json`.
- `render_coupled_commensalism.py` -- bacteria and substrate grids every 5
  time units, plus the transient/diffusion ratio over time.
- `coupled_commensalism_dirichlet_bacteria.png`,
  `..._substrates.png`, `..._term_ratio.png` -- outputs of the run.
