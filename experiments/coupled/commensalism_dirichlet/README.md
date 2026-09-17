# Coupled (time-dependent substrate) -- commensalism, dirichlet regime

Same custom chain solver as `../../commensalism_dirichlet/` (the slow-fast
version; paper eq. 2.4:
u1 consumes c1, produces c2; u2 consumes c2, produces c3; u3 consumes c3),
same rough IC, same regime convention (regime applies to c1 only; c2, c3
stay sealed). Here the substrate Newton solve is one backward-Euler step
from the previous field, `eps*dc/dt = Lap c + R`, with no plausibility
bound -- the mass term makes every step solvable.

`eps` and `Lambda` move together (`Lambda = eps / Y`). Case (C)'s yield is
`Y = 10` (vs. competition's `Y = 0.2`), so at the same `eps = 25` used for
`coupled_competition_dirichlet`, `Lambda = 2.5` here -- a much weaker
reaction. `c1(x, 0) = 5` (the paper's `c_0`); `c2, c3` start at 0, matching
"only the first substrate is initially supplied."

```
python run_coupled_commensalism.py
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
