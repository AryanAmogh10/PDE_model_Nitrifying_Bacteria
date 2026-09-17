# Coupled (time-dependent substrate) -- commensalism, neumann regime

Same solver, physics, IC, `eps = 25` (`Lambda = 2.5`) as
`../commensalism_dirichlet/` (see its README); only the regime on
c1 differs. c1's Neumann flux is the paper's literal `1e-5` divided by
`D = 1/eps` (the same D-correction convention used throughout this
project); c2, c3 stay sealed regardless of regime.

The substrate has memory here: c1 starts from the paper's reservoir
`c1(x,0) = 5` and is depleted over the run rather than pinned at the tiny
value a steady-state Neumann solve would be forced into.

```
python run_coupled_commensalism.py
python render_coupled_commensalism.py
```

## Files

Identical to the dirichlet folder's apart from `REGIME`; outputs are named
`coupled_commensalism_neumann_*`.
