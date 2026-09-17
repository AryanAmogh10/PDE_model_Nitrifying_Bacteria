# Coupled (time-dependent substrate) -- 3-species competition, robin regime

Same solver, physics, initial condition, eps = 25 (Lambda = 125), grid and
time step as `../competition_dirichlet/` (see its README for the
eps scan and the reasoning behind the choice); only the boundary regime on
the fed substrates NH4 and NO2 differs. O2 stays Dirichlet (non-limiting
co-substrate), NO3 keeps a Dirichlet-0 outlet.

Robin on the fed substrates with `Bi = 5`, `c_inf = 5`, the same pair
used in the slow-fast Robin folder, so the two are directly comparable.

```
python run_coupled_competition.py full    # 100x100 grid, T = 50
python render_coupled_competition.py
```

`run_coupled_competition.py scan` also works here (it reruns the eps scan
under this regime) but the documented scan is the Dirichlet one.

## Files

- `run_coupled_competition.py`, `render_coupled_competition.py` -- identical
  to the Dirichlet folder's apart from `REGIME`.
- `coupled_competition_robin_bacteria.png`,
  `coupled_competition_robin_substrates.png`,
  `coupled_competition_robin_term_ratio.png` -- outputs of the full run.
