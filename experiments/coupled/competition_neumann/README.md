# Coupled (time-dependent substrate) -- 3-species competition, neumann regime

Same solver, physics, initial condition, eps = 25 (Lambda = 125), grid and
time step as `../competition_dirichlet/` (see its README for the
eps scan and the reasoning behind the choice); only the boundary regime on
the fed substrates NH4 and NO2 differs. O2 stays Dirichlet (non-limiting
co-substrate), NO3 keeps a Dirichlet-0 outlet.

The Neumann flux is the paper's literal `1e-5` divided by `D = 1/eps`
(the same D-correction the slow-fast folders use, where D = 0.2 gave
`-5e-5`); at eps = 25 that is `-2.5e-4`. Unlike the quasi-steady Neumann
runs, the substrate here starts from the paper's reservoir `c_0 = 5` and
is depleted over time rather than pinned at the tiny steady value the
flux alone can sustain.

```
python run_coupled_competition.py full    # 100x100 grid, T = 50
python render_coupled_competition.py
```

`run_coupled_competition.py scan` also works here (it reruns the eps scan
under this regime) but the documented scan is the Dirichlet one.

## Files

- `run_coupled_competition.py`, `render_coupled_competition.py` -- identical
  to the Dirichlet folder's apart from `REGIME`.
- `coupled_competition_neumann_bacteria.png`,
  `coupled_competition_neumann_substrates.png`,
  `coupled_competition_neumann_term_ratio.png` -- outputs of the full run.
