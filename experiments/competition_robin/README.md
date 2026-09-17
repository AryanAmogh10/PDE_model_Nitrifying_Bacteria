# Competition -- robin regime

Self-contained: run these two scripts in order to regenerate the plots in
this folder from scratch.

```
python run_competition.py
python render_competition.py
```

`run_competition.py` solves the slow-fast PDE system and saves per-timestep
snapshots to `./results/` (not committed -- regenerated on demand).
`render_competition.py` reads those snapshots and produces
`competition_robin_bacteria.png` and `competition_robin_substrates.png` in this
same folder.
