# Commensalism -- dirichlet regime

Self-contained: run these two scripts in order to regenerate the plots in
this folder from scratch.

```
python run_commensalism.py
python render_commensalism.py
```

`run_commensalism.py` solves the slow-fast PDE system and saves per-timestep
snapshots to `./results/` (not committed -- regenerated on demand).
`render_commensalism.py` reads those snapshots and produces
`commensalism_dirichlet_bacteria.png` and `commensalism_dirichlet_substrates.png` in this
same folder.
