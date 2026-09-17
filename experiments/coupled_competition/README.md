# Coupled (time-dependent substrate) -- 3-species competition, Dirichlet

The slow-fast solver treats the substrates as quasi-steady (epsilon -> 0):
one timescale fully resolves before the other moves, which is what makes
the QSSA and its plausibility bound necessary. This folder runs the same
competition problem (Case A physics, 300-circle rough initial condition,
Dirichlet c_inf = 5) on the parallel solver in `nitrifiers/coupled/`, which
keeps `eps * dc/dt` and advances substrates and bacteria together with one
shared time step.

```
python run_coupled_competition.py scan    # ~12 min, 50x50 grid
python run_coupled_competition.py full    # 100x100 grid, T = 50
python render_coupled_competition.py
```

## Why eps and Lambda move together

In the nondimensional substrate equation `eps * c_t = Lap c - Lambda u M(c)`,
`eps = r L^2 / D` and `Lambda = r L^2 u_ref / (D c_ref Y)` both scale as `1/D`,
so `Lambda = eps / Y` (= `5 eps` here, Y = 0.2). The scan therefore varies the
physically consistent pair `(eps, 5 eps)`; holding Lambda fixed while
changing eps would not correspond to any single change of physical
parameters.

## Choosing eps

Two diagnostics are measured at every step and averaged over the run:

- `eps*|c_t| / |Lap c|` (volume-integrated): the size of the transient term
  relative to diffusion. `|R| / |Lap c|` is NOT used -- it is ~1 near any
  quasi-steady state by definition and says nothing about timescales.
- the coupled solution against the slow-fast (QSSA) solution on the same
  grid and dt, via total density at the domain centre at T = 10.

Scan results (50x50 grid, T = 10, every run 0 stalled steps, ~2 Newton
iterations per substrate step):

| eps | Lambda | eps*c_t / Lap c | coupled vs QSSA | dt 0.1 vs 0.05 |
|-----|--------|-----------------|-----------------|----------------|
| 1   | 5      | 0.013           | 0.00 %          | 0.02 %         |
| 5   | 25     | 0.059           | 0.09 %          | 0.05 %         |
| 20  | 100    | 0.169           | 1.0 %           | 0.26 %         |
| 25  | 125    | 0.190           | 1.1 %           | 0.29 %         |
| 50  | 250    | 0.267           | 2.0 %           | 0.26 %         |
| 100 | 500    | 0.378           | 9.5 %           | 0.04 %         |

The physical reading: the substrate relaxes over the colony scale `l ~ 0.2`
in a time `eps * l^2`, against a bacterial growth time `1 / r = 1`. The
substrate is slaved to the bacteria (QSSA valid) while `eps * l^2 << 1`, i.e.
`eps << 25`, and the two timescales are comparable at `eps ~ 25`. The
measured numbers follow that estimate.

**eps = 25 is used for the full run**: the crossover value, where the
transient is ~20 % of the diffusion term, the QSSA error crosses 1 %, and
dt = 0.1 is converged to 0.3 %. Two consequences worth stating plainly:

- The existing slow-fast runs at eps = 5 (`D = 0.2`) differ from the fully
  coupled solution by 0.09 %. At those parameters the quasi-steady
  approximation is not a source of disagreement with anything.
- The paper's own Case (A) parameters put eps ~ 1e4, far beyond the largest
  value scanned. At eps = 100 the coupled solver still runs cleanly and is
  already 10 % away from QSSA; reaching 1e4 is a question of run time
  (Lambda = 5e4 is stiff but the time-dependent step has no solvability
  problem), not of method.

## Files

- `run_coupled_competition.py` -- both modes; writes `results/` (not
  committed: `scan_summary.json`, per-snapshot `.npy`, `term_magnitudes.json`,
  `run_info.json`).
- `render_coupled_competition.py` -- bacteria and substrate grids every 5
  time units, plus the transient/diffusion ratio over time.
- `coupled_competition_bacteria.png`, `coupled_competition_substrates.png`,
  `coupled_competition_term_ratio.png` -- outputs of the full run.
