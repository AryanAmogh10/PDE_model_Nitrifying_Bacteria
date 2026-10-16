# AOB-comammox competition: simulation experiments

Question: how do ammonium and oxygen availability influence competition
between AOB and comammox (CMX) in a spatially growing bacterial community?

The work follows the plan in `simulation_experiments.pdf`.

## Current setup: base case and heatmap

`common.py` holds the shared setup; `base_case/` and `heatmap/` use it. They
supersede Step 1 (further below), which started AOB with half the mass of CMX
and used O2_BC = 5 nondimensional = 2500 uM, about 10x air saturation.

- Preset `eloi` (CMX ~22x higher NH4 affinity, AOB ~14x faster growth).
  `rebeca` gives every species the same affinity, so it cannot show the
  affinity-driven AOB/CMX transition.
- Every species starts with the same mass M0 = 4.75 ug per mm depth (the
  300-circle pattern, each species rescaled). The isolated systems are the
  combined system with species removed, so differences between isolated and
  combined runs come only from competition.
- Physical units: time in days (1 solver unit = 1/r_max = 1.05 days), space
  in um (L = 500 um), concentrations in uM (c_ref = 500 uM), biomass in ug per
  mm depth (the model is 2D), substrate amounts in umol per mm depth.
- Dirichlet substrate boundaries (NO2 = NO3 = 0), Case A motility, survival
  cutoff u < 1e-6, warm-started quasi-steady substrate solves, dt = 0.5.
- Each run stops just before biomass reaches the domain boundary, or at
  60 solver units (63 days).
- F_CMX = M_CMX / (M_AOB + M_CMX) (NOB ignored); it starts at exactly 0.5.

### Base case (NH4_BC = 100 uM, O2_BC = 250 uM, 100x100 grid)

Compared at 56.4 days, before any system reaches the boundary. All three
runs: 0 Newton stalls. Values in ug per mm depth (biomass) and umol per mm
depth (amounts).

| | combined | AOB system | CMX system |
|---|---|---|---|
| AOB biomass | 10.19 | 10.68 | - |
| NOB biomass | 5.52 | 6.28 | - |
| CMX biomass | 3.88 | - | 12.02 |
| NO3 produced | 8.44 | 8.40 | 5.60 |
| NH4 consumed | 11.78 | 11.74 | 5.60 |
| O2 consumed | 21.88 | 21.81 | 11.21 |
| NO2 lost through the boundary | 3.33 | 3.34 | 0 |

- CMX in the combined system ends at 32 % of its isolated biomass (3.88
  against 12.02) and below its initial mass; AOB keeps 95 % of its isolated
  biomass. F_CMX rises to 0.57 at ~10 days (AOB declines in its lag phase),
  then falls to 0.28 once AOB takes off at the colony rim.
- Nitrification in the combined system runs almost entirely through AOB+NOB:
  its NO3, NH4, O2 and NO2 figures are within 0.5 % of the AOB system's.

Plots: `base_case/base_case_comparison.png` and
`base_case/base_case_{all,aob,cmx}_fields.png`; numbers in
`base_case/base_case_summary.json`.

### Heatmap (100x100 grid)

NH4_BC = 0.5 to 1000 uM (9 values, log-spaced), O2_BC = 5, 15, 40, 100,
250 uM; one combined-system run and one CMX-alone run per point (90 runs,
0 Newton stalls). Most runs never reach the boundary and stop at 63 days;
only O2 = 250 uM with NH4 >= 150 uM touches it (at 50-54 days). Rows are
O2_BC [uM], columns NH4_BC [uM]. Plots: `heatmap/heatmap_F_cmx.png`,
`heatmap/heatmap_growth.png`, `heatmap/heatmap_resolution.png`; all numbers in
`heatmap/heatmap_summary.json`.

F_CMX = M_CMX / (M_AOB + M_CMX) at stop time (starts at 0.5):

| O2 \ NH4 | 0.5 | 1.3 | 3.3 | 8.6 | 22 | 58 | 150 | 387 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| 250 | 0.90 | 0.89 | 0.87 | 0.79 | 0.62 | 0.35 | 0.23 | 0.20 | 0.20 |
| 100 | 0.90 | 0.89 | 0.87 | 0.80 | 0.62 | 0.41 | 0.33 | 0.32 | 0.32 |
| 40  | 0.90 | 0.89 | 0.87 | 0.80 | 0.68 | 0.60 | 0.59 | 0.59 | 0.59 |
| 15  | 0.90 | 0.89 | 0.88 | 0.83 | 0.80 | 0.79 | 0.79 | 0.79 | 0.79 |
| 5   | 0.90 | 0.89 | 0.88 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 |

Transition (F_CMX = 0.5): NH4_BC = 39 uM at O2 = 100 uM and 34 uM at
O2 = 250 uM; no transition at O2 <= 40 uM.

Growth factor M(stop) / M(0):

| row \ NH4 [uM] | 0.5 | 1.3 | 3.3 | 8.6 | 22 | 58 | 150 | 387 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| AOB, combined, O2 250 | 0.09 | 0.10 | 0.12 | 0.21 | 0.50 | 1.49 | 2.79 | 3.24 | 3.20 |
| AOB, combined, O2 100 | 0.09 | 0.10 | 0.12 | 0.21 | 0.48 | 1.14 | 1.64 | 1.68 | 1.69 |
| AOB, combined, O2 40  | 0.09 | 0.10 | 0.12 | 0.20 | 0.38 | 0.54 | 0.55 | 0.56 | 0.56 |
| CMX, combined, all O2 | 0.79-0.80 | 0.80 | 0.80-0.81 | 0.79-0.81 | 0.79-0.81 | 0.79-0.80 | 0.79-0.82 | 0.79-0.82 | 0.79-0.82 |
| CMX alone, O2 250 | 0.86 | 0.88 | 0.93 | 1.04 | 1.32 | 2.00 | 2.84 | 2.73 | 2.71 |
| CMX alone, O2 100 | 0.86 | 0.88 | 0.93 | 1.04 | 1.32 | 1.83 | 1.84 | 1.84 | 1.84 |
| CMX alone, O2 40  | 0.86 | 0.88 | 0.93 | 1.04 | 1.25 | 1.25 | 1.25 | 1.25 | 1.25 |
| CMX alone, O2 15  | 0.86 | 0.88 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CMX alone, O2 5   | 0.86 | 0.88 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |

(AOB at O2 = 15 and 5 uM: 0.09-0.21 everywhere. CMX alone is read at the
same time as the combined run.)

M_CMX(combined) / M_CMX(alone), the competitive loss of CMX:

| O2 \ NH4 | 0.5 | 1.3 | 3.3 | 8.6 | 22 | 58 | 150 | 387 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| 250 | 0.92 | 0.91 | 0.87 | 0.78 | 0.61 | 0.40 | 0.29 | 0.30 | 0.30 |
| 100 | 0.92 | 0.91 | 0.87 | 0.78 | 0.61 | 0.44 | 0.43 | 0.43 | 0.43 |
| 40  | 0.92 | 0.91 | 0.87 | 0.78 | 0.64 | 0.63 | 0.63 | 0.63 | 0.63 |
| 15  | 0.92 | 0.91 | 0.88 | 0.81 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 |
| 5   | 0.92 | 0.91 | 0.89 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |

What this shows:

- CMX never grows in the combined system (0.79-0.82 everywhere), although it
  grows alone wherever NH4 >= ~9 uM and O2 >= 40 uM (up to x2.8). AOB, at the
  colony rim and growing 14x faster, takes the NH4 and O2 before it reaches
  CMX.
- The competitive loss is largest where AOB grows best (CMX keeps only 29 %
  of its isolated biomass at high NH4 and O2 = 250 uM) and small at low NH4
  or low O2 (88-92 %), where the substrate supply itself limits CMX.
- F_CMX > 0.5 does not mean CMX wins. It marks where AOB collapses (down to
  0.09) faster than CMX, which decays 14x more slowly. Within these 63 days
  there is no condition under which CMX outgrows AOB in the combined system.
- Grid check (`heatmap_resolution.png`): the 50x50 sweep gives the same
  pattern, with F_CMX within 0.07 of the 100x100 values and transitions at
  36 uM for both O2 = 100 and 250 uM (against 39 and 34 here).

To reproduce: `python base_case/run_base_case.py`, then
`python base_case/render_base_case.py`; `python heatmap/run_heatmap.py all 99 8`,
`python heatmap/run_heatmap.py cmx 99 4` (and `all 49 8` for the grid check), then `python heatmap/render_heatmap.py 99`.

## Step 1 (superseded)

## Layout

```
simulation_experiments/
    README.md                    this file
    step1_isolation/
        run_isolation.py         one run: python run_isolation.py <preset> <aob|cmx> [N T dt NH4_BC O2_BC tag]
        render_isolation.py      plots for one preset: python render_isolation.py <rebeca|eloi>
        <preset>_comparison.png  M_i(t), cumulative consumption/production, relative biomass
        <preset>_aob_fields.png  u_AOB, u_NOB, NH4, NO2, NO3, O2 over time (AOB system)
        <preset>_cmx_fields.png  u_CMX, NH4, NO2, NO3, O2 over time (CMX system)
        results/                 raw snapshots and histories (not committed, regenerated by run_isolation.py)
```

To reproduce everything in Step 1:

```
cd step1_isolation
python run_isolation.py eloi   aob 99 50 0.5 1.0 5
python run_isolation.py eloi   cmx 99 50 0.5 1.0 5
python run_isolation.py rebeca aob 99 50 0.5 1.0 5
python run_isolation.py rebeca cmx 99 50 0.5 1.0 5
python render_isolation.py eloi
python render_isolation.py rebeca
```

## Setup (Step 1)

- Solver: the quasi-steady slow-fast solver (`nitrifiers.two_dimensional`),
  Dirichlet on every substrate: NH4 = NH4_BC = 1, O2 = O2_BC = 5,
  NO2 = NO3 = 0 on the boundary. Substrate fields start at the boundary values.
- Domain: unit square, 100 x 100 nodes, slow time step dt = 0.5, run to
  T = 50 or until biomass first touches the boundary, whichever is first.
- Initial biomass: the 300-small-circles-in-a-disc condition (disc radius
  0.2, circle radius 0.05, seed 42, total density 1 on the disc). The same 300
  circles are used in both systems, so initial total biomass is identical
  (M = 0.1141). AOB system: each circle is AOB or NOB (50/50). CMX system:
  every circle is CMX.
- Kinetics (growth, decay, Monod constants, yields, production) come
  unchanged from the `rebeca` and `eloi` presets.
- Bacterial motility is the paper's Case A (d = 1e-6, a = 1e-5), NOT the
  presets' values. The presets' motility is 2.7e6 (`rebeca`) to 4.2e6
  (`eloi`) times larger and spreads the colony over the whole domain within
  one step.
- Survival cutoff: u_i is set to 0 where u_i < 1e-6 after every step.
  Without it, diffusion seeds ~1e-40 biomass everywhere and that tail grows
  exponentially in the substrate-rich region outside the colony (in an early
  probe run without the cutoff, `eloi` held 17x more AOB outside the colony
  than inside it by t = 18). The cutoff
  removes ~1e-7 of biomass in total over a run, against a total of 0.1-0.4.
- Substrate solves are warm-started from the previous step
  (`solve_newton_2d(..., C_init=...)`), see the stall note below.
- M_i(t) is the volume-weighted integral of u_i over the domain.
  "Consumed" and "produced" are time integrals of the volume-weighted
  reaction rates in solver units; only ratios between systems are meaningful.

## Choice of O2_BC

Short scans on a 50 x 50 grid, T = 20, cold-started substrate solves
(growth factor = M_total(20) / M_total(0), M_total(0) = 0.1129):

| preset | O2_BC | AOB system growth | CMX system growth | CMX Newton stalls (of 40) |
|--------|-------|-------------------|-------------------|---------------------------|
| eloi   | 0.5   | x0.82 | x1.17 | 0 |
| eloi   | 2     | x1.84 | x1.89 | 6 |
| eloi   | 5     | x1.87 | x1.88 | 4 |
| eloi   | 20    | x1.88 | x1.89 | 7 |
| rebeca | 0.5   | x0.56 | x1.08 | 0 |
| rebeca | 2     | x0.77 | x1.11 | 0 |
| rebeca | 5     | x0.85 | x1.11 | 0 |
| rebeca | 20    | x0.85 | x1.11 | 0 |

Growth stops depending on O2_BC above ~2 (`eloi`) and ~5 (`rebeca`), so
O2_BC = 5 is used for all runs: oxygen is no longer limiting, as Step 1
asks ("sufficiently large"). At the `eloi` preset's own default (O2 = 0.19)
oxygen is fully depleted inside the colony.

## Results

All values below come from the final 100 x 100 runs. Both systems are
compared at t = 35.5, the last time before the `eloi` AOB system touches the
boundary (it does so at t = 36).

### Biomass at t = 35.5

Initial total biomass is 0.1141 in every run.

| preset | system | M_AOB | M_NOB | M_CMX | M_total | growth factor |
|--------|--------|-------|-------|-------|---------|---------------|
| rebeca | AOB (AOB+NOB) | 0.0849 | 0.0290 | 0 | 0.1140 | x1.00 |
| rebeca | CMX | 0 | 0 | 0.1466 | 0.1466 | x1.28 |
| eloi   | AOB (AOB+NOB) | 0.2420 | 0.1115 | 0 | 0.3535 | x3.10 |
| eloi   | CMX | 0 | 0 | 0.2789 | 0.2789 | x2.45 |

### Consumption and production up to t = 35.5 (solver units)

| preset | system | NH4 consumed | O2 consumed | O2 per NH4 | NO3 produced | net NO2 produced | (NO3 + NO2) / NH4 |
|--------|--------|--------------|-------------|------------|--------------|------------------|-------------------|
| rebeca | AOB | 263.5 | 701.5 | 2.66 | 262.9 | 0.6   | 1.000 |
| rebeca | CMX | 257.3 | 128.6 | 0.50 | 257.3 | 0.0   | 1.000 |
| eloi   | AOB | 455.3 | 772.6 | 1.70 | 299.6 | 168.3 | 1.028 |
| eloi   | CMX | 208.6 | 389.1 | 1.87 | 214.7 | 0.0   | 1.029 |

### Values at the end of each run

| preset | system | end time | M_total | growth factor | NH4 consumed | O2 consumed | boundary touched at |
|--------|--------|----------|---------|---------------|--------------|-------------|---------------------|
| rebeca | AOB | 50.0 | 0.1387 | x1.22 | 414.6 | 1104.0 | not touched |
| rebeca | CMX | 50.0 | 0.1727 | x1.51 | 397.0 | 198.5  | not touched |
| eloi   | AOB | 36.0 | 0.3620 | x3.17 | 471.3 | 797.6  | t = 36.0 |
| eloi   | CMX | 50.0 | 0.3417 | x3.00 | 312.5 | 582.9  | not touched |

Final species masses: `rebeca` AOB system M_AOB = 0.1066, M_NOB = 0.0321;
`eloi` AOB system M_AOB = 0.2493, M_NOB = 0.1127; CMX systems are all CMX.

### Solver diagnostics

| preset | system | Newton stalls | mean Newton iterations per step | biomass removed by cutoff |
|--------|--------|---------------|---------------------------------|---------------------------|
| rebeca | AOB | 0 | 5.0  | 7.5e-7 |
| rebeca | CMX | 0 | 4.0  | 3.7e-7 |
| eloi   | AOB | 0 | 48.1 (max 66) | 6.2e-7 |
| eloi   | CMX | 0 | 5.2  | 3.5e-7 |

### Resolution check

M_total at t = 20, 50 x 50 grid (O2_BC = 5, cold-started) against the
100 x 100 runs:

| preset | system | 50 x 50 | 100 x 100 | difference |
|--------|--------|---------|-----------|------------|
| rebeca | AOB | 0.0959 | 0.0963 | 0.4 % |
| rebeca | CMX | 0.1249 | 0.1256 | 0.6 % |
| eloi   | AOB | 0.2114 | 0.1969 | 7 % |
| eloi   | CMX | 0.2125 | 0.2105 | 1 % |

## Observations

- The two presets disagree on which system grows more. In `rebeca` the CMX
  system grows more (x1.28 against x1.00 at t = 35.5): NOB decays early in
  the AOB system (0.059 to 0.029) and the total only recovers slowly. In
  `eloi` the AOB system accelerates and overtakes the CMX system at about
  t = 25, while CMX grows steadily and slowly. This matches the parameters:
  `eloi` AOB grows 14 times faster than CMX (r = 1.0 against 0.073); CMX's
  advantage in `eloi` is its ~22 times higher NH4 affinity (K = 0.00013
  against 0.0029), which cannot show at NH4_BC = 1, where NH4 is far above K.
- Oxygen cost per NH4 oxidised is much lower for CMX in `rebeca` (0.50 against
  2.66, a factor 5.3) but about equal in `eloi` (1.87 against 1.70). The
  smaller total O2 use of the `eloi` CMX system (389 against 773) is only
  because it consumed less NH4 (209 against 455).
- NO2 accumulates only in the AOB system (net 168 in `eloi`, a band behind
  the AOB front): NOB cannot keep up with AOB. The CMX system makes no NO2.
- Stoichiometry holds: NO3 produced plus net NO2 produced equals NH4
  consumed exactly in `rebeca` and to within 3 % in `eloi`.
- Colony shape: AOB forms an expanding ring with a hollow centre, NOB stays
  behind it, CMX forms a compact disc with a denser rim. NH4 is depleted
  inside the colony and the depleted zone grows.

## Caveats and open points

- Single operating point. NH4_BC = 1 and O2_BC = 5 are far above the half-
  saturation constants (0.0001-0.007), so affinity differences are not
  expected to matter here. The sweep in Step 3 has to go to low
  concentrations, log-spaced.
- The `eloi` AOB system differs by 7 % between 50 x 50 and 100 x 100 at
  t = 20, against <= 1 % for the other three. It has the steepest substrate
  layers (it is also the slowest case, ~48 Newton iterations per step). Its
  100 x 100 result has not been checked against a finer grid, so treat the
  `eloi` AOB numbers as resolved to a few percent at best.
- Motility and the survival cutoff are modelling choices made here, not
  taken from the task description. They have not yet been agreed with the
  supervisor.
- Boundary-touch time is detected as the first step with nonzero biomass on
  a boundary node. With the survival cutoff this means biomass above 1e-6.
  The 17-21 step boundary-touch times reported for the Case A runs in
  `experiments/` were measured without such a cutoff and most likely reflect
  the exponential tail described above, not the colony front. This has not
  been rechecked for those runs.

## Substrate solver stalls (and the fix)

The first `eloi` CMX run (cold-started solves, 100 x 100) had 19 stalled
Newton solves out of 100, with residuals 60-90 against a tolerance of 1e-6
after only 3-6 iterations, and allowing more iterations or a larger
plausibility bound did not help. In those steps the substrate was not
solved and the bacteria were advanced on a wrong field, so that run was
discarded. The cause is that every solve restarted from a uniform guess,
which cannot cross the steep NH4 collapse inside the colony at the `eloi`
kinetics (Damkoehler numbers ~1e6). Warm-starting each solve from the
previous step's solution removed all stalls (0 of 100 in the rerun) and cut
the mean iterations from ~25 to ~5 for CMX. The warm start is an optional
`C_init` argument of `solve_newton_2d` with its own test
(`tests/test_2d.py::test_2d_warm_start_c_init_reaches_the_same_solution_in_fewer_iterations`);
existing callers are unaffected. The `rebeca` runs had no stalls before or
after, and all four runs reported here use the warm-started code.
