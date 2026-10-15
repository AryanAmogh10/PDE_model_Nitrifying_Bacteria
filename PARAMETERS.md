# Parameters used in the simulations

All parameter regimes behind the plots in this repository. Every number in
section 1 is printed from the code (`nitrifiers/params.py`,
`nitrifiers/nondim.py`); sections 2 and 3 are the constants set at the top of
the `run_*.py` scripts. All simulated quantities are nondimensional.

## 1. Parameter presets (`nitrifiers/params.py`)

Three presets exist. Only `rebeca` and `eloi` are used for simulations
(step 1, `simulation_experiments/`); `toy` is for tests.

### Preset `toy`

Synthetic round-number preset for solver development and unit tests.

Dimensional parameters (units as in `nitrifiers/params.py`: rates in 1/day, K and c in uM, d and D in m^2/day, L in um):

| species | r | b | d | K (substrates) | Y (substrates) |
|---|---|---|---|---|---|
| AOB | 1 | 0.1 | 1e-06 | NH4 1, O2 1 | NH4 0.1, O2 0.1 |
| NOB | 1 | 0.1 | 1e-06 | NO2 1, O2 1 | NO2 0.1, O2 0.1 |
| CMX | 1 | 0.1 | 1e-06 | NH4 1, O2 1 | NH4 0.1, O2 0.1 |

| substrate | D | c_inf (boundary value) |
|---|---|---|
| NH4 | 0.0001 | 500 |
| NO2 | 0.0001 | 0 |
| NO3 | 0.0001 | 0 |
| O2 | 0.0001 | 200 |

Production fractions beta: {'AOB_to_NO2': 1.0, 'NOB_to_NO3': 1.0, 'CMX_to_NO3': 1.0}. Domain L = 500.

(`toy` is for solver development and tests only and is not used in any dataset.)

### Preset `rebeca`

Rough estimate from R. Gonzalez-Cabaleiro's e-mail (xlsx sheet1, rows 3-20).

Dimensional parameters (units as in `nitrifiers/params.py`: rates in 1/day, K and c in uM, d and D in m^2/day, L in um):

| species | r | b | d | K (substrates) | Y (substrates) |
|---|---|---|---|---|---|
| AOB | 0.5 | 0.05 | 1e-06 | NH4 1, O2 3.13 | NH4 0.0409, O2 0.06135 |
| NOB | 1.5 | 0.15 | 1e-06 | NO2 1, O2 3.13 | NO2 0.0242, O2 0.0121 |
| CMX | 0.375 | 0.0375 | 1e-06 | NH4 1, O2 3.13 | NH4 0.0651, O2 0.1302 |

| substrate | D | c_inf (boundary value) |
|---|---|---|
| NH4 | 8.64e-05 | 554 |
| NO2 | 8.64e-05 | 0 |
| NO3 | 8.64e-05 | 0 |
| O2 | 8.64e-05 | 187.5 |

Production fractions beta: {'AOB_to_NO2': 1.0, 'NOB_to_NO3': 1.0, 'CMX_to_NO3': 1.0}. Domain L = 500.

Nondimensional coefficients used by the solvers (`elliptic_coefficients('rebeca')`):

| species | rhat | bhat | Dhat | Ahat | Lambda (consumption) | Khat |
|---|---|---|---|---|---|---|
| AOB | 0.3333 | 0.03333 | 2.667 | 26.67 | NH4 1298, O2 865.2 | NH4 0.001805, O2 0.00565 |
| NOB | 1 | 0.1 | 2.667 | 26.67 | NO2 6580, O2 1.316e+04 | NO2 0.001805, O2 0.00565 |
| CMX | 0.25 | 0.025 | 2.667 | 26.67 | NH4 611.5, O2 305.8 | NH4 0.001805, O2 0.00565 |

Nondimensional boundary values c_inf_hat: {NH4: 1, NO2: 0, NO3: 0, O2: 0.3384}. LambdaProd: {AOB: 1298, NOB: 6580, CMX: 611.5}.

### Preset `eloi`

Martinez-Rabert thermodynamics-derived kinetics (T=25C, pH=7) + literature diffusion coefficients. NOB defaults to Nitrospira-type kinetics; Nitrobacter-type kept under 'NOB_nitrobacter_alt'.

Dimensional parameters (units as in `nitrifiers/params.py`: rates in 1/day, K and c in uM, d and D in m^2/day, L in um):

| species | r | b | d | K (substrates) | Y (substrates) |
|---|---|---|---|---|---|
| AOB | 0.9489 | 0.09489 | 1e-06 | NH4 1.453, O2 3.606 | NH4 0.0409, O2 0.02727 |
| NOB | 0.503 | 0.0503 | 1e-06 | NO2 8.894e-05, O2 2.594 | NO2 0.0242, O2 0.0484 |
| CMX | 0.06923 | 0.006923 | 1e-06 | NH4 0.063, O2 3.13 | NH4 0.0651, O2 0.03255 |
| NOB_nitrobacter_alt | 0.7986 | 0.07986 | 1e-06 | NO2 0.004447, O2 7.813 | NO2 0.0242, O2 0.0484 |

| substrate | D | c_inf (boundary value) |
|---|---|---|
| NH4 | 0.0001692 | 500 |
| NO2 | 0.0001651 | 0 |
| NO3 | 0.0001644 | 0 |
| O2 | 0.0001814 | 93.8 |

Production fractions beta: {'AOB_to_NO2': 1.0, 'NOB_to_NO3': 1.0, 'CMX_to_NO3': 1.0}. Domain L = 500.

Nondimensional coefficients used by the solvers (`elliptic_coefficients('eloi')`):

| species | rhat | bhat | Dhat | Ahat | Lambda (consumption) | Khat |
|---|---|---|---|---|---|---|
| AOB | 1 | 0.1 | 4.215 | 42.15 | NH4 1394, O2 1949 | NH4 0.002906, O2 0.007212 |
| NOB | 0.53 | 0.053 | 4.215 | 42.15 | NO2 1279, O2 582.1 | NO2 1.779e-07, O2 0.005188 |
| CMX | 0.07296 | 0.007296 | 4.215 | 42.15 | NH4 63.87, O2 119.1 | NH4 0.000126, O2 0.00626 |

Nondimensional boundary values c_inf_hat: {NH4: 1, NO2: 0, NO3: 0, O2: 0.1876}. LambdaProd: {AOB: 1428, NOB: 1285, CMX: 65.74}.


## 2. Paper parameter sets (competition and commensalism runs)

Nondimensional values of the source paper (arXiv:2512.13156, Table 1) for
Case (A), three-species competition, and Case (C), commensalism, with the
deviations this project makes. The kinetics are not taken from the presets.

| quantity | Case A (competition) | Case C (commensalism) | paper value if different |
|---|---|---|---|
| bacterial motility d / a | 1e-6 / 1e-5 (a/d = 10) | 1e-6 / 1e-5 | same |
| growth rate r | 1 | 1 | same |
| half-saturation K | 1 | 1 | same |
| decay b | 0.1 | 0.1 | same |
| yield Y | 0.2 | 10 | same |
| production fraction | 1 (beta), NO2/NO3 added to the paper's single-substrate Case A | 0.5 (sigma12 = sigma23) | none in Case A; same in Case C |
| substrate diffusivity D | 0.2 | 0.2 (slow-fast) / set by eps (coupled) | **1e-4 (A), 1e-5 (C)** |
| consumption strength Lambda = r / (D Y) | 25 | 0.5 | 5e4 (A), 1e4 (C) |
| O2 (competition only) | non-limiting: K_O2 = 1e-8, Lambda_O2 = 1e-6 Lambda | not present | not in paper |
| substrate boundary | per regime, section 2.2 | per regime, section 2.2 | Neumann, flux 1e-5 |
| initial substrate | boundary value | c1 = 5, c2 = c3 = 0 | c_0 = 5 |
| domain | unit square, 100 x 100 nodes | same | L = 1 |

Substrate diffusivity is raised from the paper's values so that the quasi-
steady approximation is valid (epsilon = r L^2 / D is ~1e4 at the paper's
value). This lowers Lambda by a factor 2000 (Case A) and 20000 (Case C,
slow-fast) and is the main departure from the paper; see the top-level README.

Initial biomass (all runs): 300 small circles (radius 0.05) placed at random
inside a disc of radius 0.2 at the domain centre, each assigned at random to
one of the three species, total density normalised to 1 wherever occupied
(seed 42). The step 1 runs use the same 300 circles with fixed species
assignments (section 3).

### 2.1 Solvers

| system | steps | time step | snapshots at t = | Lambda | epsilon |
|---|---|---|---|---|---|
| slow-fast competition | 50 | 1.0 | 0, 5, ..., 50 | 25 | 0 (quasi-steady) |
| slow-fast commensalism | 50 | 1.0 | 0, 5, ..., 50 | 0.5 | 0 (quasi-steady) |
| coupled competition | 500 | 0.1 | 0, 5, ..., 50 | 125 | 25 |
| coupled commensalism | 500 | 0.1 | 0, 5, ..., 50 | 2.5 | 25 |

Lambda and epsilon are tied by Lambda = eps / Y (both scale as 1/D). The
coupled runs also have an epsilon scan: eps = 1, 5, 20, 25, 50, 100 with
Lambda = eps / Y, 50 x 50 grid, T = 10, dt = 0.1 and 0.05.

### 2.2 Boundary regimes

The regime is applied to the substrates fed from outside (NH4 and NO2 in
competition, c1 in commensalism). In competition O2 stays Dirichlet 5 and NO3
has a Dirichlet 0 outlet; in commensalism c2 and c3 are sealed (zero flux).
NO2 is treated as an externally fed substrate in the competition runs (value
5 under dirichlet and robin, same flux as NH4 under neumann), as in the
earlier Case A setup, not as a pure product held at 0.

| regime | specification on the fed substrates |
|---|---|
| dirichlet | fixed value 5 |
| neumann | flux = paper's 1e-5 divided by D: -5e-5 for the slow-fast runs (D = 0.2), -2.5e-4 for the coupled runs (D = 1/eps = 0.04) |
| robin | Bi = 5, c_inf = 5 |

## 3. Step 1: AOB and CMX in isolation (`simulation_experiments/`)

| setting | value |
|---|---|
| presets | `rebeca`, `eloi` (kinetics from the preset, section 1) |
| bacterial motility | d = 1e-6, a = 1e-5 (Case A), not the presets' values |
| boundary | Dirichlet on all substrates: NH4 = 1, O2 = 5, NO2 = NO3 = 0 |
| initial substrate | equal to the boundary values |
| grid, time step | 100 x 100 nodes, dt = 0.5 |
| end | T = 50, or the time biomass first touches the boundary |
| survival cutoff | u < 1e-6 set to 0 after every step |
| substrate solve | quasi-steady (slow-fast), warm-started from the previous step |
| initial biomass | the same 300 circles; AOB system: each circle AOB or NOB (50/50); CMX system: all CMX |
| O2 choice | O2_BC = 5, from the scan in `simulation_experiments/README.md` |

The motility and survival-cutoff choices are not from the task description
and have not been agreed with the supervisor yet.
