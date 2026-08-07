import copy
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.elliptic import Grid, build_laplacian, cell_volumes
from nitrifiers.parabolic import (build_advection_matrix,
                                    build_advection_rho_jacobian,
                                    solve_parabolic, growth_rate_field, SPECIES)


def test_diffusion_operator_is_exactly_mass_conservative():
    """Weighted by elliptic.cell_volumes -- the SAME measure the operator is
    normalised by. Mass conservation is a telescoping-flux identity that holds
    only for that measure; weighting by any other turns this into a test of
    volume-normalisation truncation error instead of conservation."""
    for geometry, p in [("slab", 0), ("radial", 1), ("radial", 2)]:
        grid = Grid(N=80, geometry=geometry, p=p)
        weights = cell_volumes(grid)
        u = np.sin(np.pi * grid.r)
        Lap = build_laplacian(grid)
        assert abs(np.sum(weights * (Lap @ u))) < 1e-10


def test_advection_operator_is_exactly_mass_conservative():
    """Regression test for the volume-normalisation bug in the advection
    operator (see the module docstring): it used the approximate V ~= r^p*h,
    which is wrong by ~2x at the OUTER boundary node, while build_laplacian
    used exact volumes -- so the two operators conserved different measures
    and the coupled scheme conserved neither.

    Two things matter about how this is now tested, both of which the previous
    version got wrong:

      1. The probe u MUST be nonzero at r=1. The old probe was u = sin(pi*r),
         which vanishes there -- precisely the node carrying the factor-of-2
         error -- so the bug was invisible to it. The `+ 0.5` below is the
         whole point of the test; removing it would silently restore the blind
         spot.
      2. The assertion is EXACT conservation, not convergence toward it. The
         old test asserted only that the defect halved under refinement, which
         a genuinely non-conservative operator can satisfy; with the exposing
         probe the true defect did not converge to zero at all.
    """
    for geometry, p in [("slab", 0), ("radial", 1), ("radial", 2)]:
        for N in (40, 160):
            grid = Grid(N=N, geometry=geometry, p=p)
            u = np.sin(np.pi * grid.r) + 0.5      # nonzero at r=1 -- see (1)
            rho = np.cos(np.pi * grid.r / 2)
            Adv = build_advection_matrix(grid, rho)
            defect = abs(np.sum(cell_volumes(grid) * (Adv @ u)))
            assert defect < 1e-12, (geometry, p, N, defect)


def _uniform_r(grid):
    return grid.r


def test_bacteria_stay_nonnegative_and_grow_only_where_substrate_allows():
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=100, geometry="radial", p=2)
    r = grid.r
    # anoxic core / aerobic rim substrate field, matching Stage 3/4 zonation
    C = {
        "NH4": 0.7 + 0.3 * r,
        "NO2": 0.02 * (1 - r),
        "NO3": 0.25 * (1 - r),
        "O2": np.where(r > 0.85, (r - 0.85) / 0.15 * 0.4, 0.0),
    }
    U0 = {sp: 0.05 * np.exp(-((r - 0.5) / 0.1) ** 2) for sp in SPECIES}
    U, mass_hist, _ = solve_parabolic(coeffs, C, U0, grid, dt=0.01, n_steps=300)

    for sp in SPECIES:
        assert np.all(U[sp] >= -1e-9)
    # AOB/CMX (need O2) should end up peaking in the aerobic shell (r > 0.85),
    # not at their original seed location (r = 0.5, inside the anoxic core)
    for sp in ("AOB", "CMX"):
        assert r[np.argmax(U[sp])] > 0.85
    assert mass_hist[-1] > mass_hist[0]  # net growth in the aerobic rim


def test_pure_decay_when_no_substrate_available():
    coeffs = copy.deepcopy(elliptic_coefficients("toy"))
    grid = Grid(N=60, geometry="radial", p=2)
    r = grid.r
    C = {"NH4": np.zeros_like(r), "NO2": np.zeros_like(r),
         "NO3": np.zeros_like(r), "O2": np.zeros_like(r)}
    U0 = {sp: 0.05 * np.exp(-((r - 0.5) / 0.1) ** 2) for sp in SPECIES}
    U, mass_hist, _ = solve_parabolic(coeffs, C, U0, grid, dt=0.01, n_steps=200)
    assert mass_hist[-1] < mass_hist[0]
    assert all(m2 <= m1 + 1e-12 for m1, m2 in zip(mass_hist, mass_hist[1:]))
    for sp in SPECIES:
        assert np.all(U[sp] >= -1e-9)


def test_death_term_is_density_dependent_not_constant():
    """Regression test for a real bug found via direct visual inspection of
    arXiv:2512.13156 eq. (2.1)/(2.5): the death term is -bhat_i*rhohat
    (density-dependent), not a constant -bhat_i -- an earlier version of this
    module implemented the constant form, which an initial pdftotext-based
    extraction of the PDF had silently made look correct (it dropped the rho
    symbol throughout that document). With ample, uniform, saturating
    substrate and a spatially uniform seed (so the growth-rate field g_i is
    the same everywhere and diffusion/advection cannot cause any spatial
    redistribution), the per-step growth RATIO mass[n+1]/mass[n] must
    strictly decrease over time as rhohat builds up and the density-dependent
    sink strengthens -- a constant death term would instead give an exactly
    constant ratio (pure exponential growth) throughout."""
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=30, geometry="radial", p=2)
    Npts = grid.N + 1
    C = {"NH4": np.full(Npts, 10.0), "NO2": np.full(Npts, 10.0),
         "NO3": np.full(Npts, 10.0), "O2": np.full(Npts, 10.0)}
    U0 = {sp: np.full(Npts, 0.05) for sp in SPECIES}

    U, mass_hist, _ = solve_parabolic(coeffs, C, U0, grid, dt=0.05, n_steps=25)
    assert all(m2 > m1 for m1, m2 in zip(mass_hist, mass_hist[1:]))  # still growing
    ratios = [mass_hist[i + 1] / mass_hist[i] for i in range(len(mass_hist) - 1)]
    assert all(ratios[i + 1] < ratios[i] for i in range(len(ratios) - 1)), ratios


def test_uniform_case_reaches_analytic_equilibrium():
    """The fully-implicit solver must converge to the correct steady state.
    With ample, uniform, saturating substrate and a uniform seed, the problem
    reduces (per grid point) to the logistic ODE du_i/dt = u_i(g_i - bhat_i*rho),
    rho = sum_j u_j. For symmetric species the fixed point is rho* = g_i/bhat_i
    per point. This is the exact case where the earlier single-lag scheme both
    went unstable AND settled at the wrong equilibrium -- so hitting rho* here
    is the core correctness check for the implicit rewrite."""
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=30, geometry="radial", p=2)
    Npts = grid.N + 1
    C = {"NH4": np.full(Npts, 10.0), "NO2": np.full(Npts, 10.0),
         "NO3": np.full(Npts, 10.0), "O2": np.full(Npts, 10.0)}
    U0 = {sp: np.full(Npts, 0.05) for sp in SPECIES}

    g0 = growth_rate_field(coeffs, C, "AOB")[0]
    rho_star = g0 / coeffs["bhat"]["AOB"]

    U, mass_hist, _ = solve_parabolic(coeffs, C, U0, grid, dt=0.5, n_steps=200)
    rho = U["AOB"] + U["NOB"] + U["CMX"]
    assert np.max(np.abs(rho - rho_star)) < 1e-4 * rho_star, (rho[0], rho_star)
    assert np.ptp(rho) < 1e-8  # stays spatially uniform (no spurious symmetry break)
    assert all(m2 >= m1 - 1e-12 for m1, m2 in zip(mass_hist, mass_hist[1:]))


def test_stable_and_bounded_at_large_dt():
    """Regression test for the stability fix: the fully-implicit scheme must
    stay finite, non-negative and bounded even at large dt and high density,
    where the previous single-lag scheme blew up (oscillated / diverged). Runs
    both a uniform seed and a localised bump (cross-diffusion genuinely active)
    at dt values well beyond the old scheme's stability limit."""
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=40, geometry="radial", p=2)
    Npts = grid.N + 1
    r = grid.r
    C = {"NH4": np.full(Npts, 10.0), "NO2": np.full(Npts, 10.0),
         "NO3": np.full(Npts, 10.0), "O2": np.full(Npts, 10.0)}
    g0 = growth_rate_field(coeffs, C, "AOB")[0]
    rho_cap = g0 / coeffs["bhat"]["AOB"]

    seeds = {
        "uniform": {sp: np.full(Npts, 0.05) for sp in SPECIES},
        "bump": {sp: 0.5 * np.exp(-((r - 0.3) / 0.1) ** 2) for sp in SPECIES},
    }
    for name, U0 in seeds.items():
        for dt in (0.5, 1.0):
            U, mass_hist, _ = solve_parabolic(coeffs, C, U0, grid, dt=dt, n_steps=60)
            rho = U["AOB"] + U["NOB"] + U["CMX"]
            assert np.all(np.isfinite(rho)), (name, dt)
            for sp in SPECIES:
                assert np.all(U[sp] >= -1e-9), (name, dt, sp)
            # density must not exceed the carrying capacity by more than a hair
            assert np.max(rho) < rho_cap * 1.01, (name, dt, np.max(rho), rho_cap)


def test_advection_rho_jacobian_matches_finite_difference():
    """Unit test for build_advection_rho_jacobian: the analytic d/drho of
    (Adv(rho) @ u_i) must match a finite-difference Jacobian. This is the
    term whose omission left the Newton solver only 'modified' (stalling on
    sharp fronts); getting its sign/upwind structure right is what makes the
    full Newton converge, so it is locked in here directly rather than only
    via the solver-level convergence test."""
    grid = Grid(N=40, geometry="radial", p=2)
    Npts = grid.N + 1
    rng = np.random.default_rng(1)
    u_i = rng.random(Npts) + 0.1
    rho = np.sort(rng.random(Npts)) * 3.0  # monotone -> stable upwind directions

    M = build_advection_rho_jacobian(grid, u_i, rho).toarray()
    eps = 1e-7
    T0 = build_advection_matrix(grid, rho) @ u_i
    M_fd = np.zeros((Npts, Npts))
    for n in range(Npts):
        rp = rho.copy(); rp[n] += eps
        M_fd[:, n] = (build_advection_matrix(grid, rp) @ u_i - T0) / eps
    rel = np.max(np.abs(M - M_fd)) / np.max(np.abs(M_fd))
    assert rel < 1e-6, rel


def test_sharp_front_newton_converges_to_tolerance():
    """Regression test for the modified->full Newton fix: a sharp density
    front with strong cross-diffusion (the regime where the modified Newton
    stalled at ~1e-4 residual and emitted 'stalled with large residual'
    warnings) must now converge cleanly -- i.e. no such warning fires -- while
    staying finite, non-negative and bounded by the carrying capacity."""
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=80, geometry="radial", p=2)
    Npts = grid.N + 1
    r = grid.r
    C = {"NH4": np.full(Npts, 10.0), "NO2": np.full(Npts, 10.0),
         "NO3": np.full(Npts, 10.0), "O2": np.full(Npts, 10.0)}
    cap = growth_rate_field(coeffs, C, "AOB")[0] / coeffs["bhat"]["AOB"]
    U0 = {sp: np.where(r < 0.4, cap / 3, 1e-4) for sp in SPECIES}

    for dt in (0.2, 0.5):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            U, _, _ = solve_parabolic(coeffs, C, U0, grid, dt=dt, n_steps=40)
        stalls = [x for x in w if "stalled with large residual" in str(x.message)]
        assert not stalls, [str(x.message) for x in stalls]
        rho = U["AOB"] + U["NOB"] + U["CMX"]
        assert np.all(np.isfinite(rho))
        assert np.max(rho) < cap * 1.01
        for sp in SPECIES:
            assert np.all(U[sp] >= -1e-9)


if __name__ == "__main__":
    test_diffusion_operator_is_exactly_mass_conservative()
    test_advection_operator_is_exactly_mass_conservative()
    test_bacteria_stay_nonnegative_and_grow_only_where_substrate_allows()
    test_pure_decay_when_no_substrate_available()
    test_death_term_is_density_dependent_not_constant()
    test_uniform_case_reaches_analytic_equilibrium()
    test_stable_and_bounded_at_large_dt()
    test_advection_rho_jacobian_matches_finite_difference()
    test_sharp_front_newton_converges_to_tolerance()
    print("All Stage 5 parabolic solver sanity checks passed.")
