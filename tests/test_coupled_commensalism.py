"""
Validation for the commensalism chain solver used in
experiments/coupled/commensalism_dirichlet/run_coupled_commensalism.py
(and its neumann/robin copies, byte-identical apart from REGIME).

This solver is NOT part of nitrifiers/: unlike the competition case, its
chemistry (a single-Monod chain u1->c1, u2->c2, u3->c3 with byproduct
production, per paper eq. 2.4) does not fit the AOB/NOB/CMX co-limited
machinery in nitrifiers.two_dimensional, so it is a from-scratch Newton
solver, both for the coupled (eps kept) and QSSA (eps -> 0) form -- and it
had never been given the same rigorous checks as nitrifiers/coupled/
before this file.

It reuses (and therefore does NOT need to re-test) the grid/BC/Laplacian
machinery from nitrifiers.two_dimensional.grid2d, which is already covered
by tests/test_2d.py. What is genuinely new here is the reaction assembly
(Lambda, sigma, the CHAIN/PRODUCES coupling) and the eps/dt mass term, so
that is what these three checks isolate:

  1. DIFFUSION DECAY (closed form). With u = 0, R = 0 everywhere and the
     coupled solver's mass term is the only thing being exercised beyond
     grid2d's already-tested Laplacian: same closed-form check as
     test_coupled.py, run through this solver's own step_substrate.

  2. LARGE-DT LIMIT. The coupled step_substrate and the QSSA
     solve_qssa_substrate are two INDEPENDENTLY WRITTEN residuals for the
     same steady chemistry. If dt -> inf collapses one onto the other, that
     is strong evidence neither has a sign error or dropped term the other
     doesn't share.

  3. MASS BALANCE (exact, zero flux). Integrated over the domain, the only
     source for c2 is sigma * (what c1 lost to u1), and for c3 it's sigma *
     (what c2 lost to u2). This is checked exactly, not approximately --
     a wrong sign or a missing sigma factor breaks it immediately.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "experiments" / "coupled" / "commensalism_dirichlet"))

import run_coupled_commensalism as m  # noqa: E402
from nitrifiers.two_dimensional.grid2d import Grid2D  # noqa: E402


def _rough_ic(grid_n, peak=0.5, radius=0.25):
    grid = Grid2D(Nx=grid_n, Ny=grid_n, Lx=1.0, Ly=1.0)
    d = grid.radius().ravel()
    u = peak * np.exp(-(d / radius) ** 2)
    return grid, {s: u.copy() for s in m.SPECIES}


def test_pure_diffusion_decays_at_closed_form_rate():
    grid = Grid2D(Nx=48, Ny=48, Lx=1.0, Ly=1.0)
    specs = {s: ("dirichlet", 0.0) for s in m.SUBSTRATES}
    ops = m.SubstrateOps(grid, specs)
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    mode = (np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()
    U = {s: np.zeros(grid.Npts) for s in m.SPECIES}
    eps, T = 2.0, 0.1
    rate = 2 * np.pi ** 2 / eps
    exact = mode * np.exp(-rate * T)

    errors = []
    for n_steps in (5, 10, 20, 40):
        dt = T / n_steps
        C = {s: mode.copy() for s in m.SUBSTRATES}
        for _ in range(n_steps):
            C, _, _, method, _ = m.step_substrate(U, C, ops, eps, dt, tol=1e-12)
            assert method == "newton"
        errors.append(np.max(np.abs(C["c1"] - exact)))
    errors = np.array(errors)
    assert errors[-1] < 5e-3
    rates = np.log(errors[:-1] / errors[1:]) / np.log(2)
    assert np.all(rates > 0.85), rates


def test_large_dt_limit_matches_the_independently_written_qssa_solver():
    grid, U = _rough_ic(grid_n=24)
    old_lam = m.LAM
    m.LAM = 5.0 / m.Y_MAIN
    try:
        for regime in ("dirichlet", "neumann", "robin"):
            specs = m.make_specs(regime)
            C_qssa, method = m.solve_qssa_substrate(U, specs, grid, tol=1e-8, maxiter=150)
            assert method == "newton", regime

            ops = m.SubstrateOps(grid, specs)
            C = {"c1": np.full(grid.Npts, 0.3), "c2": np.full(grid.Npts, 0.3),
                 "c3": np.full(grid.Npts, 0.3)}
            for _ in range(6):
                C, _, _, method, _ = m.step_substrate(U, C, ops, eps=1.0, dt=1e8, tol=1e-8, maxiter=100)
                assert method == "newton", regime
            for s in m.SUBSTRATES:
                assert np.max(np.abs(C[s] - C_qssa[s])) < 1e-6, (regime, s)
    finally:
        m.LAM = old_lam


def test_zero_flux_mass_balance_across_the_whole_chain():
    grid, U = _rough_ic(grid_n=20, peak=0.8)
    old_lam = m.LAM
    m.LAM = 5.0 / m.Y_MAIN
    try:
        specs = {s: ("neumann", 0.0) for s in m.SUBSTRATES}
        ops = m.SubstrateOps(grid, specs)
        eps, dt = 0.7, 0.05
        C_old = {s: np.full(grid.Npts, 1.0) + 0.1 * grid.radius().ravel() for s in m.SUBSTRATES}
        C_new, _, _, method, _ = m.step_substrate(U, C_old, ops, eps, dt, tol=1e-13)
        assert method == "newton"

        # rebuild R(C_new) the same way step_substrate does internally, via
        # the public reaction() closure semantics reproduced here directly
        R = {s: np.zeros(grid.Npts) for s in m.SUBSTRATES}
        for spn, sub in m.CHAIN.items():
            Mc = m.monod(C_new[sub], m.K_MAIN)
            R[sub] += -m.LAM * U[spn] * Mc
            if spn in m.PRODUCES:
                R[m.PRODUCES[spn]] += m.SIGMA * m.LAM * U[spn] * Mc

        for s in m.SUBSTRATES:
            lhs = eps * float(np.sum(grid.Vflat * (C_new[s] - C_old[s]))) / dt
            rhs = float(np.sum(grid.Vflat * R[s]))
            scale = max(1.0, abs(rhs))
            assert abs(lhs - rhs) / scale < 1e-9, (s, lhs, rhs)
    finally:
        m.LAM = old_lam
