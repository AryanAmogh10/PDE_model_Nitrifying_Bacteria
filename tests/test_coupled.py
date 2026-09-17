"""
Validation for nitrifiers/coupled (time-dependent substrate, eps kept).

Three independent checks, each isolating one thing the new code adds on top
of the already-validated elliptic2d machinery:

  1. DIFFUSION DECAY (closed form). With no bacteria the substrate equation is
     eps*c_t = Lap c. On the unit square with c = 0 on the boundary, the mode
     sin(pi x) sin(pi y) decays as exp(-2 pi^2 t / eps). Backward Euler must
     converge to this at first order in dt (spatial error held fixed and
     small by using a fine grid). This pins the mass term's sign and scaling
     against eps and dt.

  2. QSSA CONSISTENCY. As dt -> inf the backward-Euler step becomes the
     steady problem the slow-fast solver already solves. Repeated huge steps
     from any start must land on solve_newton_2d's answer to solver tolerance.
     This proves the time-dependent residual and Jacobian are the elliptic
     ones plus exactly the mass term, nothing else.

  3. DISCRETE MASS BALANCE (exact). With zero-flux on every edge the FV scheme
     must satisfy eps * (int c_new - int c_old) / dt = int R(c_new) to
     round-off, one step at a time. Catches any mismatch between the volume
     measure used in the Laplacian and the one used in the mass term.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.nondim import elliptic_coefficients, SUBSTRATES
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d, reaction_and_jacobian
from nitrifiers.coupled.substrate_step2d import SubstrateOperators2D, step_substrate_2d


def _mild_coeffs(lam=25.0, K=1.0):
    """Case-A-scale chemistry (Lambda = r/(D*Y) = 25) with full production
    (beta = 1) so the NO2/NO3 cross-coupling is exercised; 'toy' has
    Lambda ~ 1e3 which is needlessly stiff for these consistency checks."""
    c = elliptic_coefficients("toy")
    lam_o2 = 1e-6 * lam
    return {
        "Lambda": {"AOB": {"NH4": lam, "O2": lam_o2}, "NOB": {"NO2": lam, "O2": lam_o2},
                   "CMX": {"NH4": lam, "O2": lam_o2}},
        "LambdaProd": {s: lam for s in SPECIES},
        "Khat": {"AOB": {"NH4": K, "O2": 1e-8}, "NOB": {"NO2": K, "O2": 1e-8}, "CMX": {"NH4": K, "O2": 1e-8}},
        "rhat": {s: 1.0 for s in SPECIES}, "bhat": {s: 0.1 for s in SPECIES},
        "Dhat": {s: 1e-6 for s in SPECIES}, "Ahat": {s: 1e-5 for s in SPECIES},
        "c_inf_hat": {"NH4": 1.0, "NO2": 0.0, "NO3": 0.0, "O2": 1.0},
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "consumed_substrates": c["consumed_substrates"], "production": c["production"],
    }


def _zero_U(grid):
    return {s: np.zeros(grid.Npts) for s in SPECIES}


def _colony_U(grid, peak=0.5, radius=0.25):
    d = grid.radius().ravel()
    u = peak * np.exp(-(d / radius) ** 2)
    return {s: u.copy() for s in SPECIES}


def test_pure_diffusion_decays_at_closed_form_rate_first_order_in_dt():
    coeffs = _mild_coeffs()
    grid = Grid2D(Nx=48, Ny=48, Lx=1.0, Ly=1.0)
    specs = {s: ("dirichlet", 0.0) for s in SUBSTRATES}
    ops = SubstrateOperators2D(grid, coeffs, specs)
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    mode = (np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()
    eps, T = 2.0, 0.1
    rate = 2 * np.pi ** 2 / eps
    exact = mode * np.exp(-rate * T)
    U = _zero_U(grid)

    errors = []
    for n_steps in (5, 10, 20, 40):
        dt = T / n_steps
        C = {s: mode.copy() for s in SUBSTRATES}
        for _ in range(n_steps):
            C, _, _, method = step_substrate_2d(coeffs, U, C, ops, eps, dt, tol=1e-12)
            assert method == "newton"
        errors.append(np.max(np.abs(C["NH4"] - exact)))
    errors = np.array(errors)
    # discrete-eigenvalue error of the 48x48 Laplacian for this mode is ~1e-4
    # relative, well below the temporal errors being measured here
    assert errors[-1] < 5e-3
    rates = np.log(errors[:-1] / errors[1:]) / np.log(2)
    assert np.all(rates > 0.85), rates


def test_large_dt_limit_recovers_the_quasi_steady_solution():
    coeffs = _mild_coeffs()
    grid = Grid2D(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    # NO3 needs an outlet here: it is produced (beta = 1) and consumed by
    # nothing, so a sealed boundary would leave the steady problem with no
    # solution at all -- the QSSA reference would correctly stall
    specs = {"NH4": ("dirichlet", 1.0), "NO2": ("robin", (3.0, 0.5)),
             "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", 0.8)}
    U = _colony_U(grid)
    C_qssa, hist, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-12, maxiter=100)
    assert method == "newton"

    ops = SubstrateOperators2D(grid, coeffs, specs)
    C = {s: np.full(grid.Npts, 0.3) for s in SUBSTRATES}
    for _ in range(6):
        C, _, _, method = step_substrate_2d(coeffs, U, C, ops, eps=1.0, dt=1e8, tol=1e-12, maxiter=100)
        assert method == "newton"
    for s in SUBSTRATES:
        assert np.max(np.abs(C[s] - C_qssa[s])) < 1e-8, s


def test_zero_flux_step_conserves_mass_exactly():
    coeffs = _mild_coeffs()
    grid = Grid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
    specs = {s: ("neumann", 0.0) for s in SUBSTRATES}
    ops = SubstrateOperators2D(grid, coeffs, specs)
    U = _colony_U(grid, peak=0.8)
    eps, dt = 0.7, 0.05
    C_old = {s: np.full(grid.Npts, 1.0) + 0.1 * grid.radius().ravel() for s in SUBSTRATES}
    C_new, _, res, method = step_substrate_2d(coeffs, U, C_old, ops, eps, dt, tol=1e-13)
    assert method == "newton"
    R, _ = reaction_and_jacobian(C_new, U, coeffs)
    for s in SUBSTRATES:
        lhs = eps * float(np.sum(grid.Vflat * (C_new[s] - C_old[s]))) / dt
        rhs = float(np.sum(grid.Vflat * R[s]))
        scale = max(1.0, abs(rhs))
        assert abs(lhs - rhs) / scale < 1e-10, (s, lhs, rhs)
