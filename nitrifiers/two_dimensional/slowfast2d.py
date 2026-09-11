"""Stage 6 in 2D: the slow-fast loop on a 2D Cartesian grid. No relaxation
fallback here (no 2D PTC solver) -- a failed elliptic step comes back as
method="newton_stalled" rather than being silently rescued."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..nondim import elliptic_coefficients
from .grid2d import Grid2D
from .elliptic2d import solve_newton_2d
from .parabolic2d import solve_parabolic_2d, total_mass_2d
from ..one_dimensional.parabolic import SPECIES


@dataclass
class SlowStepRecord2D:
    step: int
    elliptic_method: str
    elliptic_iters: int
    elliptic_residual: float
    total_mass: float


def solve_c_given_u_2d(coeffs, U, grid, boundary_nodes=None,
                        elliptic_tol=1e-8, newton_maxiter=100):
    C, hist, method = solve_newton_2d(coeffs, U, grid, bc_type="dirichlet",
                                       boundary_nodes=boundary_nodes,
                                       tol=elliptic_tol, maxiter=newton_maxiter)
    return C, method, len(hist) - 1, hist[-1]


def run_slow_loop_2d(preset_name: str, grid: Grid2D, U0: dict,
                      n_slow_steps: int, dt_slow: float,
                      boundary_nodes=None, elliptic_tol: float = 1e-8,
                      record_every: int = 1, verbose: bool = False):
    coeffs = elliptic_coefficients(preset_name)
    U = {s: np.asarray(U0[s]).ravel().copy() for s in SPECIES}

    history, snapshots = [], []
    C = None
    for step in range(n_slow_steps):
        C, method, iters, resid = solve_c_given_u_2d(
            coeffs, U, grid, boundary_nodes=boundary_nodes,
            elliptic_tol=elliptic_tol)
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt_slow, n_steps=1)

        mass = total_mass_2d(grid, U)
        history.append(SlowStepRecord2D(step=step, elliptic_method=method,
                                         elliptic_iters=iters,
                                         elliptic_residual=resid,
                                         total_mass=mass))
        if step % record_every == 0:
            snapshots.append({"step": step,
                              "U": {k: v.copy() for k, v in U.items()},
                              "C": {k: v.copy() for k, v in C.items()}})
        if verbose:
            print(f"  step {step:3d}: {method:16s} res={resid:.2e} mass={mass:.6e}")

    return U, C, history, snapshots
