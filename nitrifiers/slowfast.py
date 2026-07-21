"""
Stage 6: the full slow-scale loop from SlowFast_Nitrifiers.pdf:

    Given u^n -> solve elliptic problem for c^n -> advance u^n to u^{n+1}.

Each slow step:
  1. Solve the quasi-steady substrate system for c^n given the current
     (frozen) bacterial profile u^n -- via elliptic.solve_newton by default,
     falling back to relaxation.solve_relaxation if Newton fails to reach
     tolerance (residual above `elliptic_tol` after its iteration budget).
     This makes the Stage 3/4 comparison operational rather than just
     diagnostic: Newton is tried first (cheaper when it works), relaxation
     is the safety net for stiffer regimes where Newton stalls.
  2. Advance u^n -> u^{n+1} by one step of size dt_slow using
     parabolic.solve_parabolic (semi-implicit backward Euler), holding c^n
     fixed for that single step.

Repeat for n_slow_steps. Returns the final (u, c) state and a per-step
diagnostic history (elliptic method used, iteration counts, residuals,
total bacterial mass) for inspection.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .nondim import elliptic_coefficients
from .elliptic import Grid, solve_newton, SUBSTRATES
from .relaxation import solve_relaxation
from .parabolic import solve_parabolic, SPECIES, _total_mass


@dataclass
class SlowStepRecord:
    step: int
    elliptic_method: str
    elliptic_iters: int
    elliptic_residual: float
    total_mass: float


def solve_c_given_u(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                     elliptic_tol: float = 1e-8, newton_maxiter: int = 200,
                     relax_kwargs: dict | None = None):
    """Solve the quasi-steady substrate system for the given (fixed) u.

    solve_newton is tried first and, per its own docstring, has an INNER
    fallback to relaxation.solve_relaxation for the specific case of a
    detected degenerate/singular Jacobian (e.g. a substrate driven to exactly
    0 under zero-flux Neumann); that inner fallback is given this call's own
    elliptic_tol as its target, so a step handled by it already satisfies the
    tolerance this function was asked for. The method label returned by
    solve_newton (see its docstring) is passed straight through, so a step
    that engaged the inner fallback shows up honestly as
    "newton_inner_relax_fallback", not "newton".

    The OUTER fallback below (solve_relaxation called directly here) is a
    separate, defensive backstop: it only fires if solve_newton still returns
    a residual above elliptic_tol WITHOUT ever having engaged its inner
    fallback -- e.g. plain non-convergence within newton_maxiter that never
    actually tripped the degeneracy detector. Because the inner fallback
    already targets elliptic_tol, these two layers should not both fire for
    the same underlying degeneracy with mismatched tolerances; this one exists
    only to catch failure modes the inner detector doesn't cover.
    """
    C, hist, method = solve_newton(coeffs, U, grid, bc_type=bc_type, maxiter=newton_maxiter,
                                    tol=elliptic_tol)
    if hist[-1] < elliptic_tol:
        return C, method, len(hist) - 1, hist[-1]

    kwargs = dict(dt0=1e-2, dt_growth=1.3, max_steps=5000, steady_tol=elliptic_tol)
    if relax_kwargs:
        kwargs.update(relax_kwargs)
    C, hist_r = solve_relaxation(coeffs, U, grid, bc_type=bc_type, **kwargs)
    return C, "outer_relaxation_backstop", len(hist_r) - 1, hist_r[-1]


def run_slow_loop(preset_name: str, grid: Grid, U0: dict, n_slow_steps: int,
                   dt_slow: float, bc_type: str = "dirichlet",
                   elliptic_tol: float = 1e-8, record_every: int = 1):
    coeffs = elliptic_coefficients(preset_name)
    U = {sp: U0[sp].copy() for sp in SPECIES}

    history = []
    snapshots = []
    C = None
    for step in range(n_slow_steps):
        C, method, iters, resid = solve_c_given_u(coeffs, U, grid, bc_type=bc_type,
                                                   elliptic_tol=elliptic_tol)
        U, _, _ = solve_parabolic(coeffs, C, U, grid, dt=dt_slow, n_steps=1)

        mass = _total_mass(grid, U)
        history.append(SlowStepRecord(step=step, elliptic_method=method,
                                       elliptic_iters=iters, elliptic_residual=resid,
                                       total_mass=mass))
        if step % record_every == 0:
            snapshots.append({"step": step, "U": {k: v.copy() for k, v in U.items()},
                               "C": {k: v.copy() for k, v in C.items()}})

    return U, C, history, snapshots
