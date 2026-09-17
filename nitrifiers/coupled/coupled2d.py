"""Fully time-dependent 2D loop: substrates and bacteria advance together
with one shared dt (Lie splitting: substrate backward-Euler step with the
current u, then the existing implicit bacterial step with the new c).

Nondimensional system, epsilon kept:

    eps * dc_j/dt = Lap(c_j) + R_j(u, c)
        du_i/dt = div(Dhat_i grad u_i + Ahat_i u_i grad rho) + u_i f_i(u, c)

eps = r_max * L^2 / D_j is the ratio of the substrate diffusion time to the
bacterial growth time. The slow-fast solver is the eps -> 0 limit of this
one; the paper's own Case (A) numbers put eps ~ 1e4, i.e. the opposite
limit (slow diffusion). Here eps is a free parameter so the two regimes,
and the crossover between them, can be studied on one code path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nondim import SUBSTRATES
from ..one_dimensional.parabolic import SPECIES
from ..two_dimensional.grid2d import Grid2D
from ..two_dimensional.parabolic2d import solve_parabolic_2d, total_mass_2d
from .substrate_step2d import SubstrateOperators2D, step_substrate_2d, term_magnitudes


@dataclass
class CoupledStepRecord2D:
    step: int
    t: float
    substrate_method: str
    substrate_iters: int
    substrate_residual: float
    total_mass: float
    term_magnitudes: dict = field(default_factory=dict)


def initial_substrate_2d(coeffs: dict, bc_specs: dict, grid: Grid2D, c0: dict | None = None):
    """Default c(x, 0): the boundary/bulk value for Dirichlet and Robin
    substrates (uniform, as in the paper's c_0), 1.0 for pure-Neumann ones
    (never 0.0 -- Monod(0) = 0 would zero every reaction Jacobian)."""
    C = {}
    for sub in SUBSTRATES:
        if c0 is not None and sub in c0:
            C[sub] = np.full(grid.Npts, float(c0[sub]))
            continue
        bt, val = bc_specs[sub]
        if bt == "dirichlet":
            start = float(val)
        elif bt == "robin":
            start = float(val[1]) if val[1] > 0 else 1.0
        else:
            start = 1.0
        C[sub] = np.full(grid.Npts, start)
    return C


def run_coupled_2d(coeffs: dict, grid: Grid2D, U0: dict, bc_specs: dict,
                   eps: float, dt: float, n_steps: int,
                   C0: dict | None = None, snapshot_every: int | None = None,
                   record_magnitudes_every: int = 1, substrate_tol: float = 1e-8,
                   on_snapshot=None, verbose: bool = False):
    ops = SubstrateOperators2D(grid, coeffs, bc_specs)
    U = {s: np.asarray(U0[s]).ravel().copy() for s in SPECIES}
    C = initial_substrate_2d(coeffs, ops.bc_specs, grid) if C0 is None else \
        {s: np.asarray(C0[s]).ravel().copy() for s in SUBSTRATES}

    history, snapshots = [], []
    stalled = 0
    if snapshot_every is not None:
        snapshots.append({"step": 0, "t": 0.0, "U": {k: v.copy() for k, v in U.items()},
                          "C": {k: v.copy() for k, v in C.items()}})
        if on_snapshot is not None:
            on_snapshot(0, 0.0, U, C)

    for step in range(1, n_steps + 1):
        C_prev = C
        C, iters, resid, method = step_substrate_2d(coeffs, U, C, ops, eps, dt, tol=substrate_tol)
        if method == "newton_stalled":
            stalled += 1
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt, n_steps=1)

        t = step * dt
        mags = term_magnitudes(coeffs, U, C, ops, C_prev=C_prev, eps=eps, dt=dt) \
            if (record_magnitudes_every and step % record_magnitudes_every == 0) else {}
        history.append(CoupledStepRecord2D(step=step, t=t, substrate_method=method,
                                           substrate_iters=iters, substrate_residual=resid,
                                           total_mass=total_mass_2d(grid, U),
                                           term_magnitudes=mags))
        if snapshot_every is not None and step % snapshot_every == 0:
            snapshots.append({"step": step, "t": t, "U": {k: v.copy() for k, v in U.items()},
                              "C": {k: v.copy() for k, v in C.items()}})
            if on_snapshot is not None:
                on_snapshot(step, t, U, C)
        if verbose and (step % max(1, n_steps // 10) == 0):
            print(f"  step {step:5d} t={t:7.3f}: substrate {method:14s} iters={iters} "
                  f"res={resid:.1e} mass={history[-1].total_mass:.5f}")

    return U, C, history, snapshots, stalled
