"""Stage 4: pseudo-transient-continuation fallback/cross-check for elliptic.py.
Marches (M/dt - J) dC = F forward in pseudo-time, dt growing geometrically
from a small, stable start toward a plain Newton step."""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elliptic import (
    Grid, build_laplacian, apply_bc, reaction_and_jacobian, _assemble_global,
    SUBSTRATES, normalize_bc_specs, _default_initial_guess,
    bc_concentration_scale, bc_row_target,
)


def solve_relaxation(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                      bc_specs: dict | None = None,
                      dt0: float = 1e-2, dt_growth: float = 1.3, dt_max: float = 1e8,
                      steady_tol: float = 1e-9, max_steps: int = 5000,
                      c_max_factor: float = 5.0, verbose: bool = False):
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    bc_specs = normalize_bc_specs(coeffs, bc_type, bc_specs)

    Lap_bc = {}
    for sub in SUBSTRATES:
        bt, val = bc_specs[sub]
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bt, val)
        Lap_bc[sub] = Lb

    mass_diag = np.ones(Npts)
    mass_diag[-1] = 0.0
    M = sp.diags(np.concatenate([mass_diag] * len(SUBSTRATES)))

    c_max = c_max_factor * max(bc_concentration_scale(bt, val)
                                for bt, val in bc_specs.values())

    C = _default_initial_guess(bc_specs, Npts)
    dt = dt0
    history = []

    for step in range(max_steps):
        R, dR = reaction_and_jacobian(C, U, coeffs)
        for sub in SUBSTRATES:
            R[sub][-1] = 0.0
        F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
        for k, sub in enumerate(SUBSTRATES):
            bt, val = bc_specs[sub]
            F[(k + 1) * Npts - 1] = (Lap_bc[sub] @ C[sub])[-1] - bc_row_target(bt, val)
        _, J = _assemble_global(Lap_bc, R, dR, C)

        res_norm = np.linalg.norm(F, ord=np.inf)
        history.append(res_norm)
        if verbose and step % 20 == 0:
            print(f"PTC step={step} dt={dt:.3e} |F|_inf={res_norm:.3e}")
        if res_norm < steady_tol:
            break

        A = M / dt - J
        dC = spla.spsolve(A, F)
        for k, sub in enumerate(SUBSTRATES):
            C[sub] = np.clip(C[sub] + dC[k * Npts:(k + 1) * Npts], 0.0, c_max)

        dt = min(dt * dt_growth, dt_max)

    return C, history


def compare_with_elliptic(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                           bc_specs: dict | None = None, **relax_kwargs):
    from .elliptic import solve_newton
    C_ell, hist_ell, _ = solve_newton(coeffs, U, grid, bc_type=bc_type, bc_specs=bc_specs, maxiter=300)
    C_rel, hist_rel = solve_relaxation(coeffs, U, grid, bc_type=bc_type, bc_specs=bc_specs, **relax_kwargs)
    diffs = {sub: float(np.max(np.abs(C_ell[sub] - C_rel[sub]))) for sub in SUBSTRATES}
    return {
        "elliptic": C_ell, "elliptic_iters": len(hist_ell) - 1,
        "relaxation": C_rel, "relaxation_iters": len(hist_rel) - 1,
        "max_diff": diffs,
    }
