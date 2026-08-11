"""
Stage 4: fast-relaxation (pseudo-time-stepping) solver for the quasi-steady
substrate system, as an alternative/fallback to the direct elliptic solve in
elliptic.py -- per SlowFast_Nitrifiers.pdf: "keep u_i fixed and integrate
dc_j/dtau = D_j*Lap(c_j) + c_j*g_j(u,c) until the solution reaches a steady
state."

Implementation: pseudo-transient continuation (PTC). We march the same
residual F(C) = Lap(C) + R(u, C) used by elliptic.py forward with implicit
(backward) Euler in a pseudo-time variable, linearised with one Newton step
per pseudo-step (a standard, cheap PTC scheme):

    (M/dt - J(C^n)) * dC = F(C^n),     C^{n+1} = max(C^n + dC, 0)

where J = dF/dC is the same sparse Jacobian assembled in elliptic.py, and M is
the identity except on the two boundary rows of each substrate block (r=0
symmetry, r=1 outer BC), which stay purely algebraic (BC enforced exactly at
every pseudo-step, not just at steady state).

Adding 1/dt to the diagonal regularises the (possibly ill-conditioned, as
Stage 3 found) steady Jacobian: for small dt the scheme behaves like a very
stable, heavily-damped implicit-Euler relaxation (slow but robust); as dt is
grown geometrically each step it smoothly approaches the plain Newton step
used in elliptic.py (fast, but only once the iterate is already close to the
attracting steady state). This is exactly the "start slow/robust, end
fast" behaviour that makes PTC a good fallback when the direct elliptic
Newton solve struggles for stiff parameter regimes.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elliptic import (
    Grid, build_laplacian, apply_bc, reaction_and_jacobian, _assemble_global,
    SUBSTRATES, normalize_bc_specs, _default_initial_guess,
)


def solve_relaxation(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                      bc_specs: dict | None = None,
                      dt0: float = 1e-2, dt_growth: float = 1.3, dt_max: float = 1e8,
                      steady_tol: float = 1e-9, max_steps: int = 5000,
                      c_max_factor: float = 5.0, verbose: bool = False):
    """See elliptic.solve_newton's docstring for the bc_specs per-substrate
    interface (ITEM 1); bc_type is expanded into a per-substrate spec via
    normalize_bc_specs when bc_specs is not given, reproducing the old
    single-bc_type behaviour exactly."""
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    bc_specs = normalize_bc_specs(coeffs, bc_type, bc_specs)

    Lap_bc = {}
    for sub in SUBSTRATES:
        bt, val = bc_specs[sub]
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bt, val)
        Lap_bc[sub] = Lb

    # mass diagonal: 1 everywhere except row N (the true, fully-replaced
    # Dirichlet/Neumann BC row, purely algebraic). Row 0 (r=0) is a genuinely
    # evolving physical point using a symmetric stencil, NOT a boundary
    # constraint -- it must keep both its transient (mass) and reaction
    # terms. An earlier version zeroed both at row 0 too, which (like the
    # matching bug in elliptic.py) silently dropped the reaction term at the
    # domain center every step -- caught via the closed-form convergence
    # investigation in elliptic.py.
    mass_diag = np.ones(Npts)
    mass_diag[-1] = 0.0
    M = sp.diags(np.concatenate([mass_diag] * len(SUBSTRATES)))

    # Same physical-plausibility upper bound as elliptic.solve_newton, and for
    # the same reason: without it, this PTC iteration has no protection at all
    # against diverging to a spurious, unphysical value (only np.maximum(.,0)
    # floors it below). This was a real, live gap -- solve_newton's INNER
    # fallback hands degenerate solves to this function precisely when its own
    # guard can't find an admissible step, so if THIS function has no
    # matching guard, the combined "protected Newton + unprotected fallback"
    # is only as safe as its weakest link. Found via the ITEM 1 coupled
    # multi-substrate Neumann test, where Newton failed at iteration 0 (before
    # its own backtracking guard ever got a chance to reject anything) and the
    # unguarded relaxation fallback diverged to ~3.7e11.
    c_max = c_max_factor * max(abs(val) for _, val in bc_specs.values())

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
            # see the matching note in elliptic.py::_residual: neumann's row-N
            # target must be -val (apply_bc's own convention), not 0.0 -- the
            # earlier hardcoded 0.0 silently forced zero-flux regardless of the
            # actual requested flux value.
            F[(k + 1) * Npts - 1] = (Lap_bc[sub] @ C[sub])[-1] - (val if bt == "dirichlet" else -val)
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
    """Convenience helper for Stage 4: solve both ways and report max abs
    difference per substrate."""
    from .elliptic import solve_newton
    C_ell, hist_ell, _ = solve_newton(coeffs, U, grid, bc_type=bc_type, bc_specs=bc_specs, maxiter=300)
    C_rel, hist_rel = solve_relaxation(coeffs, U, grid, bc_type=bc_type, bc_specs=bc_specs, **relax_kwargs)
    diffs = {sub: float(np.max(np.abs(C_ell[sub] - C_rel[sub]))) for sub in SUBSTRATES}
    return {
        "elliptic": C_ell, "elliptic_iters": len(hist_ell) - 1,
        "relaxation": C_rel, "relaxation_iters": len(hist_rel) - 1,
        "max_diff": diffs,
    }
