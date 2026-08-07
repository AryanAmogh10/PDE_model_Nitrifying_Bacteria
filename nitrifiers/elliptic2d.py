"""
Stage 3 in 2D: quasi-steady substrate solver on a 2D Cartesian grid.

This is the direct 2D extension of elliptic.py and reuses its reaction physics
VERBATIM: `reaction_and_jacobian` is pure elementwise NumPy over flat arrays,
so it is already dimension-agnostic and is imported unchanged rather than
re-implemented. What changes in 2D is only the spatial coupling (the Laplacian,
from grid2d.build_laplacian_2d) and the boundary bookkeeping: in 1D exactly one
row (index N) is a Dirichlet row, whereas in 2D an entire set of nodes is
(the four domain edges, or an arbitrary embedded boundary such as a staircased
circle -- see `boundary_nodes`).

Everything else -- the coupled 4*Npts Newton system, the damped backtracking
line search, the non-negativity projection, and the physical-plausibility upper
bound that was added in 1D after a spurious-root bug -- carries over with the
same structure and the same rationale. See elliptic.py's solve_newton docstring
for why the plausibility guard exists; it is retained here because nothing about
2D makes that failure mode less likely.

NOT carried over: the inner relaxation fallback for degenerate Jacobians. There
is no 2D pseudo-transient-continuation solver yet, so a degenerate solve is
reported honestly (method="newton_stalled") rather than silently handed to a
solver that does not exist. That is a known gap, not a claim of robustness.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .nondim import SUBSTRATES
from .elliptic import reaction_and_jacobian  # dimension-agnostic; reused as-is
from .grid2d import Grid2D, build_laplacian_2d, apply_bc_2d


def _residual_2d(C, U, Lap_bc, coeffs, bc_type, bnodes):
    """Global residual for the coupled 4*Npts system.

    Boundary rows carry the algebraic BC (c_k - c_inf = 0 for Dirichlet) and
    NO reaction contribution -- the 2D analogue of the 1D `R[sub][-1] = 0`,
    generalised from "the last row" to "every boundary node". Interior rows,
    including every node adjacent to the boundary, keep their full reaction
    term (the 1D row-0 bug was precisely the mistake of stripping a reaction
    term from a row that was not actually a BC row).
    """
    Npts = len(next(iter(C.values())))
    R, dR = reaction_and_jacobian(C, U, coeffs)
    c_inf = coeffs["c_inf_hat"]

    if bc_type == "dirichlet":
        for sub in SUBSTRATES:
            R[sub][bnodes] = 0.0

    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    if bc_type == "dirichlet":
        for k, sub in enumerate(SUBSTRATES):
            base = k * Npts
            F[base + bnodes] = C[sub][bnodes] - c_inf[sub]
    return F, R, dR


def _assemble_global_2d(Lap_bc, R, dR, C, bc_type, bnodes):
    """Sparse Jacobian of _residual_2d. Diagonal (self-coupling) blocks carry
    the Laplacian; every block additionally carries the elementwise reaction
    derivative, zeroed on boundary rows so those rows stay the pure algebraic
    identity written by apply_bc_2d."""
    n = len(SUBSTRATES)
    Npts = len(next(iter(C.values())))
    idx = {sub: k for k, sub in enumerate(SUBSTRATES)}
    blocks = [[None] * n for _ in range(n)]

    for i_sub in SUBSTRATES:
        i = idx[i_sub]
        blocks[i][i] = Lap_bc[i_sub].tolil()
        for j_sub in SUBSTRATES:
            j = idx[j_sub]
            if (i_sub, j_sub) not in dR:
                continue
            d = dR[(i_sub, j_sub)].copy()
            if bc_type == "dirichlet":
                d[bnodes] = 0.0
            block = sp.diags(d, format="lil")
            blocks[i][j] = block if blocks[i][j] is None else blocks[i][j] + block

    for i in range(n):
        for j in range(n):
            if blocks[i][j] is None:
                blocks[i][j] = sp.csr_matrix((Npts, Npts))
    return sp.bmat(blocks, format="csr")


def solve_newton_2d(coeffs: dict, U: dict, grid: Grid2D,
                     bc_type: str = "dirichlet",
                     boundary_nodes: np.ndarray | None = None,
                     tol: float = 1e-10, maxiter: int = 60,
                     damped: bool = True, max_backtracks: int = 30,
                     c_max_factor: float = 5.0, verbose: bool = False):
    """Newton iteration for the 2D quasi-steady substrate system.

    U: dict species -> flat array of length grid.Npts (or a (Nx+1, Ny+1) field;
    it is flattened here). Returns (C, history, method) with C a dict of FLAT
    arrays, matching the 1D solver's 3-tuple return convention (see the 1D
    note on that signature: callers must unpack three values).

    `boundary_nodes` selects which nodes carry the Dirichlet condition and
    defaults to the four domain edges. Passing an explicit set allows an
    embedded boundary (used by the radial-symmetry validation, which imposes
    Dirichlet data on a staircased circle and on everything outside it).
    """
    Npts = grid.Npts
    U = {sp_name: np.asarray(v).ravel() for sp_name, v in U.items()}
    c_inf = coeffs["c_inf_hat"]
    # magnitude, not signed max -- same reason as the 1D fix (c_inf doubles as
    # a signed flux value under Neumann, and a signed max collapses the bound)
    c_max = c_max_factor * max(abs(v) for v in c_inf.values())

    bnodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)

    Lap0 = build_laplacian_2d(grid)
    Lap_bc = {sub: apply_bc_2d(Lap0, grid, bc_type, boundary_nodes=bnodes)
              for sub in SUBSTRATES}

    C = {sub: np.full(Npts, c_inf[sub]) for sub in SUBSTRATES}
    F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_type, bnodes)
    res = float(np.linalg.norm(F, ord=np.inf))
    history = [res]

    for it in range(maxiter):
        if res < tol:
            break
        J = _assemble_global_2d(Lap_bc, R, dR, C, bc_type, bnodes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            dX = spla.spsolve(J, -F)
        if not np.all(np.isfinite(dX)):
            warnings.warn("solve_newton_2d: non-finite Newton update (degenerate "
                           "Jacobian); returning current iterate as 'newton_stalled'.",
                           RuntimeWarning)
            return C, history, "newton_stalled"

        step = 1.0
        accepted = False
        if not damped:
            for k, sub in enumerate(SUBSTRATES):
                C[sub] = C[sub] + dX[k * Npts:(k + 1) * Npts]
            F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_type, bnodes)
            res = float(np.linalg.norm(F, ord=np.inf))
            accepted = True
        else:
            for _ in range(max_backtracks):
                raw = {sub: C[sub] + step * dX[k * Npts:(k + 1) * Npts]
                       for k, sub in enumerate(SUBSTRATES)}
                plausible = all(np.all(v <= c_max) for v in raw.values())
                C_try = {sub: np.clip(v, 0.0, c_max) for sub, v in raw.items()}
                F_try, R_try, dR_try = _residual_2d(C_try, U, Lap_bc, coeffs,
                                                     bc_type, bnodes)
                res_try = float(np.linalg.norm(F_try, ord=np.inf))
                if np.isfinite(res_try) and res_try < res and plausible:
                    C, F, R, dR, res = C_try, F_try, R_try, dR_try, res_try
                    accepted = True
                    break
                step *= 0.5
        if not accepted:
            warnings.warn(f"solve_newton_2d: backtracking exhausted at iter {it} "
                           f"(|F|_inf={res:.3e}); returning 'newton_stalled'.",
                           RuntimeWarning)
            return C, history, "newton_stalled"
        history.append(res)
        if verbose:
            print(f"  newton2d it={it} |F|_inf={res:.3e} step={step:.3g}")

    return C, history, "newton"
