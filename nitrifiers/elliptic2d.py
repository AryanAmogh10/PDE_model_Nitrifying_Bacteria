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


def normalize_bc_specs_2d(coeffs: dict, bc_type: str, bc_specs: dict | None) -> dict:
    """2D analogue of elliptic.normalize_bc_specs (ITEM 1's per-substrate
    interface, extended here for consistency between the 1D and 2D solvers).

    SCOPE LIMITATION, stated plainly rather than silently assumed: 2D's
    Neumann support (grid2d.apply_bc_2d) is zero-flux ONLY -- the 2D Laplacian
    encodes it by construction (no row replacement needed), but there is no
    2D equivalent of the 1D nonzero-flux Neumann row (apply_bc's `value`
    branch). Passing a nonzero value for a 'neumann' substrate here raises
    NotImplementedError rather than silently ignoring it or (worse) doing
    something wrong. Extending 2D to nonzero-flux Neumann would need a
    boundary-normal-aware flux row (nontrivial on a Cartesian grid at a
    corner or an embedded/staircased boundary) and its own validation; that
    is future work, not something this pass claims to have done.
    """
    if bc_specs is not None:
        missing = set(SUBSTRATES) - set(bc_specs.keys())
        if missing:
            raise ValueError(f"bc_specs is missing substrate(s): {sorted(missing)}")
        out = {}
        for sub, spec in bc_specs.items():
            if sub not in SUBSTRATES:
                raise ValueError(f"bc_specs has unknown substrate {sub!r}")
            bt, val = spec
            if bt not in ("dirichlet", "neumann"):
                raise ValueError(f"bc_specs[{sub!r}]: bc_type must be 'dirichlet' "
                                  f"or 'neumann', got {bt!r}")
            if bt == "neumann" and val != 0.0:
                raise NotImplementedError(
                    f"bc_specs[{sub!r}]: 2D Neumann only supports zero flux "
                    f"(value=0) currently; nonzero-flux Neumann in 2D is not "
                    f"implemented (see grid2d.apply_bc_2d docstring).")
            out[sub] = (bt, val)
        return out
    if bc_type not in ("dirichlet", "neumann"):
        raise ValueError("bc_type must be 'dirichlet' or 'neumann'")
    if bc_type == "neumann":
        return {sub: ("neumann", 0.0) for sub in SUBSTRATES}
    c_inf = coeffs["c_inf_hat"]
    return {sub: (bc_type, c_inf[sub]) for sub in SUBSTRATES}


def _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes):
    """Global residual for the coupled 4*Npts system.

    Boundary rows carry the algebraic BC (c_k - value = 0 for Dirichlet
    substrates, per bc_specs) and NO reaction contribution -- the 2D analogue
    of the 1D `R[sub][-1] = 0`, generalised from "the last row" to "every
    boundary node" (and now, per ITEM 1, to "every boundary node of every
    Dirichlet substrate specifically"). Interior rows, including every node
    adjacent to the boundary, keep their full reaction term (the 1D row-0 bug
    was precisely the mistake of stripping a reaction term from a row that
    was not actually a BC row).
    """
    Npts = len(next(iter(C.values())))
    R, dR = reaction_and_jacobian(C, U, coeffs)

    for sub in SUBSTRATES:
        bt, _ = bc_specs[sub]
        if bt == "dirichlet":
            R[sub][bnodes] = 0.0

    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    for k, sub in enumerate(SUBSTRATES):
        bt, val = bc_specs[sub]
        if bt == "dirichlet":
            base = k * Npts
            F[base + bnodes] = C[sub][bnodes] - val
    return F, R, dR


def _assemble_global_2d(Lap_bc, R, dR, C, bc_specs, bnodes):
    """Sparse Jacobian of _residual_2d. Diagonal (self-coupling) blocks carry
    the Laplacian; every block additionally carries the elementwise reaction
    derivative, zeroed on boundary rows (per Dirichlet substrate) so those
    rows stay the pure algebraic identity written by apply_bc_2d."""
    n = len(SUBSTRATES)
    Npts = len(next(iter(C.values())))
    idx = {sub: k for k, sub in enumerate(SUBSTRATES)}
    blocks = [[None] * n for _ in range(n)]

    for i_sub in SUBSTRATES:
        i = idx[i_sub]
        blocks[i][i] = Lap_bc[i_sub].tolil()
        bt_i, _ = bc_specs[i_sub]
        for j_sub in SUBSTRATES:
            j = idx[j_sub]
            if (i_sub, j_sub) not in dR:
                continue
            d = dR[(i_sub, j_sub)].copy()
            if bt_i == "dirichlet":
                d[bnodes] = 0.0
            block = sp.diags(d, format="lil")
            blocks[i][j] = block if blocks[i][j] is None else blocks[i][j] + block

    for i in range(n):
        for j in range(n):
            if blocks[i][j] is None:
                blocks[i][j] = sp.csr_matrix((Npts, Npts))
    return sp.bmat(blocks, format="csr")


def solve_newton_2d(coeffs: dict, U: dict, grid: Grid2D,
                     bc_type: str = "dirichlet", bc_specs: dict | None = None,
                     boundary_nodes: np.ndarray | None = None,
                     tol: float = 1e-10, maxiter: int = 60,
                     damped: bool = True, max_backtracks: int = 30,
                     c_max_factor: float = 5.0, verbose: bool = False):
    """Newton iteration for the 2D quasi-steady substrate system.

    U: dict species -> flat array of length grid.Npts (or a (Nx+1, Ny+1) field;
    it is flattened here). Returns (C, history, method) with C a dict of FLAT
    arrays, matching the 1D solver's 3-tuple return convention (see the 1D
    note on that signature: callers must unpack three values).

    ITEM 1 extension (for consistency with the 1D per-substrate interface):
    pass `bc_specs`, a dict {sub: (bc_type, value)}, to give each substrate
    its own independent bc_type. `bc_type` (the old single-string parameter)
    is still accepted and expanded into a per-substrate spec when `bc_specs`
    is not given -- exactly reproducing the old behaviour, so existing callers
    are unaffected. SCOPE LIMITATION: unlike the 1D solver, 2D Neumann only
    supports zero flux (no 2D equivalent of the 1D nonzero-flux Neumann row
    exists yet) -- see normalize_bc_specs_2d.

    `boundary_nodes` selects which nodes carry the Dirichlet condition and
    defaults to the four domain edges. Passing an explicit set allows an
    embedded boundary (used by the radial-symmetry validation, which imposes
    Dirichlet data on a staircased circle and on everything outside it).
    """
    Npts = grid.Npts
    U = {sp_name: np.asarray(v).ravel() for sp_name, v in U.items()}
    bc_specs = normalize_bc_specs_2d(coeffs, bc_type, bc_specs)
    # magnitude, not signed max -- same reason as the 1D fix (a Neumann value
    # is a signed flux, and a signed max collapses the bound)
    c_max = c_max_factor * max(abs(val) for _, val in bc_specs.values())

    bnodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)

    Lap0 = build_laplacian_2d(grid)
    Lap_bc = {sub: apply_bc_2d(Lap0, grid, bc_specs[sub][0], boundary_nodes=bnodes)
              for sub in SUBSTRATES}

    # Dirichlet substrates start at their prescribed value; Neumann (2D:
    # always zero-flux) substrates start at 1.0, not 0.0 -- see the matching
    # 1D note in elliptic._default_initial_guess for why 0.0 is a trap
    # (Monod-gating) for a substrate that co-limits multiple species' uptake.
    C = {sub: np.full(Npts, val if bt == "dirichlet" else 1.0)
         for sub, (bt, val) in bc_specs.items()}
    F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes)
    res = float(np.linalg.norm(F, ord=np.inf))
    history = [res]

    for it in range(maxiter):
        if res < tol:
            break
        J = _assemble_global_2d(Lap_bc, R, dR, C, bc_specs, bnodes)
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
            F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes)
            res = float(np.linalg.norm(F, ord=np.inf))
            accepted = True
        else:
            for _ in range(max_backtracks):
                raw = {sub: C[sub] + step * dX[k * Npts:(k + 1) * Npts]
                       for k, sub in enumerate(SUBSTRATES)}
                plausible = all(np.all(v <= c_max) for v in raw.values())
                C_try = {sub: np.clip(v, 0.0, c_max) for sub, v in raw.items()}
                F_try, R_try, dR_try = _residual_2d(C_try, U, Lap_bc, coeffs,
                                                     bc_specs, bnodes)
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
