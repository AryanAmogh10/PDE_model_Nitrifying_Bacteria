"""Stage 3 in 2D: quasi-steady substrate solver on a 2D Cartesian grid.
Direct extension of elliptic.py, same Newton/backtracking/plausibility-bound
scheme, boundary is a whole edge instead of one row. No relaxation fallback
here (no 2D PTC solver yet) -- a stuck solve returns method="newton_stalled"."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..nondim import SUBSTRATES
from ..one_dimensional.elliptic import (reaction_and_jacobian, BC_TYPES,
                                        _robin_pair, bc_concentration_scale)
from .grid2d import Grid2D, build_laplacian_2d, apply_bc_2d, boundary_flux_coefficients


def normalize_bc_specs_2d(coeffs: dict, bc_type: str, bc_specs: dict | None) -> dict:
    """2D version of elliptic.normalize_bc_specs. Neumann flux and robin
    transfer apply uniformly on all four edges, not per-edge."""
    if bc_specs is not None:
        missing = set(SUBSTRATES) - set(bc_specs.keys())
        if missing:
            raise ValueError(f"bc_specs is missing substrate(s): {sorted(missing)}")
        out = {}
        for sub, spec in bc_specs.items():
            if sub not in SUBSTRATES:
                raise ValueError(f"bc_specs has unknown substrate {sub!r}")
            bt, val = spec
            if bt not in BC_TYPES:
                raise ValueError(f"bc_specs[{sub!r}]: bc_type must be one of "
                                  f"{BC_TYPES}, got {bt!r}")
            out[sub] = (bt, _robin_pair(sub, val) if bt == "robin" else val)
        return out
    if bc_type == "robin":
        raise ValueError("bc_type='robin' needs a per-substrate bc_specs dict: "
                          "there is no Bi in coeffs to fall back on. Pass "
                          "bc_specs={sub: ('robin', (Bi, c_inf)), ...}.")
    if bc_type not in BC_TYPES:
        raise ValueError(f"bc_type must be one of {BC_TYPES}")
    if bc_type == "neumann":
        return {sub: ("neumann", 0.0) for sub in SUBSTRATES}
    c_inf = coeffs["c_inf_hat"]
    return {sub: (bc_type, c_inf[sub]) for sub in SUBSTRATES}


def _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes, flux_coef=None):
    Npts = len(next(iter(C.values())))
    R, dR = reaction_and_jacobian(C, U, coeffs)

    for sub in SUBSTRATES:
        bt, _ = bc_specs[sub]
        if bt == "dirichlet":
            R[sub][bnodes] = 0.0

    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    for k, sub in enumerate(SUBSTRATES):
        bt, val = bc_specs[sub]
        base = k * Npts
        if bt == "dirichlet":
            F[base + bnodes] = C[sub][bnodes] - val
        elif bt == "robin":
            Bi, c_inf = val
            F[base + bnodes] += flux_coef[bnodes] * Bi * (C[sub][bnodes] - c_inf)
        elif val != 0.0:
            F[base + bnodes] += flux_coef[bnodes] * val
    return F, R, dR


def _assemble_global_2d(Lap_bc, R, dR, C, bc_specs, bnodes, flux_coef=None):
    n = len(SUBSTRATES)
    Npts = len(next(iter(C.values())))
    idx = {sub: k for k, sub in enumerate(SUBSTRATES)}
    blocks = [[None] * n for _ in range(n)]

    for i_sub in SUBSTRATES:
        i = idx[i_sub]
        blocks[i][i] = Lap_bc[i_sub].tolil()
        bt_i, val_i = bc_specs[i_sub]
        if bt_i == "robin":
            Bi, _ = val_i
            robin_diag = np.zeros(Npts)
            robin_diag[bnodes] = flux_coef[bnodes] * Bi
            blocks[i][i] = blocks[i][i] + sp.diags(robin_diag, format="lil")
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
    Npts = grid.Npts
    U = {sp_name: np.asarray(v).ravel() for sp_name, v in U.items()}
    bc_specs = normalize_bc_specs_2d(coeffs, bc_type, bc_specs)
    c_max = c_max_factor * max(bc_concentration_scale(bt, val)
                                for bt, val in bc_specs.values())

    bnodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)
    flux_coef = boundary_flux_coefficients(grid)

    Lap0 = build_laplacian_2d(grid)
    Lap_bc = {sub: apply_bc_2d(Lap0, grid, bc_specs[sub][0], boundary_nodes=bnodes)
              for sub in SUBSTRATES}

    # dirichlet starts at its value, robin at its bulk c_inf, neumann at 1.0
    # (never 0.0 -- Monod(0)=0 would zero the reaction Jacobian at iterate 0)
    C = {}
    for sub, (bt, val) in bc_specs.items():
        if bt == "dirichlet":
            start = val
        elif bt == "robin":
            start = val[1] if val[1] > 0.0 else 1.0
        else:
            start = 1.0
        C[sub] = np.full(Npts, start)
    F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes, flux_coef)
    res = float(np.linalg.norm(F, ord=np.inf))
    history = [res]

    for it in range(maxiter):
        if res < tol:
            break
        J = _assemble_global_2d(Lap_bc, R, dR, C, bc_specs, bnodes, flux_coef)
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
            F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes, flux_coef)
            res = float(np.linalg.norm(F, ord=np.inf))
            accepted = True
        else:
            for _ in range(max_backtracks):
                raw = {sub: C[sub] + step * dX[k * Npts:(k + 1) * Npts]
                       for k, sub in enumerate(SUBSTRATES)}
                plausible = all(np.all(v <= c_max) for v in raw.values())
                C_try = {sub: np.clip(v, 0.0, c_max) for sub, v in raw.items()}
                F_try, R_try, dR_try = _residual_2d(C_try, U, Lap_bc, coeffs,
                                                     bc_specs, bnodes, flux_coef)
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
