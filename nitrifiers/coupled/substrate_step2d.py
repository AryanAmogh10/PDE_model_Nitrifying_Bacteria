"""Time-dependent substrate step on the 2D grid: the slow-fast solver drops
d(c)/dt (quasi-steady, epsilon -> 0); this keeps it,

    eps * d(c_j)/dt = Lap(c_j) + R_j(u, c)

and advances it by one backward-Euler step, solved with Newton. Same
reaction terms, same boundary machinery as elliptic2d; the only change is
the mass term. That term is also why there is no plausibility bound here:
a step of size dt always has a solution (the eps/dt diagonal makes the
Jacobian nonsingular), so a boundary influx larger than the consumption
capacity simply accumulates substrate over time instead of leaving Newton
chasing a steady state that does not exist."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..nondim import SUBSTRATES
from ..two_dimensional.grid2d import Grid2D, build_laplacian_2d, apply_bc_2d, boundary_flux_coefficients
from ..two_dimensional.elliptic2d import normalize_bc_specs_2d, _residual_2d, _assemble_global_2d


class SubstrateOperators2D:
    """Grid-dependent matrices assembled once and reused every step."""

    def __init__(self, grid: Grid2D, coeffs: dict, bc_specs: dict,
                 boundary_nodes: np.ndarray | None = None):
        self.grid = grid
        self.bc_specs = normalize_bc_specs_2d(coeffs, "dirichlet", bc_specs)
        self.bnodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)
        self.flux_coef = boundary_flux_coefficients(grid)
        Lap0 = build_laplacian_2d(grid)
        self.Lap_bc = {sub: apply_bc_2d(Lap0, grid, self.bc_specs[sub][0], boundary_nodes=self.bnodes)
                       for sub in SUBSTRATES}
        # mass term applies on every row that is a finite-volume balance,
        # i.e. everything except Dirichlet rows (those are c = value)
        Npts = grid.Npts
        self.mass_mask = {}
        for sub in SUBSTRATES:
            m = np.ones(Npts)
            if self.bc_specs[sub][0] == "dirichlet":
                m[self.bnodes] = 0.0
            self.mass_mask[sub] = m


def step_substrate_2d(coeffs: dict, U: dict, C_old: dict, ops: SubstrateOperators2D,
                      eps: float, dt: float, tol: float = 1e-8, maxiter: int = 50,
                      max_backtracks: int = 20):
    """One backward-Euler step of eps*dc/dt = Lap c + R(u, c), all four
    substrates coupled, Newton from C_old. Returns (C_new, n_iter, residual,
    method) with method in {"newton", "newton_stalled"}."""
    grid, bc_specs, bnodes, flux_coef, Lap_bc = ops.grid, ops.bc_specs, ops.bnodes, ops.flux_coef, ops.Lap_bc
    Npts = grid.Npts
    a = eps / dt
    U = {s: np.asarray(v).ravel() for s, v in U.items()}
    C_old = {s: np.asarray(v).ravel() for s, v in C_old.items()}
    mass_vec = np.concatenate([a * ops.mass_mask[sub] for sub in SUBSTRATES])
    mass_diag = sp.diags(mass_vec, format="csr")
    C_old_vec = np.concatenate([C_old[sub] for sub in SUBSTRATES])

    def residual(C):
        F, R, dR = _residual_2d(C, U, Lap_bc, coeffs, bc_specs, bnodes, flux_coef)
        C_vec = np.concatenate([C[sub] for sub in SUBSTRATES])
        F = F - mass_vec * (C_vec - C_old_vec)
        return F, R, dR

    C = {s: v.copy() for s, v in C_old.items()}
    F, R, dR = residual(C)
    res = float(np.linalg.norm(F, ord=np.inf))
    # residual entries scale with eps/dt * c, so an absolute tol would be
    # unreachable in round-off for small dt; measure against that scale
    c_scale = max(1.0, float(np.max(np.abs(C_old_vec))))
    tol_eff = tol * max(1.0, a * c_scale)
    n_iter = 0
    for it in range(maxiter):
        if res < tol_eff:
            break
        J = _assemble_global_2d(Lap_bc, R, dR, C, bc_specs, bnodes, flux_coef) - mass_diag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            dX = spla.spsolve(J.tocsc(), -F)
        if not np.all(np.isfinite(dX)):
            return C, n_iter, res, "newton_stalled"
        step = 1.0
        accepted = False
        for _ in range(max_backtracks):
            C_try = {sub: np.maximum(C[sub] + step * dX[k * Npts:(k + 1) * Npts], 0.0)
                     for k, sub in enumerate(SUBSTRATES)}
            F_try, R_try, dR_try = residual(C_try)
            res_try = float(np.linalg.norm(F_try, ord=np.inf))
            if np.isfinite(res_try) and res_try < res:
                C, F, R, dR, res = C_try, F_try, R_try, dR_try, res_try
                accepted = True
                break
            step *= 0.5
        n_iter += 1
        if not accepted:
            return C, n_iter, res, "newton_stalled"
    return C, n_iter, res, "newton"


def term_magnitudes(coeffs: dict, U: dict, C: dict, ops: SubstrateOperators2D,
                    C_prev: dict | None = None, eps: float | None = None, dt: float | None = None):
    """Volume-weighted L1 norms of the three terms of eps*c_t = Lap c + R,
    per substrate, on the finite-volume rows. Note |R| / |Lap c| is ~1 near
    ANY quasi-steady state (that is what quasi-steady means), so it says
    nothing about timescales. The informative ratio is
    transient / diffusion = eps*|c_t| / |Lap c|: << 1 means the substrate is
    slaved to u (QSSA valid), ~1 means the substrate's own dynamics matter."""
    grid = ops.grid
    U = {s: np.asarray(v).ravel() for s, v in U.items()}
    C = {s: np.asarray(v).ravel() for s, v in C.items()}
    _, R, _ = _residual_2d(C, U, ops.Lap_bc, coeffs, ops.bc_specs, ops.bnodes, ops.flux_coef)
    out = {}
    for sub in SUBSTRATES:
        m = ops.mass_mask[sub] > 0
        diff = ops.Lap_bc[sub] @ C[sub]
        rec = {"diffusion": float(np.sum(grid.Vflat[m] * np.abs(diff[m]))),
               "reaction": float(np.sum(grid.Vflat[m] * np.abs(R[sub][m])))}
        if C_prev is not None and eps is not None and dt is not None:
            ct = eps * (C[sub] - np.asarray(C_prev[sub]).ravel()) / dt
            rec["transient"] = float(np.sum(grid.Vflat[m] * np.abs(ct[m])))
        out[sub] = rec
    return out
