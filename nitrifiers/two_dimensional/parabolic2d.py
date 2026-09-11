"""Stage 5 in 2D: fully-implicit bacterial density evolution on a 2D
Cartesian grid, direct extension of parabolic.py (same scheme, true Newton
per step including the cross-diffusion rho-derivative). Zero-flux on all
four edges. Unlike 1D's build_advection_matrix, the control volume here
(dx_i*dy_j) is exact, so this operator conserves mass exactly at any
resolution -- the 1D version's r^p*h approximation is still first-order only."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid2d import Grid2D, build_laplacian_2d
from ..one_dimensional.parabolic import SPECIES, growth_rate_field


def _face_index_arrays(grid: Grid2D):
    Nx, Ny = grid.Nx, grid.Ny
    ny1 = Ny + 1
    ix, jx = np.meshgrid(np.arange(Nx), np.arange(ny1), indexing="ij")
    xP = (ix * ny1 + jx).ravel()
    xQ = ((ix + 1) * ny1 + jx).ravel()
    xinvP = (1.0 / grid.dx[ix]).ravel()
    xinvQ = (1.0 / grid.dx[ix + 1]).ravel()
    iy, jy = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), indexing="ij")
    yP = (iy * ny1 + jy).ravel()
    yQ = (iy * ny1 + jy + 1).ravel()
    yinvP = (1.0 / grid.dy[jy]).ravel()
    yinvQ = (1.0 / grid.dy[jy + 1]).ravel()
    return (xP, xQ, xinvP, xinvQ), (yP, yQ, yinvP, yinvQ)


def build_advection_matrix_2d(grid: Grid2D, rho: np.ndarray) -> sp.csr_matrix:
    rho = np.asarray(rho).ravel()
    (xP, xQ, xiP, xiQ), (yP, yQ, yiP, yiQ) = _face_index_arrays(grid)
    rows, cols, vals = [], [], []
    for (P, Q, invP, invQ, hstep) in ((xP, xQ, xiP, xiQ, grid.hx),
                                       (yP, yQ, yiP, yiQ, grid.hy)):
        vel = -(rho[Q] - rho[P]) / hstep
        up = np.where(vel >= 0, P, Q)
        rows.append(P); cols.append(up); vals.append(-invP * vel)
        rows.append(Q); cols.append(up); vals.append(+invQ * vel)
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(grid.Npts, grid.Npts))


def build_advection_rho_jacobian_2d(grid: Grid2D, u_i: np.ndarray,
                                     rho: np.ndarray) -> sp.csr_matrix:
    u_i = np.asarray(u_i).ravel()
    rho = np.asarray(rho).ravel()
    (xP, xQ, xiP, xiQ), (yP, yQ, yiP, yiQ) = _face_index_arrays(grid)
    rows, cols, vals = [], [], []
    for (P, Q, invP, invQ, hstep) in ((xP, xQ, xiP, xiQ, grid.hx),
                                       (yP, yQ, yiP, yiQ, grid.hy)):
        vel = -(rho[Q] - rho[P]) / hstep
        up = np.where(vel >= 0, P, Q)
        w = u_i[up] / hstep
        rows += [P, P, Q, Q]
        cols += [P, Q, P, Q]
        vals += [-invP * w, +invP * w, +invQ * w, -invQ * w]
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(grid.Npts, grid.Npts))


def total_mass_2d(grid: Grid2D, U: dict) -> float:
    return float(sum(np.sum(grid.Vflat * np.asarray(U[s]).ravel()) for s in SPECIES))


def _residual_2d(Uk, Un, g, Dhat, Ahat, bhat, Lap, grid, dt):
    rho = Uk["AOB"] + Uk["NOB"] + Uk["CMX"]
    Adv = build_advection_matrix_2d(grid, rho)
    F = {}
    for i in SPECIES:
        F[i] = ((Uk[i] - Un[i]) / dt
                - Dhat[i] * (Lap @ Uk[i])
                - Ahat[i] * (Adv @ Uk[i])
                - (g[i] - bhat[i] * rho) * Uk[i])
    return F, rho, Adv


def solve_parabolic_2d(coeffs: dict, C: dict, U0: dict, grid: Grid2D,
                        dt: float, n_steps: int,
                        newton_tol: float = 1e-8, newton_rtol: float = 1e-6,
                        newton_maxiter: int = 30, max_backtracks: int = 20):
    Npts = grid.Npts
    Lap = build_laplacian_2d(grid)
    n_sp = len(SPECIES)
    idx = {s: k for k, s in enumerate(SPECIES)}

    Dhat = {s: coeffs["Dhat"][s] for s in SPECIES}
    Ahat = {s: coeffs["Ahat"][s] for s in SPECIES}
    bhat = {s: coeffs["bhat"][s] for s in SPECIES}
    g = {s: growth_rate_field(coeffs, C, s) for s in SPECIES}

    U = {s: np.asarray(U0[s]).ravel().copy() for s in SPECIES}
    I = sp.identity(Npts, format="csr")
    mass_history = [total_mass_2d(grid, U)]

    for step in range(1, n_steps + 1):
        Un = {s: U[s].copy() for s in SPECIES}
        Uk = {s: U[s].copy() for s in SPECIES}
        F, rho, Adv = _residual_2d(Uk, Un, g, Dhat, Ahat, bhat, Lap, grid, dt)
        res = max(np.max(np.abs(F[i])) for i in SPECIES)
        res0 = max(res, 1e-30)

        for _ in range(newton_maxiter):
            if res < newton_tol or res < newton_rtol * res0:
                break
            blocks = [[None] * n_sp for _ in range(n_sp)]
            for i in SPECIES:
                bi = idx[i]
                Mi = build_advection_rho_jacobian_2d(grid, Uk[i], rho)
                cross = sp.diags(bhat[i] * Uk[i]) - Ahat[i] * Mi
                blocks[bi][bi] = (I / dt - Dhat[i] * Lap - Ahat[i] * Adv
                                   - sp.diags(g[i] - bhat[i] * rho) + cross)
                for kname in SPECIES:
                    if kname != i:
                        blocks[bi][idx[kname]] = cross
            J = sp.bmat(blocks, format="csr")
            dX = spla.spsolve(J, -np.concatenate([F[i] for i in SPECIES]))
            if not np.all(np.isfinite(dX)):
                warnings.warn(f"parabolic2d: non-finite Newton update at step {step}",
                               RuntimeWarning)
                break
            dU = {i: dX[idx[i] * Npts:(idx[i] + 1) * Npts] for i in SPECIES}

            step_len, improved = 1.0, False
            for _ in range(max_backtracks):
                U_try = {i: Uk[i] + step_len * dU[i] for i in SPECIES}
                F_t, rho_t, Adv_t = _residual_2d(U_try, Un, g, Dhat, Ahat, bhat,
                                                  Lap, grid, dt)
                res_t = max(np.max(np.abs(F_t[i])) for i in SPECIES)
                if np.isfinite(res_t) and res_t < res:
                    Uk, F, rho, Adv, res = U_try, F_t, rho_t, Adv_t, res_t
                    improved = True
                    break
                step_len *= 0.5
            if not improved:
                break

        if res > max(1e-5, 1e-3 * res0):
            warnings.warn(f"parabolic2d Newton stalled at step {step} "
                           f"(res={res:.3e}, start {res0:.3e}).", RuntimeWarning)
        for i in SPECIES:
            neg = -np.min(Uk[i])
            if neg > 1e-8:
                warnings.warn(f"parabolic2d: clipped negative u_{i}={neg:.2e} at "
                               f"step {step}", RuntimeWarning)
            Uk[i] = np.maximum(Uk[i], 0.0)

        U = Uk
        mass_history.append(total_mass_2d(grid, U))

    return U, mass_history
