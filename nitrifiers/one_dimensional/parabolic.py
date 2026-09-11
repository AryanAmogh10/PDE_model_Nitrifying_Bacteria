"""Stage 5: finite-volume parabolic solver for the 3 bacterial density
equations, given a fixed substrate field c:

    d(uhat_i)/dt = Dhat_i*Lap(uhat_i) + Ahat_i*div(uhat_i*grad(rhohat))
                    + uhat_i*[rhat_i*M(chat_p;Khat_ip)*M(chat_O2;Khat_iO2) - bhat_i*rhohat]

rhohat = sum of the three species. Death term is density-dependent
(bhat_i*rhohat), not a constant rate. Fully implicit backward Euler, true
Newton per step (including the cross-diffusion rho-derivative in every
column block, since rho = sum_j u_j -- needed for real convergence on sharp
fronts, not just a modified-Newton approximation). Zero-flux BCs at both ends."""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elliptic import (Grid, build_laplacian, monod, cell_volumes, face_area,
                        SPECIES, PRIMARY, SECONDARY)


def build_advection_matrix(grid: Grid, rho: np.ndarray) -> sp.csr_matrix:
    """Upwind FV discretisation of div(u*grad(rho)) for fixed rho."""
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)
    Npts = N + 1
    rows, cols, vals = [], [], []

    for i in range(N):
        area = face_area(grid, 0.5 * (r[i] + r[i + 1]))
        vel = -(rho[i + 1] - rho[i]) / h
        upwind = i if vel >= 0 else i + 1
        rows.append(i); cols.append(upwind); vals.append(-(area / V[i]) * vel)
        rows.append(i + 1); cols.append(upwind); vals.append((area / V[i + 1]) * vel)

    return sp.csr_matrix((vals, (rows, cols)), shape=(Npts, Npts))


def build_advection_rho_jacobian(grid: Grid, u_i: np.ndarray,
                                  rho: np.ndarray) -> sp.csr_matrix:
    """d/drho of (Adv(rho) @ u_i), at frozen upwind directions."""
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)
    Npts = N + 1
    rows, cols, vals = [], [], []
    for f in range(N):
        area = face_area(grid, 0.5 * (r[f] + r[f + 1]))
        a_f = area / V[f]
        c_f = area / V[f + 1]
        vel = -(rho[f + 1] - rho[f]) / h
        s = f if vel >= 0 else f + 1
        w = u_i[s] / h
        rows += [f, f, f + 1, f + 1]
        cols += [f, f + 1, f, f + 1]
        vals += [-a_f * w, a_f * w, c_f * w, -c_f * w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(Npts, Npts))


def growth_rate_field(coeffs: dict, C: dict, species: str) -> np.ndarray:
    """Growth-only part of the reaction term; death (density-dependent) is
    handled separately every step, not folded in here."""
    p_sub, s_sub = PRIMARY[species], SECONDARY[species]
    rhat = coeffs["rhat"][species]
    Kp = coeffs["Khat"][species][p_sub]
    Ks = coeffs["Khat"][species][s_sub]
    return rhat * monod(C[p_sub], Kp) * monod(C[s_sub], Ks)


def _parabolic_residual(Uk: dict, Un: dict, g: dict, Dhat: dict, Ahat: dict,
                         bhat: dict, Lap, grid: Grid, dt: float):
    rho = Uk["AOB"] + Uk["NOB"] + Uk["CMX"]
    Adv = build_advection_matrix(grid, rho)
    F = {}
    for i in SPECIES:
        F[i] = ((Uk[i] - Un[i]) / dt
                - Dhat[i] * (Lap @ Uk[i])
                - Ahat[i] * (Adv @ Uk[i])
                - (g[i] - bhat[i] * rho) * Uk[i])
    return F, rho, Adv


def solve_parabolic(coeffs: dict, C: dict, U0: dict, grid: Grid,
                     dt: float, n_steps: int, record_every: int = 1,
                     newton_tol: float = 1e-8, newton_rtol: float = 1e-6,
                     newton_maxiter: int = 30, max_backtracks: int = 20):
    Npts = grid.N + 1
    Lap = build_laplacian(grid)
    n_sp = len(SPECIES)

    Dhat = {sp_name: coeffs["Dhat"][sp_name] for sp_name in SPECIES}
    Ahat = {sp_name: coeffs["Ahat"][sp_name] for sp_name in SPECIES}
    bhat = {sp_name: coeffs["bhat"][sp_name] for sp_name in SPECIES}
    g = {sp_name: growth_rate_field(coeffs, C, sp_name) for sp_name in SPECIES}

    U = {sp_name: U0[sp_name].copy() for sp_name in SPECIES}
    snapshots = [{"t": 0.0, "U": {k: v.copy() for k, v in U.items()}}]
    mass_history = [_total_mass(grid, U)]

    I = sp.identity(Npts, format="csr")
    idx = {sp_name: k for k, sp_name in enumerate(SPECIES)}
    t = 0.0

    for step in range(1, n_steps + 1):
        Un = {sp_name: U[sp_name].copy() for sp_name in SPECIES}
        Uk = {sp_name: U[sp_name].copy() for sp_name in SPECIES}

        F, rho, Adv = _parabolic_residual(Uk, Un, g, Dhat, Ahat, bhat, Lap, grid, dt)
        res = max(np.max(np.abs(F[i])) for i in SPECIES)
        res0 = max(res, 1e-30)

        for it in range(newton_maxiter):
            if res < newton_tol or res < newton_rtol * res0:
                break
            blocks = [[None] * n_sp for _ in range(n_sp)]
            for i in SPECIES:
                bi = idx[i]
                death_diag = g[i] - bhat[i] * rho
                Mi = build_advection_rho_jacobian(grid, Uk[i], rho)
                cross = sp.diags(bhat[i] * Uk[i]) - Ahat[i] * Mi
                J_ii = (I / dt - Dhat[i] * Lap - Ahat[i] * Adv
                        - sp.diags(death_diag) + cross)
                blocks[bi][bi] = J_ii
                for kname in SPECIES:
                    if kname == i:
                        continue
                    blocks[bi][idx[kname]] = cross
            J = sp.bmat(blocks, format="csr")
            Fvec = np.concatenate([F[i] for i in SPECIES])
            dX = spla.spsolve(J, -Fvec)
            dU = {i: dX[idx[i] * Npts:(idx[i] + 1) * Npts] for i in SPECIES}

            step_len = 1.0
            improved = False
            for _ in range(max_backtracks):
                U_trial = {i: Uk[i] + step_len * dU[i] for i in SPECIES}
                F_trial, rho_t, Adv_t = _parabolic_residual(
                    U_trial, Un, g, Dhat, Ahat, bhat, Lap, grid, dt)
                res_trial = max(np.max(np.abs(F_trial[i])) for i in SPECIES)
                if np.isfinite(res_trial) and res_trial < res:
                    Uk, F, rho, Adv, res = U_trial, F_trial, rho_t, Adv_t, res_trial
                    improved = True
                    break
                step_len *= 0.5
            if not improved:
                break

        if res > max(1e-5, 1e-3 * res0):
            warnings.warn(
                f"parabolic Newton stalled at step {step} with large residual "
                f"(res={res:.3e}, start-of-step {res0:.3e}).", RuntimeWarning)

        for i in SPECIES:
            neg = -np.min(Uk[i])
            if neg > 1e-8:
                warnings.warn(f"parabolic: clipped negative u_{i}={neg:.2e} at "
                               f"step {step}", RuntimeWarning)
            Uk[i] = np.maximum(Uk[i], 0.0)

        U = Uk
        t += dt
        mass_history.append(_total_mass(grid, U))
        if step % record_every == 0:
            snapshots.append({"t": t, "U": {k: v.copy() for k, v in U.items()}})

    return U, mass_history, snapshots


def _total_mass(grid: Grid, U: dict) -> float:
    V = cell_volumes(grid)
    return float(sum(np.sum(V * U[sp_name]) for sp_name in SPECIES))
