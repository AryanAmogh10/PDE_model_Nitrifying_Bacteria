"""
Stage 5: finite-volume parabolic solver for the 3 non-dimensional bacterial
density equations (AOB, NOB, CMX), given a fixed substrate field c. Solves

    d(uhat_i)/d(that) = Dhat_i * Lap(uhat_i) + Ahat_i * div(uhat_i * grad(rhohat))
                         + uhat_i * [rhat_i * M(chat_p;Khat_ip) * M(chat_O2;Khat_iO2) - bhat_i * rhohat]

on the same 1D radial/slab grid used in elliptic.py/relaxation.py, with
rhohat := uhat_AOB + uhat_NOB + uhat_CMX.

IMPORTANT: the death term is bhat_i * rhohat (density-DEPENDENT crowding
mortality), not a plain constant bhat_i. This was verified directly against
the rendered PDF of arXiv:2512.13156 eq. (2.1) and eq. (2.5) (page images,
not the earlier pdftotext extraction, which silently dropped the rho symbol
throughout that document -- an easy trap, since "- b_i" and "- b_i*rho" look
identical once rho is missing). An earlier version of this module used a
constant bhat_i, discovered to be wrong by cross-checking the PDF images
directly rather than trusting the text extraction a second time. With a
constant death term biomass has no self-limiting mechanism in substrate-rich
regions (only substrate depletion and mass-conserving cross-diffusion act as
brakes) and can in principle grow unbounded; with bhat_i*rhohat, growth
saturates near rhohat ~ rhat_i*M_i/bhat_i, as intended.

Discretisation:
    - Diffusion term: reuses elliptic.build_laplacian (same conservative
      radial finite-volume Laplacian).
    - Cross-diffusion (advection) term: a first-order upwind finite-volume
      scheme for the flux -grad(rhohat) (bacteria move down the total-density
      gradient, away from crowding, matching the sign convention in
      arXiv:2512.13156 Sec. 2.1/3.1 and its Appendix A discretisation),
      normalised by the SAME exact control volumes as build_laplacian
      (elliptic.cell_volumes), so the scheme conserves mass EXACTLY under
      zero-flux boundaries with the reaction term switched off.

      This last point was for a long time false in a way no test caught. The
      operator used to normalise by the interior shortcut V_i ~= r_i^p*h,
      which build_laplacian had already been corrected away from. That is
      wrong by O(h^2/r_i^2) on the first interior rows and, more seriously, by
      a factor of ~2 at the OUTER boundary node (full cell instead of the half
      cell that actually exists) in every geometry including the slab. The
      operator still telescoped exactly against its OWN volumes, so it looked
      self-consistent -- but it meant diffusion conserved one measure and
      advection another, and the coupled scheme conserved neither, with an
      O(1) discrepancy that did not vanish under refinement. The regression
      test missed it because its probe field u = sin(pi*r) vanishes at r=1,
      exactly the node carrying the factor-of-2 error; a probe that is nonzero
      there exposes a ~100x larger, non-converging defect. The test now uses
      such a probe and asserts EXACT conservation.
    - Reaction term u_i*f_i(c, rhohat): the growth part g_i (Monod, function
      of the fixed c only) is precomputed once; the death part bhat_i*rhohat
      depends on rhohat = sum_i u_i and is treated FULLY IMPLICITLY (see time
      stepping).
    - Time stepping: FULLY IMPLICIT backward Euler, solved each step by a
      (modified) Newton iteration on the coupled 3*(N+1) system for
      (u_AOB, u_NOB, u_CMX). Both the density-dependent death term
      -bhat_i*rhohat and the cross-diffusion operator Adv(rhohat) use rhohat
      at the NEW time level (not lagged). The per-step residual for species i
      is

          F_i(U) = (u_i - u_i^n)/dt - Dhat_i*Lap*u_i - Ahat_i*Adv(rho)*u_i
                    - (g_i - bhat_i*rho) elementwise* u_i,     rho = sum_j u_j,

      and the FULL (true) Newton Jacobian blocks (i = row species, k = col
      species; M_i := d[Adv(rho)@u_i]/drho from build_advection_rho_jacobian)
      are

          J_ii = I/dt - Dhat_i*Lap - Ahat_i*Adv(rho)
                  - diag(g_i - bhat_i*rho) + bhat_i*diag(u_i) - Ahat_i*M_i
          J_ik = bhat_i*diag(u_i) - Ahat_i*M_i    (i != k)

      Both the death-term coupling bhat_i*diag(u_i) and the cross-diffusion
      rho-derivative -Ahat_i*M_i appear in every column-species block because
      rho = sum_j u_j. An earlier version omitted the -Ahat_i*M_i term (a
      *modified* Newton); its residual was still exact, so it converged to the
      right solution on smooth/uniform fields, but on sharp density fronts
      with strong cross-diffusion it stalled at ~1e-4 residual instead of
      tol. Including M_i (verified against a finite-difference Jacobian)
      restores true Newton convergence on fronts. A backtracking line search
      on the residual norm keeps the step robust. This whole scheme replaces
      a still-earlier single-lag (rhohat^n) semi-implicit scheme that was both
      unstable AND converged to the wrong equilibrium at moderate density once
      the death term was made correctly density-dependent.

    Boundary conditions: homogeneous Neumann (zero-flux) for all u_i at both
    r=0 (natural, from the symmetric discretisation) and r=1 (no bacteria
    leave the aggregate), matching the Neumann BCs used throughout
    arXiv:2512.13156.

STABILITY NOTE: an earlier version of this module used a single-lag
(rhohat^n) semi-implicit scheme. Once the death term was corrected to be
density-dependent (-bhat_i*rhohat, matching arXiv:2512.13156 eq. (2.1)/(2.5),
verified against the rendered PDF), that lagged scheme became unstable AND
converged to the wrong equilibrium at moderate density -- because rhohat can
now grow to O(1) or larger before self-limiting, a regime the earlier
(incorrect, constant-b) model never reached. The fully-implicit Newton scheme
above resolves this: it is verified stable at large dt and converges to the
analytically-correct equilibrium (for uniform, saturating substrate and
symmetric species, rho -> g_i/bhat_i per grid point; checked in
tests/test_parabolic.py::test_uniform_case_reaches_analytic_equilibrium).
"""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elliptic import Grid, build_laplacian, monod, cell_volumes, face_area

SPECIES = ("AOB", "NOB", "CMX")
PRIMARY = {"AOB": "NH4", "NOB": "NO2", "CMX": "NH4"}
SECONDARY = {"AOB": "O2", "NOB": "O2", "CMX": "O2"}


def build_advection_matrix(grid: Grid, rho: np.ndarray) -> sp.csr_matrix:
    """First-order upwind finite-volume discretisation of
    div(u * grad(rho)) for a FIXED rho field (linear operator in u).
    Sign convention: bacteria move down the rho gradient (away from
    crowding); see module docstring."""
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)   # EXACT volumes -- shared with build_laplacian
    Npts = N + 1
    rows, cols, vals = [], [], []

    for i in range(N):  # face between node i and i+1
        area = face_area(grid, 0.5 * (r[i] + r[i + 1]))
        vel = -(rho[i + 1] - rho[i]) / h  # positive = flow toward +r
        upwind = i if vel >= 0 else i + 1

        # cell i's EAST face and cell (i+1)'s WEST face are the same face, so
        # the two contributions carry equal-and-opposite flux and cancel in the
        # V-weighted sum -> exact conservation. The special-cased row 0
        # coefficient 2*(p+1)/h that used to live here is not lost: it is
        # exactly face_area(h/2)/V[0], so the general formula reproduces it.
        rows.append(i); cols.append(upwind); vals.append(-(area / V[i]) * vel)
        rows.append(i + 1); cols.append(upwind); vals.append((area / V[i + 1]) * vel)

    return sp.csr_matrix((vals, (rows, cols)), shape=(Npts, Npts))


def build_advection_rho_jacobian(grid: Grid, u_i: np.ndarray,
                                  rho: np.ndarray) -> sp.csr_matrix:
    """d/drho of (Adv(rho) @ u_i), the derivative of the cross-diffusion
    operator applied to a fixed species field u_i with respect to the
    total-density field rho, at frozen upwind directions.

    Since the advection flux vel_f = -(rho[f+1]-rho[f])/h enters linearly and
    the coefficient on each face is (area factor)*vel_f (upwind held fixed),
    the derivative is the sparse matrix M with, per face f (w_f := u_i[s_f]/h,
    s_f = upwind node, a_f/c_f the same row-i / row-(i+1) area factors as in
    build_advection_matrix):

        M[f,   f]   += -a_f w_f ;  M[f,   f+1] += +a_f w_f
        M[f+1, f]   += +c_f w_f ;  M[f+1, f+1] += -c_f w_f

    Verified against a finite-difference Jacobian to relative error ~1e-9.
    Because rho = sum_j u_j (so d rho / d u_k = 1 for every species k), this
    same M contributes to *every* column-species block of the Newton Jacobian
    for the residual of species i -- i.e. J_ik gets -Ahat_i*M for all k. Its
    omission was what made the earlier scheme only a *modified* Newton, which
    stalled at ~1e-4 residual on sharp fronts; including it restores true
    Newton convergence there."""
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)   # EXACT volumes -- must match build_advection_matrix
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
    """g_i(c) = rhat_i * M(c_p;Khat_ip) * M(c_O2;Khat_iO2), the growth-only
    part of the per-capita reaction term, evaluated on the fixed substrate
    field C (dict of arrays keyed NH4/NO2/NO3/O2). Does NOT include the death
    term -- that is bhat_i*rhohat (density-dependent, see module docstring)
    and must be recomputed every step from the current rhohat, not folded in
    here once."""
    p_sub, s_sub = PRIMARY[species], SECONDARY[species]
    rhat = coeffs["rhat"][species]
    Kp = coeffs["Khat"][species][p_sub]
    Ks = coeffs["Khat"][species][s_sub]
    return rhat * monod(C[p_sub], Kp) * monod(C[s_sub], Ks)


def _parabolic_residual(Uk: dict, Un: dict, g: dict, Dhat: dict, Ahat: dict,
                         bhat: dict, Lap, grid: Grid, dt: float):
    """Fully-implicit backward-Euler residual F_i(Uk) for each species, plus
    the pieces (rho, Adv) reused when assembling the Jacobian. See the module
    docstring for the exact form."""
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
    """Advance u_AOB, u_NOB, u_CMX forward n_steps of size dt by FULLY IMPLICIT
    backward Euler (modified Newton per step, see module docstring), for a
    fixed substrate field C. Returns final U dict, the total-mass history
    (sum_i integral of u_i, trapezoid-in-r with radial volume weighting), and
    a list of recorded U snapshots every `record_every` steps (including the
    initial condition)."""
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
        Uk = {sp_name: U[sp_name].copy() for sp_name in SPECIES}  # Newton starts at u^n

        F, rho, Adv = _parabolic_residual(Uk, Un, g, Dhat, Ahat, bhat, Lap, grid, dt)
        res = max(np.max(np.abs(F[i])) for i in SPECIES)
        res0 = max(res, 1e-30)

        for it in range(newton_maxiter):
            # converged: absolute floor, or residual reduced by newton_rtol
            # relative to the start-of-step residual (near equilibrium the
            # start-of-step residual is already tiny, so this triggers fast).
            if res < newton_tol or res < newton_rtol * res0:
                break
            # Assemble the full (true) Newton Jacobian. For the residual of
            # species i, every column-species k gets the cross-diffusion
            # rho-derivative -Ahat_i*M_i (since rho = sum_j u_j), plus the
            # death-term coupling bhat_i*diag(u_i); the i==i block also carries
            # the time, diffusion, direct-advection and reaction-diagonal terms.
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

            # backtracking line search on the residual inf-norm
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
                # line search cannot reduce the residual further -> we are at
                # the achievable floor for this step (this is the normal way a
                # step near equilibrium terminates, not a failure); stop
                # iterating and accept the current iterate.
                break

        # only a genuinely large leftover residual is worth warning about
        if res > max(1e-5, 1e-3 * res0):
            warnings.warn(
                f"parabolic Newton stalled at step {step} with large residual "
                f"(res={res:.3e}, start-of-step {res0:.3e}).", RuntimeWarning)

        # final safety clip; with the implicit scheme any negativity is a tiny
        # rounding effect, not the O(1) undershoot the old lagged scheme could
        # produce -- warn if it is ever non-trivial.
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
    """Control-volume-weighted total bacterial mass, summed over species.

    Uses the SAME exact volumes the operators are normalised by (see
    elliptic.cell_volumes). It previously used the approximate weights
    r^p*h with an ad-hoc halving of the last entry, which did not match
    either operator's normalisation -- so the reported "mass" was not the
    quantity the discrete scheme actually conserves.
    """
    V = cell_volumes(grid)
    return float(sum(np.sum(V * U[sp_name]) for sp_name in SPECIES))
