"""Stage 3: elliptic (quasi-steady-state) solver for the 4 substrate equations,
given a fixed bacterial density profile. geometry='slab' (p=0), 'radial' with
p=1 (cylindrical) or p=2 (spherical, default). BCs at rhat=1: 'dirichlet',
'neumann' or 'robin'; rhat=0 always uses symmetry (zero-flux)."""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..nondim import SUBSTRATES, SPECIES, PRIMARY, SECONDARY  # noqa: F401

_EPS_FLOOR = 1e-12


def monod(c, K):
    c = np.maximum(c, 0.0)
    return c / (K + c + _EPS_FLOOR)


def dmonod(c, K):
    c = np.maximum(c, 0.0)
    return K / (K + c + _EPS_FLOOR) ** 2


class Grid:
    def __init__(self, N: int = 100, geometry: str = "radial", p: int = 2):
        if geometry not in ("slab", "radial"):
            raise ValueError("geometry must be 'slab' or 'radial'")
        self.geometry = geometry
        self.p = 0 if geometry == "slab" else p
        self.N = N
        self.h = 1.0 / N
        self.r = np.linspace(0.0, 1.0, N + 1)


def face_area(grid: Grid, r_face: float) -> float:
    p = grid.p
    return 0.0 if (p > 0 and r_face <= 0.0) else r_face ** p


def cell_volumes(grid: Grid) -> np.ndarray:
    """Exact control-volume size per node. Shared by build_laplacian and the
    parabolic.py operators -- they must all use the same measure or the
    conservation identity doesn't hold."""
    N, h, p = grid.N, grid.h, grid.p
    r = grid.r

    def vol(r_w, r_e):
        return (r_e - r_w) if p == 0 else (r_e ** (p + 1) - r_w ** (p + 1)) / (p + 1)

    V = np.empty(N + 1)
    V[0] = vol(0.0, 0.5 * h)
    for i in range(1, N):
        V[i] = vol(r[i] - 0.5 * h, r[i] + 0.5 * h)
    V[N] = vol(r[N] - 0.5 * h, r[N])
    return V


def build_laplacian(grid: Grid) -> sp.csr_matrix:
    """Conservative FV discretisation of Lap(c) = (1/r^p) d/dr(r^p dc/dr)."""
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)
    rows, cols, vals = [], [], []

    w_e0 = face_area(grid, 0.5 * h) / (h * V[0])
    rows += [0, 0]
    cols += [0, 1]
    vals += [-w_e0, w_e0]

    for i in range(1, N):
        w_e = face_area(grid, r[i] + 0.5 * h) / (h * V[i])
        w_w = face_area(grid, r[i] - 0.5 * h) / (h * V[i])
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [w_w, -(w_e + w_w), w_e]

    # row N is zero-flux by default; apply_bc overwrites it for dirichlet/neumann
    w_w = face_area(grid, r[N] - 0.5 * h) / (h * V[N])
    rows += [N, N]
    cols += [N - 1, N]
    vals += [w_w, -w_w]

    return sp.csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))


BC_TYPES = ("dirichlet", "neumann", "robin")


def _robin_pair(sub, val):
    try:
        Bi, c_inf = val
    except (TypeError, ValueError):
        raise ValueError(f"bc_specs[{sub!r}]: robin value must be a (Bi, c_inf) "
                          f"pair, got {val!r}") from None
    Bi, c_inf = float(Bi), float(c_inf)
    if Bi < 0.0:
        raise ValueError(f"bc_specs[{sub!r}]: robin Bi must be >= 0, got {Bi}")
    return Bi, c_inf


def bc_concentration_scale(bc_type: str, value) -> float:
    # robin's concentration scale is c_inf, not Bi (a transfer coefficient)
    if bc_type == "robin":
        return abs(value[1])
    return abs(value)


def bc_row_target(bc_type: str, value) -> float:
    if bc_type == "dirichlet":
        return value
    if bc_type == "robin":
        Bi, c_inf = value
        return Bi * c_inf
    return -value   # neumann


def apply_bc(Lap: sp.csr_matrix, rhs: np.ndarray, grid: Grid, bc_type: str, value):
    """'dirichlet': c[N]=value. 'neumann': -dc/dr|_1=value. 'robin':
    -dc/dr|_1 = Bi*(c[N]-c_inf), value=(Bi, c_inf); Bi=0 reduces to
    zero-flux neumann exactly, Bi->inf approaches dirichlet c=c_inf."""
    N, h = grid.N, grid.h
    Lap = Lap.tolil()
    if bc_type == "dirichlet":
        Lap.rows[N] = [N]
        Lap.data[N] = [1.0]
        rhs[N] = value
    elif bc_type == "neumann":
        Lap.rows[N] = [N - 1, N]
        Lap.data[N] = [-1.0 / h, 1.0 / h]
        rhs[N] = -value
    elif bc_type == "robin":
        Bi, c_inf = value
        Lap.rows[N] = [N - 1, N]
        Lap.data[N] = [-1.0 / h, 1.0 / h + Bi]
        rhs[N] = Bi * c_inf
    else:
        raise ValueError("bc_type must be 'dirichlet', 'neumann' or 'robin'")
    return Lap.tocsr(), rhs


def normalize_bc_specs(coeffs: dict, bc_type: str, bc_specs: dict | None) -> dict:
    """Per-substrate {sub: (bc_type, value)}; value is a scalar for
    dirichlet/neumann, a (Bi, c_inf) pair for robin. Falls back to one
    bc_type for all four substrates if bc_specs isn't given."""
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
    c_inf = coeffs["c_inf_hat"]
    return {sub: (bc_type, c_inf[sub]) for sub in SUBSTRATES}


def _default_initial_guess(bc_specs: dict, Npts: int) -> dict:
    # neumann/robin never start at 0.0 -- Monod(0;K)=0 zeroes the reaction
    # Jacobian at the first iterate for any co-limiting substrate
    out = {}
    for sub, (bt, val) in bc_specs.items():
        if bt == "dirichlet":
            start = val
        elif bt == "robin":
            c_inf = val[1]
            start = c_inf if c_inf > 0.0 else 1.0
        else:
            start = 1.0
        out[sub] = np.full(Npts, start)
    return out


def reaction_and_jacobian(C: dict, U: dict, coeffs: dict):
    """Returns R (dict substrate -> array) and dR (dict (row_sub, col_sub) -> array)."""
    Lambda, Khat, beta, prod = coeffs["Lambda"], coeffs["Khat"], coeffs["beta"], coeffs["production"]
    LambdaProd = coeffs["LambdaProd"]

    uptake = {}
    duptake = {}
    for sp_name in SPECIES:
        p_sub, s_sub = PRIMARY[sp_name], SECONDARY[sp_name]
        Mp = monod(C[p_sub], Khat[sp_name][p_sub])
        Ms = monod(C[s_sub], Khat[sp_name][s_sub])
        uptake[sp_name] = Mp * Ms
        duptake[sp_name] = {
            p_sub: dmonod(C[p_sub], Khat[sp_name][p_sub]) * Ms,
            s_sub: Mp * dmonod(C[s_sub], Khat[sp_name][s_sub]),
        }

    R = {sub: np.zeros_like(next(iter(C.values()))) for sub in SUBSTRATES}
    dR = {}

    def add(row, col, arr):
        if (row, col) not in dR:
            dR[(row, col)] = np.zeros_like(arr)
        dR[(row, col)] += arr

    for sp_name in SPECIES:
        p_sub, s_sub = PRIMARY[sp_name], SECONDARY[sp_name]
        u_i = U[sp_name]
        lam_p = Lambda[sp_name][p_sub]
        lam_s = Lambda[sp_name][s_sub]
        prod_sub, beta_key = prod[sp_name]
        b = beta[beta_key]

        R[p_sub] += -lam_p * u_i * uptake[sp_name]
        add(p_sub, p_sub, -lam_p * u_i * duptake[sp_name][p_sub])
        add(p_sub, s_sub, -lam_p * u_i * duptake[sp_name][s_sub])

        R[s_sub] += -lam_s * u_i * uptake[sp_name]
        add(s_sub, p_sub, -lam_s * u_i * duptake[sp_name][p_sub])
        add(s_sub, s_sub, -lam_s * u_i * duptake[sp_name][s_sub])

        lam_prod = LambdaProd[sp_name]
        R[prod_sub] += b * lam_prod * u_i * uptake[sp_name]
        add(prod_sub, p_sub, b * lam_prod * u_i * duptake[sp_name][p_sub])
        add(prod_sub, s_sub, b * lam_prod * u_i * duptake[sp_name][s_sub])

    return R, dR


def _assemble_global(Lap_bc: dict, R: dict, dR: dict, C: dict):
    n = len(SUBSTRATES)
    Npts = len(next(iter(C.values())))
    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])

    blocks = [[None] * n for _ in range(n)]
    idx = {sub: k for k, sub in enumerate(SUBSTRATES)}
    for i_sub in SUBSTRATES:
        i = idx[i_sub]
        blocks[i][i] = Lap_bc[i_sub].tolil()
        for j_sub in SUBSTRATES:
            j = idx[j_sub]
            key = (i_sub, j_sub)
            if key in dR:
                d = dR[key].copy()
                d[-1] = 0.0   # row N is the BC row, no reaction term there
                block = sp.diags(d, format="lil")
                if blocks[i][j] is None:
                    blocks[i][j] = block
                else:
                    blocks[i][j] = blocks[i][j] + block
    for i in range(n):
        for j in range(n):
            if blocks[i][j] is None:
                blocks[i][j] = sp.csr_matrix((Npts, Npts))
    J = sp.bmat(blocks, format="csr")
    return F, J


def _residual(C, U, Lap_bc, coeffs, bc_specs):
    Npts = len(next(iter(C.values())))
    R, dR = reaction_and_jacobian(C, U, coeffs)
    for sub in SUBSTRATES:
        R[sub][-1] = 0.0
    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    for k, sub in enumerate(SUBSTRATES):
        bt, val = bc_specs[sub]
        F[(k + 1) * Npts - 1] = (Lap_bc[sub] @ C[sub])[-1] - bc_row_target(bt, val)
    return F, R, dR


def solve_newton(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                  bc_specs: dict | None = None, C_init: dict | None = None,
                  tol: float = 1e-10, maxiter: int = 50, verbose: bool = False,
                  damped: bool = True, max_backtracks: int = 25,
                  c_max_factor: float = 5.0, relaxation_fallback: bool = True):
    """Damped Newton with a physical-plausibility bound (c_max_factor*max|bc
    value|) during backtracking, and an automatic fallback to solve_relaxation
    if the Jacobian is genuinely degenerate. Returns (C, history, method)."""
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    bc_specs = normalize_bc_specs(coeffs, bc_type, bc_specs)
    c_max = c_max_factor * max(bc_concentration_scale(bt, val)
                                for bt, val in bc_specs.values())

    Lap_bc = {}
    for sub in SUBSTRATES:
        bt, val = bc_specs[sub]
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bt, val)
        Lap_bc[sub] = Lb

    if C_init is not None:
        C = {sub: np.full(Npts, C_init[sub]) if np.isscalar(C_init[sub])
             else np.asarray(C_init[sub], dtype=float).copy() for sub in SUBSTRATES}
    else:
        C = _default_initial_guess(bc_specs, Npts)
    history = []
    F, R, dR = _residual(C, U, Lap_bc, coeffs, bc_specs)
    res_norm = np.linalg.norm(F, ord=np.inf)
    history.append(res_norm)

    def _fall_back_to_relaxation(reason: str):
        warnings.warn(f"solve_newton: {reason} -- falling back to solve_relaxation (PTC).",
                       RuntimeWarning)
        from .relaxation import solve_relaxation
        C_relax, hist_relax = solve_relaxation(coeffs, U, grid, bc_specs=bc_specs, steady_tol=tol)
        return C_relax, history + list(hist_relax), "newton_inner_relax_fallback"

    for it in range(maxiter):
        if res_norm < tol:
            break
        _, J = _assemble_global(Lap_bc, R, dR, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            dX = spla.spsolve(J, -F)

        if not np.all(np.isfinite(dX)) and relaxation_fallback:
            return _fall_back_to_relaxation(
                f"non-finite update at iter {it} (|F|_inf={res_norm:.3e})")

        step = 1.0
        if not damped:
            for k, sub in enumerate(SUBSTRATES):
                C[sub] = C[sub] + dX[k * Npts:(k + 1) * Npts]
            F, R, dR = _residual(C, U, Lap_bc, coeffs, bc_specs)
            res_norm = np.linalg.norm(F, ord=np.inf)
        else:
            accepted = False
            for _ in range(max_backtracks):
                C_trial = {sub: np.clip(C[sub] + step * dX[k * Npts:(k + 1) * Npts], 0.0, c_max)
                           for k, sub in enumerate(SUBSTRATES)}
                plausible = all(np.all(C[sub] + step * dX[k * Npts:(k + 1) * Npts] <= c_max)
                                 for k, sub in enumerate(SUBSTRATES))
                F_trial, R_trial, dR_trial = _residual(C_trial, U, Lap_bc, coeffs, bc_specs)
                res_trial = np.linalg.norm(F_trial, ord=np.inf)
                if np.isfinite(res_trial) and res_trial < res_norm and plausible:
                    C, F, R, dR, res_norm = C_trial, F_trial, R_trial, dR_trial, res_trial
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                if relaxation_fallback:
                    return _fall_back_to_relaxation(
                        f"backtracking exhausted at iter {it} (|F|_inf={res_norm:.3e})")
                warnings.warn(
                    f"Newton backtracking exhausted without reducing the residual "
                    f"(stuck at |F|_inf={res_norm:.3e}); accepting smallest step anyway.",
                    RuntimeWarning,
                )
                C, F, R, dR, res_norm = C_trial, F_trial, R_trial, dR_trial, res_trial
        history.append(res_norm)
        if verbose:
            print(f"Newton it={it} |F|_inf={res_norm:.3e} step={step:.3g}")
    return C, history, "newton"


def solve_picard(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                  bc_specs: dict | None = None,
                  tol: float = 1e-8, maxiter: int = 2000, relax: float = 1.0,
                  verbose: bool = False):
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    bc_specs = normalize_bc_specs(coeffs, bc_type, bc_specs)

    Lap_bc = {}
    for sub in SUBSTRATES:
        bt, val = bc_specs[sub]
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bt, val)
        Lap_bc[sub] = Lb

    C = _default_initial_guess(bc_specs, Npts)
    history = []
    for it in range(maxiter):
        R, _ = reaction_and_jacobian(C, U, coeffs)
        C_new = {}
        max_change = 0.0
        for sub in SUBSTRATES:
            bt, val = bc_specs[sub]
            rhs = -R[sub]
            rhs[0] = 0.0
            rhs[-1] = bc_row_target(bt, val)
            c_sol = spla.spsolve(Lap_bc[sub], rhs)
            C_new[sub] = (1 - relax) * C[sub] + relax * c_sol
            max_change = max(max_change, np.max(np.abs(C_new[sub] - C[sub])))
        C = C_new
        history.append(max_change)
        if verbose and it % 50 == 0:
            print(f"Picard it={it} max|dC|={max_change:.3e}")
        if max_change < tol:
            break
    return C, history
