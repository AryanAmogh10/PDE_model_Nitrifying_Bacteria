"""
Stage 3: elliptic (quasi-steady-state) solver for the 4 non-dimensional substrate
equations, given a fixed bacterial density profile (u_AOB, u_NOB, u_CMX).

Solves, on a 1D domain rhat in [0, 1] (dimensionless radius/slab coordinate,
x = L*rhat), the leading-order (eps -> 0) system derived in nondim.py:

    0 = Lap(chat_j) + Rhat_j(uhat, chat),   j in {NH4, NO2, NO3, O2}

with Lap the (possibly radial) Laplacian and Rhat_j the reaction term assembled
from the Lambda/Khat/beta coefficients returned by nondim.elliptic_coefficients.

Two solvers are provided (both should converge to the same solution for a
well-posed problem; comparing them is the point of Stage 4):
    - solve_newton: full Newton iteration on the coupled 4*(N+1)-dimensional
      nonlinear system, using the analytic Jacobian (sparse, block-coupled
      through the Monod terms). Fast (quadratic) convergence when it converges.
    - solve_picard: lagged fixed-point iteration, solving each substrate's
      linear diffusion equation with the reaction term evaluated at the
      previous iterate. Slower (linear) convergence but each step only
      requires 4 independent tridiagonal-like solves; more robust when Newton
      struggles (e.g. very large Lambda / stiff parameter regimes).

Geometry: geometry='slab' (Cartesian, exponent p=0) or geometry='radial' with
p=1 (cylindrical) or p=2 (spherical, the default -- appropriate for a granule).
Boundary conditions at rhat=1: 'dirichlet' (bulk/feed value chat_inf) or
'neumann' (prescribed flux, default 0 = no-flux). At rhat=0 a natural
symmetry (zero-flux) condition is always used.
"""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .nondim import SUBSTRATES  # ("NH4", "NO2", "NO3", "O2")

SPECIES = ("AOB", "NOB", "CMX")
PRIMARY = {"AOB": "NH4", "NOB": "NO2", "CMX": "NH4"}
SECONDARY = {"AOB": "O2", "NOB": "O2", "CMX": "O2"}

_EPS_FLOOR = 1e-12  # floor to keep Monod terms well-defined for c <~ 0 during iteration


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
    """Area of the control-volume face located at radius r_face.

    r^p, except at the true geometric singularity r=0 for p>0, where the face
    collapses to zero area. For the slab (p=0) every face has unit area
    including the one at r=0 (r^0 = 1 identically, no geometric collapse);
    zero flux there comes from the mirror-symmetry ghost node instead, which
    is what the row-0 stencil already encodes.
    """
    p = grid.p
    return 0.0 if (p > 0 and r_face <= 0.0) else r_face ** p


def cell_volumes(grid: Grid) -> np.ndarray:
    """EXACT control-volume measure of every node's cell,
    V_i = (r_e^{p+1} - r_w^{p+1}) / (p+1), with half-cells at both ends.

    THIS IS THE SINGLE SOURCE OF TRUTH for the volume normalisation of every
    conservative operator in the 1D codebase -- build_laplacian, and (since
    the fix recorded below) parabolic.build_advection_matrix,
    parabolic.build_advection_rho_jacobian and parabolic._total_mass.

    They must all share it. A conservative finite-volume operator normalised
    by one measure but integrated against another conserves nothing: the
    telescoping identity sum_i V_i*(Op @ u)_i = 0 holds only for the SAME V_i
    the operator divided by. This is not a theoretical concern -- it was the
    actual state of the code: build_laplacian was corrected to exact volumes,
    but the advection operator was left normalising by the interior shortcut
    V_i ~= r_i^p*h. That shortcut is wrong by O(h^2/r_i^2) on the first few
    interior rows AND by a factor of ~2 at the OUTER boundary node (where it
    uses a full cell instead of the half cell that actually exists), in every
    geometry including the slab. The result was that diffusion conserved one
    measure and advection another, so the coupled Stage 5 scheme conserved
    neither; the discrepancy was O(1) and did not vanish under refinement.
    """
    N, h, p = grid.N, grid.h, grid.p
    r = grid.r

    def vol(r_w, r_e):
        return (r_e - r_w) if p == 0 else (r_e ** (p + 1) - r_w ** (p + 1)) / (p + 1)

    V = np.empty(N + 1)
    V[0] = vol(0.0, 0.5 * h)                      # half-cell at the centre
    for i in range(1, N):
        V[i] = vol(r[i] - 0.5 * h, r[i] + 0.5 * h)
    V[N] = vol(r[N] - 0.5 * h, r[N])              # half-cell at the outer edge
    return V


def build_laplacian(grid: Grid) -> sp.csr_matrix:
    """Conservative finite-volume discretisation of Lap(c) = (1/r^p) d/dr(r^p dc/dr),
    derived uniformly for every row (including r=0 and the outer boundary row) by
    integrating the flux balance over each node's true control volume and dividing
    by that control volume's EXACT size:

        Lap_i = [r_e^p (c_{i+1}-c_i)/h - r_w^p (c_i-c_{i-1})/h] / V_i,
        V_i = (r_e^{p+1} - r_w^{p+1}) / (p+1)   (p == 0 (slab): V_i = r_e - r_w = h)

    with r_w, r_e the (exact) west/east control-volume faces of node i:
    r_w=max(r_i-h/2, 0), r_e=r_i+h/2 for interior/centre rows, and r_e=r_N
    (no face beyond the domain) for the last row.

    An earlier version used the standard *interior* finite-volume shortcut
    V_i ~= r_i^p * h (exact only in the h->0 limit) for ALL rows i=1..N-1.
    That approximation's relative error is O(h^2/r_i^2) -- negligible for
    r_i >> h, but ~8% at i=1 (r_i=h) and still ~2% at i=2, since r_i is itself
    O(h) there. This is NOT a row-0 error (the r=0 row's own formula, derived
    by exact half-cell integration over V_0=(h/2)^(p+1)/(p+1), works out to
    the same 2*(p+1)/h^2 coefficient either way) -- it is an under-resolved
    volume normalisation on the first few rows *next to* the centre, and was
    found to be the dominant source of the ~1.8 (not ~2.0) empirical
    convergence order in the closed-form reaction benchmarks: those first few
    rows carry an O(1)-relative, O(h^2)-absolute local truncation error that
    pollutes the global norm at a rate that only asymptotes to 2nd order
    once h/r_i -> 0 for the worst-affected row, i.e. slower than the
    generic O(h^2) interior rate. Using the exact V_i everywhere removes it.
    """
    N, h = grid.N, grid.h
    r = grid.r
    V = cell_volumes(grid)          # shared exact measure -- see its docstring
    rows, cols, vals = [], [], []

    # r = 0 (i = 0): half-cell [0, h/2]. The inner face has zero area for p>0,
    # so only the outer face at h/2 can contribute.
    w_e0 = face_area(grid, 0.5 * h) / (h * V[0])
    rows += [0, 0]
    cols += [0, 1]
    vals += [-w_e0, w_e0]

    # interior points i = 1..N-1: full cell [r_i-h/2, r_i+h/2], exact V_i.
    for i in range(1, N):
        w_e = face_area(grid, r[i] + 0.5 * h) / (h * V[i])
        w_w = face_area(grid, r[i] - 0.5 * h) / (h * V[i])
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [w_w, -(w_e + w_w), w_e]

    # last row (i = N): a genuine conservative zero-flux (homogeneous Neumann)
    # half-cell [r_N-h/2, r_N], exact V_N (mirrors the r=0 row's treatment).
    # apply_bc below still overwrites this row when a non-zero-flux
    # (Dirichlet/Neumann) outer BC is needed -- this is only the correct
    # standalone zero-flux form.
    w_w = face_area(grid, r[N] - 0.5 * h) / (h * V[N])
    rows += [N, N]
    cols += [N - 1, N]
    vals += [w_w, -w_w]

    return sp.csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))


def apply_bc(Lap: sp.csr_matrix, rhs: np.ndarray, grid: Grid, bc_type: str, value: float):
    """Overwrite the last row of (Lap, rhs) with the chosen outer BC at rhat=1.
    'dirichlet': c[N] = value.  'neumann': -dc/dr|_1 = value (value=0 -> no-flux)."""
    N, h = grid.N, grid.h
    Lap = Lap.tolil()
    if bc_type == "dirichlet":
        Lap.rows[N] = [N]
        Lap.data[N] = [1.0]
        rhs[N] = value
    elif bc_type == "neumann":
        # first-order one-sided: (c[N]-c[N-1])/h = -value  =>  c[N]-c[N-1]+h*value=0
        Lap.rows[N] = [N - 1, N]
        Lap.data[N] = [-1.0 / h, 1.0 / h]
        rhs[N] = -value
    else:
        raise ValueError("bc_type must be 'dirichlet' or 'neumann'")
    return Lap.tocsr(), rhs


def reaction_and_jacobian(C: dict, U: dict, coeffs: dict):
    """C, U: dict of numpy arrays (same length), keyed by substrate/species name.
    Returns R (dict substrate -> array) and dR (dict (row_sub, col_sub) -> array,
    only nonzero entries present)."""
    Lambda, Khat, beta, prod = coeffs["Lambda"], coeffs["Khat"], coeffs["beta"], coeffs["production"]
    LambdaProd = coeffs["LambdaProd"]

    # Uptake_i := M(c_p;K_ip) * M(c_O2;K_iO2) -- NO rhat_i factor here: Lambda[i][j]
    # (and LambdaProd[i]) already contain the full, un-normalised r_i. An earlier
    # version multiplied by rhat_i here too, double-counting the growth rate --
    # invisible on the toy/rebeca presets (nearly-equal r_i across species) but a
    # real bug on eloi, where r_i varies ~13x across species (caught in review).
    uptake = {}
    duptake = {}  # duptake[species][substrate] = d(uptake)/d(c_substrate)
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
    dR = {}  # (row, col) -> array

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

        # consumption of primary substrate
        R[p_sub] += -lam_p * u_i * uptake[sp_name]
        add(p_sub, p_sub, -lam_p * u_i * duptake[sp_name][p_sub])
        add(p_sub, s_sub, -lam_p * u_i * duptake[sp_name][s_sub])

        # consumption of secondary substrate (O2)
        R[s_sub] += -lam_s * u_i * uptake[sp_name]
        add(s_sub, p_sub, -lam_s * u_i * duptake[sp_name][p_sub])
        add(s_sub, s_sub, -lam_s * u_i * duptake[sp_name][s_sub])

        # production of downstream substrate (1:1 molar mass balance); uses
        # LambdaProd (built with the PRODUCED substrate's diffusivity), not
        # lam_p (which carries the SOURCE substrate's diffusivity) -- an
        # earlier version reused lam_p here, a bug invisible when all
        # substrates share one D (toy/rebeca) but real when D differs across
        # substrates (eloi), caught in review.
        lam_prod = LambdaProd[sp_name]
        R[prod_sub] += b * lam_prod * u_i * uptake[sp_name]
        add(prod_sub, p_sub, b * lam_prod * u_i * duptake[sp_name][p_sub])
        add(prod_sub, s_sub, b * lam_prod * u_i * duptake[sp_name][s_sub])

    return R, dR


def _assemble_global(Lap_bc: dict, R: dict, dR: dict, C: dict):
    """Build the global residual vector and sparse Jacobian for the coupled
    4*(N+1) Newton system. Lap_bc[sub] is the (BC-applied) Laplacian for that
    substrate (identical operator for all substrates before BC, but BC values
    differ so we keep per-substrate copies)."""
    n = len(SUBSTRATES)
    Npts = len(next(iter(C.values())))
    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    # interior/BC row 0 and N of F must not double count BC residual; apply_bc already
    # folds the BC into Lap_bc and R is zeroed on BC rows by the caller.

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
                # only row N (the true Dirichlet/Neumann BC row, entirely
                # replaced by apply_bc) has no reaction contribution -- row 0
                # is the r=0 *interior* equation (using a symmetric stencil,
                # not a BC substitution) and must keep its reaction term.
                # Dropping it there was a bug: it left a fixed, non-vanishing
                # defect at a single grid point, which is exactly what
                # degrades a 2nd-order scheme to apparent 1st-order global
                # convergence (caught via a closed-form benchmark + direct
                # Jacobian diff against a hand-built reference matrix).
                d[-1] = 0.0
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


def _residual(C, U, Lap_bc, coeffs, bc_type):
    Npts = len(next(iter(C.values())))
    R, dR = reaction_and_jacobian(C, U, coeffs)
    c_inf = coeffs["c_inf_hat"]
    # only row N is a true BC row (fully replaced by apply_bc, no PDE meaning);
    # row 0 (r=0) is a real interior-type equation and keeps its reaction term
    # -- see the note in _assemble_global for why this matters.
    for sub in SUBSTRATES:
        R[sub][-1] = 0.0
    F = np.concatenate([Lap_bc[sub] @ C[sub] + R[sub] for sub in SUBSTRATES])
    for k, sub in enumerate(SUBSTRATES):
        # Dirichlet row N target is the prescribed value c_inf[sub]; Neumann row
        # N (built by apply_bc as (c[N]-c[N-1])/h = -value, value=c_inf[sub] --
        # see solve_newton) must be driven toward that SAME -value, not 0. An
        # earlier version hardcoded 0.0 here for the neumann branch, silently
        # forcing every "neumann" solve to converge to zero-flux regardless of
        # the actual (nonzero) value passed to apply_bc -- caught by comparing
        # against the closed-form nonzero-flux Neumann solutions (Task 2/Fix 1).
        F[(k + 1) * Npts - 1] = (Lap_bc[sub] @ C[sub])[-1] - (c_inf[sub] if bc_type == "dirichlet" else -c_inf[sub])
    return F, R, dR


def solve_newton(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                  tol: float = 1e-10, maxiter: int = 50, verbose: bool = False,
                  damped: bool = True, max_backtracks: int = 25,
                  c_max_factor: float = 5.0, relaxation_fallback: bool = True):
    """Newton iteration on the coupled 4*(N+1) system. With damped=True, uses a
    backtracking line search (halving the step until the residual norm decreases
    AND every concentration stays within a physical-plausibility band) to control
    two distinct failure modes seen under Neumann boundary conditions (see the
    Task 1(c) investigation):

    (1) A spurious-root guard: c_max_factor*max(c_inf) upper-bounds every
        concentration. Without it, an undamped/under-constrained backtrack step
        can be accepted purely because it happens to reduce the residual norm,
        even though it lands on a numerically-real but physically nonsensical
        root (observed: NH4 -> ~100, NO3 -> ~5400 under a sealed zero-flux
        Neumann + reaction system, an order of magnitude above anything the
        system could actually supply). Rejecting any trial step that pushes a
        concentration above this band -- and continuing to halve until one that
        doesn't is found -- makes that spurious region unreachable, since
        shrinking the step continuously pulls the trial back toward the current
        (already-plausible) iterate.
    (2) A genuine-degeneracy fallback: at a state where an entire substrate has
        been driven to exactly 0 everywhere (the physically correct answer for,
        e.g., O2 in a sealed system with active consumption and no resupply),
        the zero-flux Neumann Jacobian can become exactly singular -- there is
        no small, plausible step left that reduces the residual, because the
        physical answer has already been reached and the remaining ~1e-10
        residual is a discretization/floating-point remainder, not something a
        better Newton step can fix. Rather than force this and either stall (the
        old behaviour) or, worse, accept the physically-implausible best-effort
        trial, this is detected (non-finite Newton update, or backtracking
        exhausted with no residual-reducing *and* plausible step found) and the
        solve is handed off to relaxation.solve_relaxation, whose pseudo-time
        M/dt diagonal regularises exactly this kind of degenerate steady state.
        This is expected, reported behaviour for that regime, not a failure.

    Returns (C, history, method), where method is "newton" for an ordinary
    converged (or maxiter-exhausted) Newton solve, or
    "newton_inner_relax_fallback" when (2) above engaged. Callers that need to
    know whether the returned C actually came from Newton or from the inner PTC
    fallback (e.g. slowfast.py::solve_c_given_u's diagnostic method label)
    should use this rather than inferring it from the residual alone.
    """
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    c_inf = coeffs["c_inf_hat"]
    # max(abs(.)), not max(.): c_inf doubles as the Neumann FLUX value (see
    # apply_bc), which is signed (negative = influx) -- using the signed max
    # here would silently collapse c_max to 0 whenever the largest-magnitude
    # c_inf entry happens to be negative, clipping every trial concentration
    # to exactly 0 regardless of step size. Caught by the Fix 1 nonzero-flux
    # Neumann regression test, which uses a negative (influx) flux value.
    c_max = c_max_factor * max(abs(v) for v in c_inf.values())

    Lap_bc = {}
    for sub in SUBSTRATES:
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bc_type, c_inf[sub])
        Lap_bc[sub] = Lb

    C = {sub: np.full(Npts, c_inf[sub]) for sub in SUBSTRATES}
    history = []
    F, R, dR = _residual(C, U, Lap_bc, coeffs, bc_type)
    res_norm = np.linalg.norm(F, ord=np.inf)
    history.append(res_norm)

    def _fall_back_to_relaxation(reason: str):
        warnings.warn(
            f"solve_newton: {reason} -- falling back to solve_relaxation (PTC) "
            f"for this solve; this is expected behaviour for degenerate "
            f"Neumann/zero-substrate regimes, not a failure.",
            RuntimeWarning,
        )
        from .relaxation import solve_relaxation
        # steady_tol=tol (the CALLER's own requested tolerance), not
        # solve_relaxation's own default (1e-9): this fallback exists so a
        # degenerate Newton solve still meets whatever tolerance the caller
        # actually asked solve_newton for, rather than silently over- or
        # under-shooting it. slowfast.py::solve_c_given_u also has its own,
        # separate outer Newton-then-relaxation fallback; that one is kept as a
        # defensive backstop for cases where solve_newton returns above
        # elliptic_tol WITHOUT hitting this inner fallback at all (e.g. maxiter
        # exhausted mid-convergence, never actually stalling) -- see its
        # docstring. With steady_tol matched here, the two layers should not
        # both fire for the same underlying degeneracy with different targets.
        C_relax, hist_relax = solve_relaxation(coeffs, U, grid, bc_type=bc_type, steady_tol=tol)
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
                f"Newton linear solve returned a non-finite update at iter {it} "
                f"(|F|_inf={res_norm:.3e}); Jacobian is degenerate"
            )

        step = 1.0
        if not damped:
            for k, sub in enumerate(SUBSTRATES):
                C[sub] = C[sub] + dX[k * Npts:(k + 1) * Npts]
            F, R, dR = _residual(C, U, Lap_bc, coeffs, bc_type)
            res_norm = np.linalg.norm(F, ord=np.inf)
        else:
            accepted = False
            for _ in range(max_backtracks):
                C_trial = {sub: np.clip(C[sub] + step * dX[k * Npts:(k + 1) * Npts], 0.0, c_max)
                           for k, sub in enumerate(SUBSTRATES)}
                plausible = all(np.all(C[sub] + step * dX[k * Npts:(k + 1) * Npts] <= c_max)
                                 for k, sub in enumerate(SUBSTRATES))
                F_trial, R_trial, dR_trial = _residual(C_trial, U, Lap_bc, coeffs, bc_type)
                res_trial = np.linalg.norm(F_trial, ord=np.inf)
                if np.isfinite(res_trial) and res_trial < res_norm and plausible:
                    C, F, R, dR, res_norm = C_trial, F_trial, R_trial, dR_trial, res_trial
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                if relaxation_fallback:
                    return _fall_back_to_relaxation(
                        f"Newton backtracking exhausted {max_backtracks} halvings without "
                        f"a residual-reducing, physically-plausible step at iter {it} "
                        f"(stuck at |F|_inf={res_norm:.3e})"
                    )
                warnings.warn(
                    f"Newton backtracking exhausted {max_backtracks} halvings without "
                    f"reducing the residual (stuck at |F|_inf={res_norm:.3e}); "
                    f"accepting smallest step anyway. Iteration may have stalled -- "
                    f"consider solve_relaxation as a fallback.",
                    RuntimeWarning,
                )
                C, F, R, dR, res_norm = C_trial, F_trial, R_trial, dR_trial, res_trial
        history.append(res_norm)
        if verbose:
            print(f"Newton it={it} |F|_inf={res_norm:.3e} step={step:.3g}")
    return C, history, "newton"


def solve_picard(coeffs: dict, U: dict, grid: Grid, bc_type: str = "dirichlet",
                  tol: float = 1e-8, maxiter: int = 2000, relax: float = 1.0,
                  verbose: bool = False):
    Npts = grid.N + 1
    Lap0 = build_laplacian(grid)
    c_inf = coeffs["c_inf_hat"]

    Lap_bc = {}
    for sub in SUBSTRATES:
        rhs0 = np.zeros(Npts)
        Lb, _ = apply_bc(Lap0, rhs0, grid, bc_type, c_inf[sub])
        Lap_bc[sub] = Lb

    C = {sub: np.full(Npts, c_inf[sub]) for sub in SUBSTRATES}
    history = []
    for it in range(maxiter):
        R, _ = reaction_and_jacobian(C, U, coeffs)
        C_new = {}
        max_change = 0.0
        for sub in SUBSTRATES:
            rhs = -R[sub]
            rhs[0] = 0.0
            # see the matching note in _residual: neumann's row-N target must be
            # -c_inf[sub] (apply_bc's own convention), not 0.0.
            rhs[-1] = c_inf[sub] if bc_type == "dirichlet" else -c_inf[sub]
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
