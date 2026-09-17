"""Case C (commensalism, paper eq. 2.4) on the COUPLED solver: eps kept,
substrates and bacteria advance together with one dt. Same custom chain
chemistry as the slow-fast commensalism folders (single Monod per species,
u1 -> c2 and u2 -> c3 byproduct production), same rough IC, same regime
convention (regime on c1 only; c2, c3 sealed). The substrate Newton solve
is now one backward-Euler step from the previous field with the eps/dt mass
term, and there is no plausibility bound: the mass term makes every step
solvable.

eps and Lambda move together (both scale as 1/D): Lambda = eps / Y. With
the Case C yield Y = 10 and eps = 25 (the crossover value found in
coupled_competition_dirichlet), Lambda = 2.5.

Usage: python run_coupled_commensalism.py"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, portable
from nitrifiers.two_dimensional.grid2d import Grid2D, build_laplacian_2d, apply_bc_2d, boundary_flux_coefficients
from nitrifiers.two_dimensional.parabolic2d import build_advection_matrix_2d, build_advection_rho_jacobian_2d

REGIME = "neumann"  # hardcoded: this folder replicates the neumann regime only

OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

SPECIES = ["u1", "u2", "u3"]
SUBSTRATES = ["c1", "c2", "c3"]
CHAIN = {"u1": "c1", "u2": "c2", "u3": "c3"}         # species i consumes c_i
PRODUCES = {"u1": "c2", "u2": "c3"}                   # byproduct chain, u3 produces nothing

# Table 1, Case (C): d=1e-6, a=1e-5, r=1, K=1, b=0.1, D=1e-5 (paper), Y=10,
# sigma12=sigma23=0.5, c1(x,0)=5, c2=c3=0 initially. D substituted to 0.2
# for the same QSSA-validity reason used throughout this project (see README).
D_I, A_OVER_D_RATIO, R_MAIN, K_MAIN, B_MAIN, Y_MAIN = 1e-6, 10.0, 1.0, 1.0, 0.1, 10.0
EPS = 25.0
SIGMA = 0.5
C1_INIT = 5.0
GRID_N = 99
DT, T_FINAL = 0.1, 50.0
N_STEPS = int(round(T_FINAL / DT))
SNAP_EVERY = int(round(5.0 / DT))

LAM = EPS / Y_MAIN   # Lambda = r L^2 u_ref / (D c_ref Y) = eps / Y in these units


def initial_conditions_biomass(Nx, Ny, Lx, Ly, random_seed):
    np.random.seed(random_seed)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    main_radius = 0.2
    center = (0.5, 0.5)
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    support = distance_from_center <= main_radius
    total_density = 1.0
    populations = np.zeros((3, Nx, Ny))
    num_circles = 300
    small_radius = 0.05
    for _ in range(num_circles):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, main_radius - small_radius)
        cx = center[0] + r * np.cos(theta)
        cy = center[1] + r * np.sin(theta)
        small_circle = (X - cx) ** 2 + (Y - cy) ** 2 <= small_radius ** 2
        species_index = np.random.choice(3)
        populations[species_index] += small_circle.astype(float)
    for i in range(3):
        populations[i][~support] = 0
    density_sum = np.sum(populations, axis=0)
    density_sum[density_sum == 0] = 1
    populations = populations / density_sum * total_density
    return populations


def monod(c, K):
    return c / (K + c)


def dmonod(c, K):
    return K / (K + c) ** 2


def make_specs(regime):
    """Regime applies only to c1 (the externally-supplied substrate); c2, c3
    stay zero-flux Neumann always, matching the paper's byproduct-only
    structure for those two."""
    if regime == "dirichlet":
        c1_bc = ("dirichlet", C1_INIT)
    elif regime == "neumann":
        c1_bc = ("neumann", -(1e-5 * EPS))   # paper's 1e-5 divided by D = 1/eps
    else:
        c1_bc = ("robin", (5.0, C1_INIT))
    return {"c1": c1_bc, "c2": ("neumann", 0.0), "c3": ("neumann", 0.0)}


class SubstrateOps:
    def __init__(self, grid, specs):
        self.grid, self.specs = grid, specs
        self.bnodes = grid.boundary_flat
        self.flux_coef = boundary_flux_coefficients(grid)
        Lap0 = build_laplacian_2d(grid)
        self.Lap_bc = {s: apply_bc_2d(Lap0, grid, specs[s][0], boundary_nodes=self.bnodes) for s in SUBSTRATES}
        self.mass_mask = {}
        for s in SUBSTRATES:
            m = np.ones(grid.Npts)
            if specs[s][0] == "dirichlet":
                m[self.bnodes] = 0.0
            self.mass_mask[s] = m


def step_substrate(U, C_old, ops, eps, dt, tol=1e-8, maxiter=50):
    """One backward-Euler step of eps*dc/dt = Lap c + R(u, c), Newton from
    C_old. No plausibility bound: the eps/dt diagonal makes the step always
    solvable. Returns (C, n_iter, res, method, term_mags)."""
    grid, specs, bnodes, flux_coef, Lap_bc = ops.grid, ops.specs, ops.bnodes, ops.flux_coef, ops.Lap_bc
    Npts = grid.Npts
    a = eps / dt
    mass_vec = np.concatenate([a * ops.mass_mask[s] for s in SUBSTRATES])
    mass_diag = sp.diags(mass_vec, format="csr")
    C_old_vec = np.concatenate([C_old[s] for s in SUBSTRATES])

    def reaction(C):
        R = {s: np.zeros(Npts) for s in SUBSTRATES}
        dR = {(s, s): np.zeros(Npts) for s in SUBSTRATES}
        for spn, sub in CHAIN.items():
            u_i = U[spn]
            M = monod(C[sub], K_MAIN)
            dM = dmonod(C[sub], K_MAIN)
            R[sub] += -LAM * u_i * M
            dR[(sub, sub)] += -LAM * u_i * dM
            if spn in PRODUCES:
                prod_sub = PRODUCES[spn]
                R[prod_sub] += SIGMA * LAM * u_i * M
                dR[(prod_sub, sub)] = dR.get((prod_sub, sub), np.zeros(Npts)) + SIGMA * LAM * u_i * dM
        return R, dR

    def residual(C):
        R, dR = reaction(C)
        Rb = {s: R[s].copy() for s in SUBSTRATES}
        for s in SUBSTRATES:
            if specs[s][0] == "dirichlet":
                Rb[s][bnodes] = 0.0
        F = np.concatenate([Lap_bc[s] @ C[s] + Rb[s] for s in SUBSTRATES])
        for k, s in enumerate(SUBSTRATES):
            bt, val = specs[s]
            base = k * Npts
            if bt == "dirichlet":
                F[base + bnodes] = C[s][bnodes] - val
            elif bt == "robin":
                Bi, c_inf = val
                F[base + bnodes] += flux_coef[bnodes] * Bi * (C[s][bnodes] - c_inf)
            elif val != 0.0:
                F[base + bnodes] += flux_coef[bnodes] * val
        C_vec = np.concatenate([C[s] for s in SUBSTRATES])
        F = F - mass_vec * (C_vec - C_old_vec)
        return F, R, dR

    def assemble(dR):
        n = len(SUBSTRATES)
        idx = {s: k for k, s in enumerate(SUBSTRATES)}
        blocks = [[None] * n for _ in range(n)]
        for s in SUBSTRATES:
            i = idx[s]
            blocks[i][i] = Lap_bc[s].tolil()
            bt, val = specs[s]
            if bt == "robin":
                Bi, _ = val
                diag = np.zeros(Npts)
                diag[bnodes] = flux_coef[bnodes] * Bi
                blocks[i][i] = blocks[i][i] + sp.diags(diag, format="lil")
        for (row_s, col_s), d in dR.items():
            i, j = idx[row_s], idx[col_s]
            dd = d.copy()
            if specs[row_s][0] == "dirichlet":
                dd[bnodes] = 0.0
            block = sp.diags(dd, format="lil")
            blocks[i][j] = block if blocks[i][j] is None else blocks[i][j] + block
        for i in range(n):
            for j in range(n):
                if blocks[i][j] is None:
                    blocks[i][j] = sp.csr_matrix((Npts, Npts))
        return sp.bmat(blocks, format="csr") - mass_diag

    C = {s: v.copy() for s, v in C_old.items()}
    F, R, dR = residual(C)
    res = float(np.linalg.norm(F, ord=np.inf))
    tol_eff = tol * max(1.0, a * max(1.0, float(np.max(np.abs(C_old_vec)))))
    n_iter = 0
    method = "newton"
    for it in range(maxiter):
        if res < tol_eff:
            break
        J = assemble(dR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            dX = spla.spsolve(J.tocsc(), -F)
        if not np.all(np.isfinite(dX)):
            method = "newton_stalled"
            break
        step, accepted = 1.0, False
        for _ in range(20):
            C_try = {s: np.maximum(C[s] + step * dX[k * Npts:(k + 1) * Npts], 0.0) for k, s in enumerate(SUBSTRATES)}
            F_try, R_try, dR_try = residual(C_try)
            res_try = float(np.linalg.norm(F_try, ord=np.inf))
            if np.isfinite(res_try) and res_try < res:
                C, F, R, dR, res = C_try, F_try, R_try, dR_try, res_try
                accepted = True
                break
            step *= 0.5
        n_iter += 1
        if not accepted:
            method = "newton_stalled"
            break

    mags = {}
    for s in SUBSTRATES:
        m = ops.mass_mask[s] > 0
        diff = Lap_bc[s] @ C[s]
        ct = a * (C[s] - C_old[s])
        mags[s] = {"diffusion": float(np.sum(grid.Vflat[m] * np.abs(diff[m]))),
                   "reaction": float(np.sum(grid.Vflat[m] * np.abs(R[s][m]))),
                   "transient": float(np.sum(grid.Vflat[m] * np.abs(ct[m])))}
    return C, n_iter, res, method, mags


def solve_bacteria_step(U, C, grid, dt):
    Npts = grid.Npts
    Lap0 = build_laplacian_2d(grid)
    growth = {spn: R_MAIN * monod(C[CHAIN[spn]], K_MAIN) for spn in SPECIES}
    Un = {spn: U[spn].copy() for spn in SPECIES}
    Uk = {spn: U[spn].copy() for spn in SPECIES}
    I = sp.identity(Npts, format="csr")

    for it in range(60):
        rho = Uk["u1"] + Uk["u2"] + Uk["u3"]
        Adv = build_advection_matrix_2d(grid, rho)
        F = {}
        for spn in SPECIES:
            F[spn] = ((Uk[spn] - Un[spn]) / dt - D_I * (Lap0 @ Uk[spn])
                      - (A_OVER_D_RATIO * D_I) * (Adv @ Uk[spn])
                      - (growth[spn] - B_MAIN * rho) * Uk[spn])
        res = float(np.linalg.norm(np.concatenate(list(F.values())), ord=np.inf))
        if res < 1e-9:
            break
        n = len(SPECIES)
        idx = {spn: k for k, spn in enumerate(SPECIES)}
        blocks = [[None] * n for _ in range(n)]
        for spn in SPECIES:
            i = idx[spn]
            diagJ = (1.0 / dt) * I - D_I * Lap0 - (A_OVER_D_RATIO * D_I) * Adv \
                    - sp_diags(growth[spn] - B_MAIN * rho)
            blocks[i][i] = diagJ
            rho_jac = build_advection_rho_jacobian_2d(grid, Uk[spn], rho)
            for spn2 in SPECIES:
                j = idx[spn2]
                extra = -(A_OVER_D_RATIO * D_I) * rho_jac - sp_diags(Uk[spn] * (-B_MAIN))
                if spn2 == spn:
                    blocks[i][j] = blocks[i][j] + extra
                else:
                    blocks[i][j] = extra if blocks[i][j] is None else blocks[i][j] + extra
        J = sp.bmat(blocks, format="csr")
        Fvec = np.concatenate([F[spn] for spn in SPECIES])
        dX = spla.spsolve(J, -Fvec)
        for k, spn in enumerate(SPECIES):
            Uk[spn] = np.clip(Uk[spn] + dX[k * Npts:(k + 1) * Npts], 0.0, None)
    return Uk


def sp_diags(v):
    return sp.diags(v, format="csr")


def main():
    import json
    grid = Grid2D(Nx=GRID_N, Ny=GRID_N, Lx=1.0, Ly=1.0)
    populations = initial_conditions_biomass(100, 100, 1.0, 1.0, random_seed=42)
    U = {spn: populations[k].ravel() for k, spn in enumerate(SPECIES)}
    # paper's c_1(x, 0) = 5 everywhere; c_2, c_3 are byproducts, start at 0
    C = {"c1": np.full(grid.Npts, C1_INIT), "c2": np.zeros(grid.Npts), "c3": np.zeros(grid.Npts)}
    specs = make_specs(REGIME)
    ops = SubstrateOps(grid, specs)
    print(f"regime={REGIME} specs={specs} eps={EPS} Lambda={LAM:.4f} dt={DT} steps={N_STEPS}")

    def save_snapshot(t, U, C):
        for spn in SPECIES:
            np.save(OUT / f"t{t}_u_{spn}.npy", U[spn])
        for s in SUBSTRATES:
            np.save(OUT / f"t{t}_c_{s}.npy", C[s])

    def boundary_mass(U):
        return float(sum(np.sum(grid.Vflat[grid.boundary_flat] * U[spn][grid.boundary_flat]) for spn in SPECIES))

    save_snapshot(0, U, C)
    term_mags = {}
    boundary_mass_history = [(0.0, boundary_mass(U))]
    stall_count = 0
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        C, iters, res, method, mags = step_substrate(U, C, ops, EPS, DT)
        stall_count += method == "newton_stalled"
        U = solve_bacteria_step(U, C, grid, DT)
        t = step * DT
        bmass = boundary_mass(U)
        boundary_mass_history.append((t, bmass))
        if step % SNAP_EVERY == 0:
            ti = int(round(t))
            save_snapshot(ti, U, C)
            term_mags[str(ti)] = mags
            total = float(sum(np.sum(grid.Vflat * U[spn]) for spn in SPECIES))
            print(f"  step {step:4d} t={t:5.1f}: substrate {method:14s} iters={iters} res={res:.1e} "
                  f"mass={total:.5f} boundary_mass={bmass:.6f} max_u={max(float(np.max(v)) for v in U.values()):.4f}")

    print(f"\ndone in {time.time()-t0:.0f}s, {stall_count} stalled substrate steps out of {N_STEPS}")
    np.save(OUT / "grid_shape.npy", np.array([GRID_N, GRID_N]))
    bmh = np.array(boundary_mass_history)
    np.save(OUT / "boundary_mass_history.npy", bmh)
    first = bmh[bmh[:, 1] > 1e-8]
    print(f"biomass first reaches the boundary at t = {first[0, 0] if len(first) else None}")
    (OUT / "term_magnitudes.json").write_text(json.dumps(term_mags, indent=2))
    (OUT / "run_info.json").write_text(json.dumps({"regime": REGIME, "eps": EPS, "lambda": LAM, "sigma": SIGMA,
                                                    "dt": DT, "T": T_FINAL, "grid_n": GRID_N,
                                                    "stalled": stall_count}, indent=2, default=str))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
