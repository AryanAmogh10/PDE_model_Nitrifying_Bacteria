"""Case C (commensalism, paper eq. 2.4), rough IC, one regime per invocation.
Self-contained: the repo's reaction_and_jacobian is hardcoded to the
co-limited AOB/NOB/CMX chemistry (each species needs 2 Monod terms), but
paper's commensalism chain needs 1 Monod term per species plus a single
cross-production term (u1 -> c2, u2 -> c3). Reuses only the substrate-
and species-agnostic 2D grid/advection machinery; reaction + Newton here
are custom, built directly from paper eq. (2.4) and Table 1 Case (C).
Usage: python run_commensalism.py {dirichlet,neumann,robin}"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, portable
from nitrifiers.two_dimensional.grid2d import Grid2D, build_laplacian_2d, apply_bc_2d, boundary_flux_coefficients
from nitrifiers.two_dimensional.parabolic2d import build_advection_matrix_2d, build_advection_rho_jacobian_2d

REGIME = "dirichlet"  # hardcoded: this folder replicates the dirichlet regime only

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
D_SUBSTRATE = 0.2
SIGMA = 0.5
C1_INIT = 5.0
GRID_N = 99
DT_SLOW, N_SLOW_STEPS = 1.0, 50
SNAPSHOT_STEPS = set(range(0, N_SLOW_STEPS + 1, 5))

LAM = R_MAIN / (D_SUBSTRATE * Y_MAIN)   # same for all 3 species (r,Y identical)


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
        c1_bc = ("neumann", -(1e-5 / D_SUBSTRATE))
    else:
        c1_bc = ("robin", (5.0, C1_INIT))
    return {"c1": c1_bc, "c2": ("neumann", 0.0), "c3": ("neumann", 0.0)}


def solve_substrate_newton(U, grid, specs, tol=1e-6, maxiter=150, c_max_factor=5.0):
    Npts = grid.Npts
    bnodes = grid.boundary_flat
    flux_coef = boundary_flux_coefficients(grid)
    Lap0 = build_laplacian_2d(grid)
    Lap_bc = {s: apply_bc_2d(Lap0, grid, specs[s][0], boundary_nodes=bnodes) for s in SUBSTRATES}

    scale = {"dirichlet": lambda v: abs(v), "robin": lambda v: abs(v[1]), "neumann": lambda v: C1_INIT}
    c_max = c_max_factor * max(scale[bt](val) for bt, val in specs.values())

    C = {}
    for s, (bt, val) in specs.items():
        if bt == "dirichlet":
            start = val
        elif bt == "robin":
            start = val[1] if val[1] > 0 else 1.0
        else:
            start = 1.0
        C[s] = np.full(Npts, start)

    def residual(C):
        R = {s: np.zeros(Npts) for s in SUBSTRATES}
        dR = {(s, s): np.zeros(Npts) for s in SUBSTRATES}
        for sp, sub in CHAIN.items():
            u_i = U[sp]
            M = monod(C[sub], K_MAIN)
            dM = dmonod(C[sub], K_MAIN)
            R[sub] += -LAM * u_i * M
            dR[(sub, sub)] += -LAM * u_i * dM
            if sp in PRODUCES:
                prod_sub = PRODUCES[sp]
                R[prod_sub] += SIGMA * LAM * u_i * M
                dR[(prod_sub, sub)] = dR.get((prod_sub, sub), np.zeros(Npts)) + SIGMA * LAM * u_i * dM
        for s in SUBSTRATES:
            bt, _ = specs[s]
            if bt == "dirichlet":
                R[s][bnodes] = 0.0
        F = np.concatenate([Lap_bc[s] @ C[s] + R[s] for s in SUBSTRATES])
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
        return F, R, dR

    def assemble(C, dR):
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
        return sp.bmat(blocks, format="csr")

    F, R, dR = residual(C)
    res = float(np.linalg.norm(F, ord=np.inf))
    for it in range(maxiter):
        if res < tol:
            break
        J = assemble(C, dR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            dX = spla.spsolve(J, -F)
        if not np.all(np.isfinite(dX)):
            return C, "newton_stalled"
        step = 1.0
        accepted = False
        for _ in range(30):
            raw = {s: C[s] + step * dX[k * Npts:(k + 1) * Npts] for k, s in enumerate(SUBSTRATES)}
            plausible = all(np.all(v <= c_max) for v in raw.values())
            C_try = {s: np.clip(v, 0.0, c_max) for s, v in raw.items()}
            F_try, R_try, dR_try = residual(C_try)
            res_try = float(np.linalg.norm(F_try, ord=np.inf))
            if np.isfinite(res_try) and res_try < res and plausible:
                C, F, R, dR, res = C_try, F_try, R_try, dR_try, res_try
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return C, "newton_stalled"
    return C, "newton"


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
    grid = Grid2D(Nx=GRID_N, Ny=GRID_N, Lx=1.0, Ly=1.0)
    populations = initial_conditions_biomass(100, 100, 1.0, 1.0, random_seed=42)
    U = {sp: populations[k].ravel() for k, sp in enumerate(SPECIES)}
    C = {"c1": np.full(grid.Npts, C1_INIT), "c2": np.zeros(grid.Npts), "c3": np.zeros(grid.Npts)}
    specs = make_specs(REGIME)
    print(f"regime={REGIME} specs={specs} Lambda={LAM:.4f}")

    center_idx = (GRID_N // 2) * (GRID_N + 1) + (GRID_N // 2)
    center_history = [sum(float(U[sp][center_idx]) for sp in SPECIES)]
    boundary_mass_history = []

    def save_snapshot(step, U, C):
        for sp in SPECIES:
            np.save(OUT / f"t{step}_u_{sp}.npy", U[sp])
        for s in SUBSTRATES:
            np.save(OUT / f"t{step}_c_{s}.npy", C[s])

    C, method = solve_substrate_newton(U, grid, specs)
    save_snapshot(0, U, C)
    bmass = float(sum(np.sum(grid.Vflat[grid.boundary_flat] * U[sp][grid.boundary_flat]) for sp in SPECIES))
    boundary_mass_history.append((0, bmass))

    stall_count = 0
    t0 = time.time()
    for step in range(1, N_SLOW_STEPS + 1):
        C, method = solve_substrate_newton(U, grid, specs)
        if method == "newton_stalled":
            stall_count += 1
        U = solve_bacteria_step(U, C, grid, DT_SLOW)
        center_history.append(sum(float(U[sp][center_idx]) for sp in SPECIES))
        bmass = float(sum(np.sum(grid.Vflat[grid.boundary_flat] * U[sp][grid.boundary_flat]) for sp in SPECIES))
        boundary_mass_history.append((step, bmass))
        if step in SNAPSHOT_STEPS:
            save_snapshot(step, U, C)
        if step % 5 == 0:
            print(f"  step {step:3d}: center_rho={center_history[-1]:.5f} "
                  f"boundary_mass={bmass:.6f} max_u={max(float(np.max(v)) for v in U.values()):.4f}")

    print(f"\ndone in {time.time()-t0:.1f}s, {stall_count} stalled substrate solves out of {N_SLOW_STEPS}")
    np.save(OUT / "grid_shape.npy", np.array([GRID_N, GRID_N]))
    np.save(OUT / "center_history.npy", np.array(center_history))
    np.save(OUT / "boundary_mass_history.npy", np.array(boundary_mass_history))
    bmh = np.array(boundary_mass_history)
    first_touch = bmh[bmh[:, 1] > 1e-8]
    print(f"boundary mass first becomes nonzero at step: {int(first_touch[0,0]) if len(first_touch) else None}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
