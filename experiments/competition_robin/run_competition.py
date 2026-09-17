"""Case A (3-species competition), rough IC, one regime per invocation.
Usage: python run_competition.py {dirichlet,neumann,robin}
Snapshots U (3 species) and C (4 substrates) every 5 slow steps, 0..50, so
the structure just before biomass reaches the domain boundary is visible."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, portable
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.nondim import SUBSTRATES
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d
from nitrifiers.two_dimensional.parabolic2d import solve_parabolic_2d

REGIME = "robin"  # hardcoded: this folder replicates the robin regime only

OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


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


D_I, A_OVER_D_RATIO, R_MAIN, K_MAIN, B_MAIN, Y_MAIN, D_SUBSTRATE = 1e-6, 10.0, 1.0, 1.0, 0.1, 0.2, 0.2
C_INF_DIRICHLET = 5.0
GRID_N = 99
DT_SLOW, N_SLOW_STEPS = 1.0, 50
SNAPSHOT_STEPS = set(range(0, N_SLOW_STEPS + 1, 5))


def make_coeffs():
    lam_primary = R_MAIN / (D_SUBSTRATE * Y_MAIN)
    lam_o2 = 1e-6 * lam_primary
    khat_o2 = 1e-8
    Dhat = {sp: D_I for sp in SPECIES}
    Ahat = {sp: A_OVER_D_RATIO * D_I for sp in SPECIES}
    rhat = {sp: 1.0 for sp in SPECIES}
    bhat = {sp: B_MAIN for sp in SPECIES}
    Lambda = {"AOB": {"NH4": lam_primary, "O2": lam_o2}, "NOB": {"NO2": lam_primary, "O2": lam_o2}, "CMX": {"NH4": lam_primary, "O2": lam_o2}}
    LambdaProd = {sp: lam_primary for sp in SPECIES}
    Khat = {"AOB": {"NH4": K_MAIN, "O2": khat_o2}, "NOB": {"NO2": K_MAIN, "O2": khat_o2}, "CMX": {"NH4": K_MAIN, "O2": khat_o2}}
    # 1:1 stoichiometric conversion: AOB's NH4 uptake becomes NO2, NOB's NO2
    # uptake becomes NO3, CMX's NH4 uptake (full pathway) becomes NO3 directly.
    # Previously 0.0 here -- an inherited placeholder that silently made NO3
    # a static, never-produced field in every regime (verified: std==0.0
    # exactly at every timestep). This was never intentional physics.
    beta = {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0}
    production = {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")}
    consumed = {"AOB": ("NH4", "O2"), "NOB": ("NO2", "O2"), "CMX": ("NH4", "O2")}
    c_inf_hat = {"NH4": C_INF_DIRICHLET, "NO2": C_INF_DIRICHLET, "NO3": 0.0, "O2": C_INF_DIRICHLET}
    return {"Lambda": Lambda, "LambdaProd": LambdaProd, "Khat": Khat, "rhat": rhat, "bhat": bhat,
            "Dhat": Dhat, "Ahat": Ahat, "c_inf_hat": c_inf_hat, "beta": beta,
            "consumed_substrates": consumed, "production": production}


def make_specs(regime):
    # NO3 is now genuinely produced (beta=1.0 below), so it needs an outlet:
    # sealed (zero-flux) leaves it with no steady state once anything feeds
    # it, confirmed by 50/50 stalled Newton solves. Dirichlet c_inf=0 gives
    # it the same kind of open boundary O2 already has -- physically the
    # fully-oxidised end product is free to diffuse out, not trapped.
    if regime == "dirichlet":
        return {"NH4": ("dirichlet", C_INF_DIRICHLET), "NO2": ("dirichlet", C_INF_DIRICHLET),
                "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", C_INF_DIRICHLET)}
    if regime == "neumann":
        flux_val = -(1e-5 / D_SUBSTRATE)
        return {"NH4": ("neumann", flux_val), "NO2": ("neumann", flux_val),
                "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", C_INF_DIRICHLET)}
    BI = 5.0
    return {"NH4": ("robin", (BI, C_INF_DIRICHLET)), "NO2": ("robin", (BI, C_INF_DIRICHLET)),
            "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", C_INF_DIRICHLET)}


def main():
    grid = Grid2D(Nx=GRID_N, Ny=GRID_N, Lx=1.0, Ly=1.0)
    populations = initial_conditions_biomass(100, 100, 1.0, 1.0, random_seed=42)
    U = {sp: populations[k].ravel() for k, sp in enumerate(SPECIES)}
    coeffs = make_coeffs()
    specs = make_specs(REGIME)
    print(f"regime={REGIME} specs={specs}")

    center_idx = (GRID_N // 2) * (GRID_N + 1) + (GRID_N // 2)
    center_history = [sum(float(U[sp][center_idx]) for sp in SPECIES)]
    boundary_mass_history = []

    def save_snapshot(step, U, C):
        for sp in SPECIES:
            np.save(OUT / f"t{step}_u_{sp}.npy", U[sp])
        for sub in SUBSTRATES:
            np.save(OUT / f"t{step}_c_{sub}.npy", C[sub])

    stall_count = 0
    t0 = time.time()
    C, hist, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150, c_max_factor=5.0)
    save_snapshot(0, U, C)
    boundary_mass = float(sum(np.sum(grid.Vflat[grid.boundary_flat] * U[sp][grid.boundary_flat]) for sp in SPECIES))
    boundary_mass_history.append((0, boundary_mass))

    for step in range(1, N_SLOW_STEPS + 1):
        C, hist, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150, c_max_factor=5.0)
        if method == "newton_stalled":
            stall_count += 1
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=DT_SLOW, n_steps=1)
        center_history.append(sum(float(U[sp][center_idx]) for sp in SPECIES))
        boundary_mass = float(sum(np.sum(grid.Vflat[grid.boundary_flat] * U[sp][grid.boundary_flat]) for sp in SPECIES))
        boundary_mass_history.append((step, boundary_mass))
        if step in SNAPSHOT_STEPS:
            save_snapshot(step, U, C)
        if step % 5 == 0:
            print(f"  step {step:3d}: center_rho={center_history[-1]:.5f} "
                  f"boundary_mass={boundary_mass:.6f} max_u={max(float(np.max(v)) for v in U.values()):.4f}")

    print(f"\ndone in {time.time()-t0:.1f}s, {stall_count} stalled steps out of {N_SLOW_STEPS}")
    np.save(OUT / "grid_shape.npy", np.array([GRID_N, GRID_N]))
    np.save(OUT / "center_history.npy", np.array(center_history))
    np.save(OUT / "boundary_mass_history.npy", np.array(boundary_mass_history))
    first_touch = next((s for s, m in boundary_mass_history if m > 1e-8), None)
    print(f"boundary mass first becomes nonzero at step: {first_touch}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
