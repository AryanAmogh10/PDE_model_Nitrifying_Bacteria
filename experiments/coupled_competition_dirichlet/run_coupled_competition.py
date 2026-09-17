"""Fully time-dependent (eps kept) 3-species competition, rough IC, one
boundary regime per folder (REGIME below). Two modes:

    python run_coupled_competition.py scan   # eps scan on a coarse grid
    python run_coupled_competition.py full   # one long run at EPS_CHOSEN

Nondimensional substrate equation: eps * dc/dt = Lap c - Lambda u M(c).
eps and Lambda are not independent: both scale as 1/D, so Lambda = eps / Y
(= 5 eps with Y = 0.2). The scan therefore moves the physically consistent
pair (eps, 5 eps) and, at each eps, measures the volume-integrated
transient term eps*|c_t| against |Lap c| over the run (|R|/|Lap c| is ~1 at
any quasi-steady state and says nothing): transient/diffusion << 1 means
the substrate is slaved to u (QSSA valid), ~1 means the substrate's own
dynamics matter -- the regime where reaction/diffusion and bacterial
growth act on comparable timescales. It also checks
dt-convergence (dt vs dt/2) so the chosen eps is one the time stepping
actually resolves. The slow-fast (QSSA) solver on the same grid and dt is
the eps -> 0 reference."""
import sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, portable
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.nondim import SUBSTRATES
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d
from nitrifiers.two_dimensional.parabolic2d import solve_parabolic_2d
from nitrifiers.coupled.coupled2d import run_coupled_2d

MODE = sys.argv[1] if len(sys.argv) > 1 else "scan"
REGIME = "dirichlet"  # hardcoded: this folder replicates the dirichlet regime only
HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def initial_conditions_biomass(Nx, Ny, Lx, Ly, random_seed):
    np.random.seed(random_seed)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    main_radius = 0.2
    center = (0.5, 0.5)
    support = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) <= main_radius
    populations = np.zeros((3, Nx, Ny))
    for _ in range(300):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, main_radius - 0.05)
        cx, cy = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
        small_circle = (X - cx) ** 2 + (Y - cy) ** 2 <= 0.05 ** 2
        populations[np.random.choice(3)] += small_circle.astype(float)
    for i in range(3):
        populations[i][~support] = 0
    density_sum = np.sum(populations, axis=0)
    density_sum[density_sum == 0] = 1
    return populations / density_sum * 1.0


D_I, A_OVER_D_RATIO, R_MAIN, K_MAIN, B_MAIN, Y_MAIN = 1e-6, 10.0, 1.0, 1.0, 0.1, 0.2
C_INF = 5.0
EPS_CHOSEN = 25.0         # from the scan: eps*l^2 ~ 1 crossover, see README
T_SCAN, T_FULL = 10.0, 50.0
GRID_SCAN, GRID_FULL = 49, 99


def make_coeffs(lam):
    lam_o2 = 1e-6 * lam
    return {
        "Lambda": {"AOB": {"NH4": lam, "O2": lam_o2}, "NOB": {"NO2": lam, "O2": lam_o2}, "CMX": {"NH4": lam, "O2": lam_o2}},
        "LambdaProd": {sp: lam for sp in SPECIES},
        "Khat": {"AOB": {"NH4": K_MAIN, "O2": 1e-8}, "NOB": {"NO2": K_MAIN, "O2": 1e-8}, "CMX": {"NH4": K_MAIN, "O2": 1e-8}},
        "rhat": {sp: 1.0 for sp in SPECIES}, "bhat": {sp: B_MAIN for sp in SPECIES},
        "Dhat": {sp: D_I for sp in SPECIES}, "Ahat": {sp: A_OVER_D_RATIO * D_I for sp in SPECIES},
        "c_inf_hat": {"NH4": C_INF, "NO2": C_INF, "NO3": 0.0, "O2": C_INF},
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "production": {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")},
        "consumed_substrates": {"AOB": ("NH4", "O2"), "NOB": ("NO2", "O2"), "CMX": ("NH4", "O2")},
    }


def make_specs(regime, eps):
    """Regime applies to NH4/NO2 (the fed substrates); O2 stays Dirichlet
    (non-limiting co-substrate), NO3 stays Dirichlet 0 (outlet for the
    produced end product). The Neumann flux is the paper's literal 1e-5
    divided by D = 1/eps, the same D-correction the slow-fast folders use
    (there D = 0.2, i.e. eps = 5)."""
    if regime == "dirichlet":
        fed = ("dirichlet", C_INF)
    elif regime == "neumann":
        fed = ("neumann", -(1e-5 * eps))
    else:
        fed = ("robin", (5.0, C_INF))
    return {"NH4": fed, "NO2": fed, "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", C_INF)}


SPECS = make_specs(REGIME, EPS_CHOSEN)
# paper's c(x, 0) = c_0 = 5 for the fed substrates. The substrate has memory
# in this solver, so this matters -- under Neumann in particular the initial
# reservoir feeds growth until depleted, whereas the quasi-steady solver can
# hold no reservoir at all (its Neumann c sits at ~1e-4 everywhere).
C0 = {"NH4": C_INF, "NO2": C_INF, "NO3": 0.0, "O2": C_INF}


def setup(grid_n):
    grid = Grid2D(Nx=grid_n, Ny=grid_n, Lx=1.0, Ly=1.0)
    pops = initial_conditions_biomass(grid_n + 1, grid_n + 1, 1.0, 1.0, random_seed=42)
    U0 = {sp: pops[k].ravel() for k, sp in enumerate(SPECIES)}
    return grid, U0


def center_index(grid_n):
    return (grid_n // 2) * (grid_n + 1) + (grid_n // 2)


def run_qssa_reference(grid, U0, coeffs, specs, dt, T):
    """Slow-fast solver (eps -> 0) on the same grid/dt for comparison."""
    U = {s: v.copy() for s, v in U0.items()}
    cidx = center_index(grid.Nx)
    hist = [sum(float(U[s][cidx]) for s in SPECIES)]
    stalls = 0
    for _ in range(int(round(T / dt))):
        C, _, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150)
        stalls += method == "newton_stalled"
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt, n_steps=1)
        hist.append(sum(float(U[s][cidx]) for s in SPECIES))
    return np.array(hist), stalls


def scan():
    grid, U0 = setup(GRID_SCAN)
    cidx = center_index(GRID_SCAN)
    summary = {}
    for eps in (1.0, 5.0, 20.0, 25.0, 50.0, 100.0):
        lam = eps / Y_MAIN
        coeffs = make_coeffs(lam)
        row = {"lambda": lam}
        for dt in (0.1, 0.05):
            t0 = time.time()
            specs = make_specs(REGIME, eps)
            U, C, history, _, stalled = run_coupled_2d(coeffs, grid, U0, specs, eps=eps, dt=dt,
                                                        n_steps=int(round(T_SCAN / dt)),
                                                        C0={s: np.full(grid.Npts, v) for s, v in C0.items()},
                                                        record_magnitudes_every=1)
            ratios = [np.mean([h.term_magnitudes[s]["transient"] /
                               max(h.term_magnitudes[s]["diffusion"], 1e-30) for s in ("NH4", "NO2")])
                      for h in history]
            row[f"dt={dt}"] = {
                "center_rho_T": sum(float(U[s][cidx]) for s in SPECIES),
                "max_u_T": max(float(np.max(v)) for v in U.values()),
                "transient_over_diffusion_mean": float(np.mean(ratios)),
                "transient_over_diffusion_range": [float(np.min(ratios)), float(np.max(ratios))],
                "substrate_newton_iters_mean": float(np.mean([h.substrate_iters for h in history])),
                "stalled": stalled, "wall_s": time.time() - t0,
            }
            print(f"eps={eps:6.1f} lam={lam:6.1f} dt={dt}: center_rho(T)={row[f'dt={dt}']['center_rho_T']:.4f} "
                  f"eps*c_t/Lap={row[f'dt={dt}']['transient_over_diffusion_mean']:.4f} "
                  f"iters={row[f'dt={dt}']['substrate_newton_iters_mean']:.2f} stalled={stalled} "
                  f"({row[f'dt={dt}']['wall_s']:.0f}s)")
        a, b = row["dt=0.1"]["center_rho_T"], row["dt=0.05"]["center_rho_T"]
        row["dt_convergence_rel_diff"] = abs(a - b) / max(abs(b), 1e-12)
        qssa_hist, qstalls = run_qssa_reference(grid, U0, coeffs, specs, dt=0.1, T=T_SCAN)
        row["qssa_center_rho_T"] = float(qssa_hist[-1])
        row["qssa_stalled"] = qstalls
        row["coupled_vs_qssa_rel_diff"] = abs(a - qssa_hist[-1]) / max(abs(qssa_hist[-1]), 1e-12)
        print(f"   dt-conv rel diff={row['dt_convergence_rel_diff']:.4f}   "
              f"QSSA center_rho(T)={qssa_hist[-1]:.4f} (stalls {qstalls})  "
              f"coupled-vs-QSSA rel diff={row['coupled_vs_qssa_rel_diff']:.4f}")
        summary[str(eps)] = row
    (OUT / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved", OUT / "scan_summary.json")


def full():
    grid, U0 = setup(GRID_FULL)
    lam = EPS_CHOSEN / Y_MAIN
    coeffs = make_coeffs(lam)
    dt = 0.1
    snap_every = int(round(5.0 / dt))
    cidx = center_index(GRID_FULL)
    center_hist, boundary_hist = [], []

    def on_snapshot(step, t, U, C):
        for s in SPECIES:
            np.save(OUT / f"t{int(round(t))}_u_{s}.npy", U[s])
        for s in SUBSTRATES:
            np.save(OUT / f"t{int(round(t))}_c_{s}.npy", C[s])

    t0 = time.time()
    U, C, history, _, stalled = run_coupled_2d(coeffs, grid, U0, SPECS, eps=EPS_CHOSEN, dt=dt,
                                                n_steps=int(round(T_FULL / dt)), snapshot_every=snap_every,
                                                C0={s: np.full(grid.Npts, v) for s, v in C0.items()},
                                                record_magnitudes_every=snap_every, on_snapshot=on_snapshot,
                                                verbose=True)
    print(f"done in {time.time()-t0:.0f}s, regime={REGIME}, eps={EPS_CHOSEN}, lambda={lam}, "
          f"stalled substrate steps={stalled}")
    np.save(OUT / "grid_shape.npy", np.array([GRID_FULL, GRID_FULL]))
    mags = {str(int(round(h.t))): h.term_magnitudes for h in history if h.term_magnitudes}
    (OUT / "term_magnitudes.json").write_text(json.dumps(mags, indent=2))
    (OUT / "run_info.json").write_text(json.dumps({"regime": REGIME, "specs": {k: list(v) if isinstance(v, tuple) else v
                                                    for k, v in SPECS.items()}, "eps": EPS_CHOSEN, "lambda": lam,
                                                    "dt": dt, "T": T_FULL, "grid_n": GRID_FULL, "stalled": stalled},
                                                   indent=2, default=str))


if __name__ == "__main__":
    scan() if MODE == "scan" else full()
