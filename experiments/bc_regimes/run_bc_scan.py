"""Boundary-condition regimes of the Case A competition problem (time-scale-separation solver).
Scans (i) the Neumann flux and (ii) the Robin exchange number Bi and outer concentration c_inf and records the
total biomass at t = 20 relative to the start, the number of failed substrate solves and the largest substrate
concentration. 50 x 50 nodes, dt = 1 (20 steps), same initial circles as the other Case A runs.
    python run_bc_scan.py            (about 30-60 minutes with 6 workers)"""
import sys, json, importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("runcomp", HERE.parent / "competition_robin" / "run_competition.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d
from nitrifiers.two_dimensional.parabolic2d import solve_parabolic_2d

N = 49
STEPS = 20
FLUX = [1e-5, 5e-5, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
BI = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
CINF = [0.5, 1.0, 2.0, 5.0, 10.0]


def run(job):
    kind, par = job
    grid = Grid2D(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    pop = rc.initial_conditions_biomass(N + 1, N + 1, 1.0, 1.0, random_seed=42)
    U = {sp: pop[k].ravel() for k, sp in enumerate(SPECIES)}
    coeffs = rc.make_coeffs()
    if kind == "neumann":
        g = -par
        specs = {"NH4": ("neumann", g), "NO2": ("neumann", g), "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", rc.C_INF_DIRICHLET)}
        cinf = rc.C_INF_DIRICHLET
    else:
        bi, cinf = par
        specs = {"NH4": ("robin", (bi, cinf)), "NO2": ("robin", (bi, cinf)), "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", cinf)}
        coeffs["c_inf_hat"] = {"NH4": cinf, "NO2": cinf, "NO3": 0.0, "O2": cinf}
    V = grid.Vflat
    m0 = float(sum(np.sum(V * U[s]) for s in SPECIES))
    stalled, cmax, touch = 0, 0.0, None
    for step in range(1, STEPS + 1):
        C, hist, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150, c_max_factor=5.0)
        stalled += method == "newton_stalled"
        cmax = max(cmax, float(max(np.max(C["NH4"]), np.max(C["NO2"]))))
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=1.0, n_steps=1)
        if touch is None and sum(np.sum(V[grid.boundary_flat] * U[s][grid.boundary_flat]) for s in SPECIES) > 1e-8:
            touch = step
    m1 = float(sum(np.sum(V * U[s]) for s in SPECIES))
    rho_max = float(max(np.max(U[s]) for s in SPECIES))
    return {"kind": kind, "par": par, "m_rel": m1 / m0, "stalled": int(stalled), "cmax": cmax, "touch": touch, "rho_max": rho_max}


if __name__ == "__main__":
    jobs = [("neumann", q) for q in FLUX] + [("robin", (b, c)) for b in BI for c in CINF]
    with Pool(6) as pool:
        res = []
        for r in pool.imap_unordered(run, jobs):
            print(r, flush=True); res.append(r)
    (HERE / "bc_scan.json").write_text(json.dumps(res))
