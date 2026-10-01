"""Step 1: AOB and CMX nitrification strategies in isolation.

    AOB system: AOB + NOB present, u_CMX(t=0) = 0
    CMX system: CMX present,       u_AOB(t=0) = u_NOB(t=0) = 0

QSSA slow-fast solver, Dirichlet on every substrate (NH4 = NH4_BC,
O2 = O2_BC, NO2 = NO3 = 0) and substrate initial values equal to those
boundary values.

Modelling choices (see README for the reasoning):
  * kinetics (r, b, K, Lambda, beta) from the 'rebeca' / 'eloi' presets;
    bacterial motility from the paper's Case A (d = 1e-6, a = 1e-5),
    because the presets' motility (~4e6 times larger) spreads the colony
    over the whole domain in one step;
  * survival cutoff: u_i set to 0 where u_i < U_CUT after every step,
    otherwise diffusion seeds ~1e-40 biomass everywhere and that tail
    grows exponentially in the substrate-rich region outside the colony.

Usage:
    python run_isolation.py <preset> <system> [N] [T] [dt] [NH4_BC] [O2_BC] [tag]
    preset in {rebeca, eloi}, system in {aob, cmx}"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, portable
from nitrifiers.nondim import elliptic_coefficients, SUBSTRATES
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.one_dimensional.elliptic import reaction_and_jacobian
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d
from nitrifiers.two_dimensional.parabolic2d import solve_parabolic_2d

D_MOT, A_MOT = 1e-6, 1e-5
U_CUT = 1e-6
SNAP_EVERY_T = 5.0


def circle_layout(n_points, random_seed=42):
    """Replays the user's initial_conditions_biomass RNG sequence exactly and
    returns (X, Y, list of (cx, cy, original_species_index)), so every
    system and step 3 share one identical circle geometry."""
    np.random.seed(random_seed)
    x = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, x, indexing="ij")
    circles = []
    for _ in range(300):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.2 - 0.05)
        circles.append((0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta), np.random.choice(3)))
    return X, Y, circles


def initial_biomass(n_points, system):
    """Same 300 circles and support for every system; total density 1 on the
    support, so initial total biomass is identical across systems.
    aob: each circle AOB or NOB (50/50, separate RNG); cmx: every circle CMX;
    all: the original 3-way assignment (step 3)."""
    X, Y, circles = circle_layout(n_points)
    support = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.2 ** 2
    rng = np.random.default_rng(7)
    pops = {s: np.zeros_like(X) for s in SPECIES}
    for cx, cy, k in circles:
        disc = ((X - cx) ** 2 + (Y - cy) ** 2 <= 0.05 ** 2).astype(float)
        if system == "cmx":
            sp = "CMX"
        elif system == "aob":
            sp = "AOB" if rng.random() < 0.5 else "NOB"
        else:
            sp = SPECIES[k]
        pops[sp] += disc
    total = sum(pops.values())
    total[total == 0] = 1.0
    return {s: np.where(support, pops[s] / total, 0.0).ravel() for s in SPECIES}


def make_coeffs(preset):
    c = elliptic_coefficients(preset)
    c["Dhat"] = {s: D_MOT for s in SPECIES}
    c["Ahat"] = {s: A_MOT for s in SPECIES}
    return c


def run(preset, system, N=99, T=50.0, dt=0.5, nh4_bc=1.0, o2_bc=None, out=None, verbose=True):
    coeffs = make_coeffs(preset)
    if o2_bc is None:
        o2_bc = coeffs["c_inf_hat"]["O2"]
    grid = Grid2D(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    U = initial_biomass(N + 1, system)
    specs = {"NH4": ("dirichlet", nh4_bc), "O2": ("dirichlet", o2_bc),
             "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0)}
    V, bnd = grid.Vflat, grid.boundary_flat
    n_steps = int(round(T / dt))
    snap_every = max(1, int(round(SNAP_EVERY_T / dt)))

    hist = {"t": [], "M": {s: [] for s in SPECIES}, "stock": {s: [] for s in SUBSTRATES},
            "rate": {s: [] for s in SUBSTRATES}, "boundary_mass": [], "cut_mass": [],
            "newton_iters": [], "stalled": 0}

    def record(t, U, C, cut):
        R, _ = reaction_and_jacobian(C, U, coeffs)
        hist["t"].append(t)
        for s in SPECIES:
            hist["M"][s].append(float(np.sum(V * U[s])))
        for s in SUBSTRATES:
            hist["stock"][s].append(float(np.sum(V * C[s])))
            hist["rate"][s].append(float(np.sum(V * R[s])))
        hist["boundary_mass"].append(float(sum(np.sum(V[bnd] * U[s][bnd]) for s in SPECIES)))
        hist["cut_mass"].append(cut)

    def snapshot(t, U, C):
        if out is None:
            return
        for s in SPECIES:
            np.save(out / f"t{t:g}_u_{s}.npy", U[s])
        for s in SUBSTRATES:
            np.save(out / f"t{t:g}_c_{s}.npy", C[s])

    t0 = time.time()
    touch_t = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C, h, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150)
    record(0.0, U, C, 0.0)
    snapshot(0, U, C)
    for k in range(1, n_steps + 1):
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt, n_steps=1)
        cut = 0.0
        for s in SPECIES:
            small = U[s] < U_CUT
            cut += float(np.sum(V[small] * U[s][small]))
            U[s][small] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C, h, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150, C_init=C)
            if method == "newton_stalled":
                C, h, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150)
        hist["newton_iters"].append(len(h) - 1)
        hist["stalled"] += method == "newton_stalled"
        hist["max_stall_residual"] = max(hist.get("max_stall_residual", 0.0), h[-1]) if method == "newton_stalled" else hist.get("max_stall_residual", 0.0)
        t = k * dt
        record(t, U, C, cut)
        if touch_t is None and hist["boundary_mass"][-1] > 0.0:
            touch_t = t
        if k % snap_every == 0:
            snapshot(t, U, C)
            if verbose:
                print(f"  t={t:6.1f} M={ {s: round(hist['M'][s][-1], 5) for s in SPECIES} } "
                      f"NH4rate={hist['rate']['NH4'][-1]:.3e} O2min={C['O2'].min():.2e} "
                      f"bmass={hist['boundary_mass'][-1]:.1e} stalled={hist['stalled']}", flush=True)
        if touch_t is not None:
            break

    info = {"preset": preset, "system": system, "N": N, "T_requested": T, "T_end": hist["t"][-1],
            "dt": dt, "NH4_BC": nh4_bc, "O2_BC": o2_bc, "D_mot": D_MOT, "A_mot": A_MOT, "U_cut": U_CUT,
            "boundary_touch_t": touch_t, "stalled": hist["stalled"], "max_stall_residual": hist.get("max_stall_residual", 0.0),
            "newton_iters_mean": float(np.mean(hist["newton_iters"])) if hist["newton_iters"] else 0.0,
            "cut_mass_total": float(np.sum(hist["cut_mass"])), "wall_s": time.time() - t0}
    if out is not None:
        np.save(out / "grid_shape.npy", np.array([N, N]))
        (out / "history.json").write_text(json.dumps(hist))
        (out / "run_info.json").write_text(json.dumps(info, indent=2))
    return hist, info


if __name__ == "__main__":
    a = sys.argv[1:]
    preset, system = a[0], a[1]
    N = int(a[2]) if len(a) > 2 else 99
    T = float(a[3]) if len(a) > 3 else 50.0
    dt = float(a[4]) if len(a) > 4 else 0.5
    nh4 = float(a[5]) if len(a) > 5 else 1.0
    o2 = float(a[6]) if len(a) > 6 and a[6] != "default" else None
    tag = a[7] if len(a) > 7 else ""
    out = Path(__file__).parent / "results" / f"{preset}_{system}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{preset} {system} N={N} T={T} dt={dt} NH4_BC={nh4} O2_BC={o2 or 'preset default'}")
    hist, info = run(preset, system, N, T, dt, nh4, o2, out)
    print(json.dumps(info, indent=2))
