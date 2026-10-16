"""Shared setup for the AOB vs comammox experiments (base case, heatmap).

Preset kinetics ('eloi' by default) with the paper's Case A motility and a
survival cutoff (both explained in simulation_experiments/README.md);
quasi-steady substrate solve, warm-started each step; Dirichlet substrate
boundaries. Initial biomass: the 300-circle condition, every species
rescaled to the same total mass M0, so AOB and CMX start equal and the
isolated systems are the combined system with species removed."""
import sys, json, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, portable
from nitrifiers.params import load_preset, BIOFILM_DENSITY_KG_M3
from nitrifiers.nondim import elliptic_coefficients, nondimensionalize, SUBSTRATES
from nitrifiers.one_dimensional.parabolic import SPECIES
from nitrifiers.one_dimensional.elliptic import reaction_and_jacobian
from nitrifiers.two_dimensional.grid2d import Grid2D
from nitrifiers.two_dimensional.elliptic2d import solve_newton_2d
from nitrifiers.two_dimensional.parabolic2d import solve_parabolic_2d

D_MOT, A_MOT = 1e-6, 1e-5
U_CUT = 1e-6
SYSTEM_SPECIES = {"aob": ("AOB", "NOB"), "cmx": ("CMX",), "all": ("AOB", "NOB", "CMX")}


class Units:
    """Conversion from solver (nondimensional) values to physical units."""

    def __init__(self, preset):
        nd = nondimensionalize(preset)
        p = load_preset(preset)
        self.tau_day = nd.tau_slow_day
        self.L_um = p["domain"]["L"]
        self.c_ref_uM = nd.c_ref_uM
        self.D = {s: p["substrates"][s]["D"] for s in SUBSTRATES}
        self._L_m = p["domain"]["L"] * 1e-6

    def days(self, t):
        return np.asarray(t) * self.tau_day

    def uM(self, c):
        return np.asarray(c) * self.c_ref_uM

    def biomass_ug_per_mm(self, M):
        # integral of u over the unit square -> packed density * L^2, per unit depth
        return np.asarray(M) * BIOFILM_DENSITY_KG_M3 * self._L_m ** 2 * 1e6

    def amount_umol_per_mm(self, sub, integral):
        # time and area integral of the nondimensional rate -> D_j c_ref tau, per unit depth
        return np.asarray(integral) * self.D[sub] * self.c_ref_uM * 1e-3 * self.tau_day * 1e3


def circle_layout(n_points, random_seed=42):
    """Replays the user's initial_conditions_biomass RNG sequence exactly."""
    np.random.seed(random_seed)
    x = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, x, indexing="ij")
    circles = []
    for _ in range(300):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.2 - 0.05)
        circles.append((0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta), np.random.choice(3)))
    return X, Y, circles


def initial_biomass(grid, system):
    """Original 3-way circle assignment, each species rescaled to the common
    mass M0 (the mean of the three original masses, so the combined total is
    unchanged). Isolated systems drop the absent species."""
    X, Y, circles = circle_layout(grid.Nx + 1)
    support = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.2 ** 2
    pops = {s: np.zeros_like(X) for s in SPECIES}
    for cx, cy, k in circles:
        pops[SPECIES[k]] += ((X - cx) ** 2 + (Y - cy) ** 2 <= 0.05 ** 2).astype(float)
    total = sum(pops.values())
    total[total == 0] = 1.0
    U = {s: np.where(support, pops[s] / total, 0.0).ravel() for s in SPECIES}
    mass = {s: float(np.sum(grid.Vflat * U[s])) for s in SPECIES}
    M0 = float(np.mean(list(mass.values())))
    keep = SYSTEM_SPECIES[system]
    return {s: (U[s] * M0 / mass[s] if s in keep else np.zeros_like(U[s])) for s in SPECIES}


def make_coeffs(preset):
    c = elliptic_coefficients(preset)
    c["Dhat"] = {s: D_MOT for s in SPECIES}
    c["Ahat"] = {s: A_MOT for s in SPECIES}
    return c


def run(preset, system, nh4_bc, o2_bc, N=49, T_max=60.0, dt=0.5, out=None,
        snap_every_t=5.0, verbose=False):
    """nh4_bc, o2_bc nondimensional. Stops at T_max or at the first step with
    biomass on a boundary node. Returns (history, info)."""
    coeffs = make_coeffs(preset)
    grid = Grid2D(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    U = initial_biomass(grid, system)
    specs = {"NH4": ("dirichlet", nh4_bc), "O2": ("dirichlet", o2_bc),
             "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0)}
    V, bnd = grid.Vflat, grid.boundary_flat
    snap_every = max(1, int(round(snap_every_t / dt)))
    hist = {"t": [], "M": {s: [] for s in SPECIES}, "stock": {s: [] for s in SUBSTRATES},
            "rate": {s: [] for s in SUBSTRATES}, "boundary_mass": [], "newton_iters": [], "stalled": 0}

    def solve(C_init=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C, h, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150, C_init=C_init)
            if method == "newton_stalled" and C_init is not None:
                C, h, method = solve_newton_2d(coeffs, U, grid, bc_specs=specs, tol=1e-6, maxiter=150)
        return C, h, method

    def record(t, C):
        R, _ = reaction_and_jacobian(C, U, coeffs)
        hist["t"].append(t)
        for s in SPECIES:
            hist["M"][s].append(float(np.sum(V * U[s])))
        for s in SUBSTRATES:
            hist["stock"][s].append(float(np.sum(V * C[s])))
            hist["rate"][s].append(float(np.sum(V * R[s])))
        hist["boundary_mass"].append(float(sum(np.sum(V[bnd] * U[s][bnd]) for s in SPECIES)))

    def snapshot(t, C):
        if out is not None:
            for s in SPECIES:
                np.save(out / f"t{t:g}_u_{s}.npy", U[s])
            for s in SUBSTRATES:
                np.save(out / f"t{t:g}_c_{s}.npy", C[s])

    t0 = time.time()
    C, h, method = solve()
    record(0.0, C)
    snapshot(0, C)
    touch_t = None
    for k in range(1, int(round(T_max / dt)) + 1):
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt, n_steps=1)
        for s in SPECIES:
            U[s][U[s] < U_CUT] = 0.0
        C, h, method = solve(C_init=C)
        hist["newton_iters"].append(len(h) - 1)
        hist["stalled"] += method == "newton_stalled"
        t = k * dt
        record(t, C)
        if k % snap_every == 0:
            snapshot(t, C)
            if verbose:
                print(f"  t={t:6.1f} M={ {s: round(hist['M'][s][-1], 5) for s in SPECIES} } "
                      f"stalled={hist['stalled']}", flush=True)
        if hist["boundary_mass"][-1] > 0.0:
            touch_t = t
            break

    info = {"preset": preset, "system": system, "NH4_BC": nh4_bc, "O2_BC": o2_bc, "N": N, "dt": dt,
            "T_max": T_max, "T_end": hist["t"][-1], "boundary_touch_t": touch_t,
            "stalled": hist["stalled"], "newton_iters_mean": float(np.mean(hist["newton_iters"])),
            "wall_s": time.time() - t0}
    if out is not None:
        np.save(out / "grid_shape.npy", np.array([N, N]))
        (out / "history.json").write_text(json.dumps(hist))
        (out / "run_info.json").write_text(json.dumps(info, indent=2))
    return hist, info


def last_valid_index(hist, info):
    """Index of the last recorded time before the boundary was touched."""
    return len(hist["t"]) - (2 if info["boundary_touch_t"] is not None else 1)


def f_cmx(hist, i):
    a, c = hist["M"]["AOB"][i], hist["M"]["CMX"][i]
    return c / (a + c) if a + c > 0 else float("nan")


def cumulative(hist, sub):
    """Time integral of the volume-integrated rate (nondimensional)."""
    t, r = np.asarray(hist["t"]), np.asarray(hist["rate"][sub])
    return np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))])
