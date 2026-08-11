"""
ITEM 4: redesigned diffusion-length threshold sweep for 2D sector formation
(closes out B8(c) from report/audit_checklist.md).

BACKGROUND. B8(a)/(b) reproduced genuine sector formation with the paper's
own Table 1 Case (A) parameters (d_i=1e-6, a_i=1e-5, r_i=1, K_i=1, b_i=0.1,
Y_i=0.2, D=1e-4), by reconfiguring the 2D solver's coefficient structure to
approximate System (2.3) (three symmetric species competing for ONE shared
substrate) rather than this project's nitrifier-specific trophic network
(AOB -> NO2 -> NOB, etc, which is a food CHAIN, not direct competition).
B8(c) swept d_i to test whether the predicted diffusion-length threshold
(d_i << L^2/T) is real, but that sweep used a COARSER grid/larger dt than the
single confirmed run purely because a finer version timed out -- and the
inconsistent resolution across sweep points is exactly what produced a
non-monotonic (hence uninterpretable) interior of the sweep. That specific
flaw -- resolution changing point-to-point under compute pressure -- is what
this script fixes, not the physical setup, which is unchanged from B8.

REDESIGN, addressing the flaw directly:
  1. ONE fixed, modest resolution for every single point in the sweep (no
     point gets a finer or coarser grid than any other), so a non-monotonic
     result can no longer be blamed on inconsistent numerics.
  2. A coarse bracketing pass (4 log-spaced points across the 6-decade range
     already known to span SECTORS -> UNIFORM) followed by BISECTION
     refinement of the bracket where the classification flips -- i.e. compute
     is spent narrowing the actual transition, not re-confirming points deep
     inside a region whose classification is already unambiguous. This is
     the "bisection-style sampling" requested: fewer total runs than either
     the original 5-point sweep OR a naive fine uniform grid, while directly
     targeting the threshold location instead of the whole range.
  3. Grid/timestep budget picked to keep each run to a few seconds: 28x28
     (vs. the confirmed run's 48x48), dt_slow=1.0, 15 slow steps (vs. T=50),
     elliptic_tol=1e-6 (vs. 1e-10). This is explicitly coarser than the B8(a)/
     (b) confirmed run and not a substitute for it -- it is a *relative*,
     same-resolution-throughout comparison across d_i, not a replacement for
     the finer single-point confirmation already on record.

REPRODUCING THE SYSTEM-2.3 APPROXIMATION. The nitrifier reaction network
(nondim.py's CONSUMED_SUBSTRATES/PRODUCTION) structurally has AOB producing
NO2 (NOB's own primary substrate) -- a trophic cascade, not the paper's direct
shared-resource competition. To approximate direct competition instead:
  - Every species gets the SAME primary-substrate Monod constant (Khat=1) and
    the SAME Damkohler-like consumption strength (Lambda), so their growth
    laws are literally identical functions of their own primary substrate,
    matching System (2.3)'s single growth law r*c/(K+c)-b applied to 3
    symmetric competitors.
  - The nitrifier network still requires two DIFFERENT substrate names as
    "primary" (NH4 for AOB/CMX, NO2 for NOB) -- both are given IDENTICAL
    Dirichlet boundary values and Khat/Lambda, so they behave as two
    identically-parameterised copies of the same resource rather than a
    genuinely single field. This is an approximation, stated plainly.
  - Cross-feeding (AOB/NOB/CMX "producing" downstream substrate) is switched
    off entirely (beta=0 for every conversion), removing the food-chain
    dependency the raw network would otherwise impose -- without this, NOB
    would depend on AOB's output rather than competing independently, which
    is not System (2.3).
  - O2 (the obligatory secondary substrate in this project's Monod term) is
    made non-limiting by giving it a vanishingly small Khat (1e-8) and an
    ample, never-depleted Dirichlet supply (Lambda_O2 negligible) -- so
    M(c_O2; Khat_O2) approx 1 everywhere, and growth is controlled by the
    primary substrate alone, matching System (2.3)'s single-substrate growth
    term.

This is the SAME kind of reconfiguration B8(a)/(b) used (a documented
approximation of a 4-substrate trophic network standing in for a 1-substrate
symmetric competition model), not a literal re-implementation of System
(2.3)'s own PDE from scratch -- that from-scratch route is what ITEM 2 took
for the 1D travelling-wave benchmark, and is available as a template if a
future pass wants a first-principles 2D reproduction instead.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.grid2d import Grid2D
from nitrifiers.parabolic import SPECIES
from nitrifiers.slowfast2d import run_slow_loop_2d


# ---------------------------------------------------------------------------
# Fixed physical setup, matching B8(a)/(b)'s confirmed parameters except for
# resolution/duration (see module docstring point 3).
# ---------------------------------------------------------------------------
A_OVER_D_RATIO = 10.0
R_MAIN, K_MAIN, B_MAIN, Y_MAIN = 1.0, 1.0, 0.1, 0.2
D_SUBSTRATE = 1e-4          # paper's Table 1 Case (A) substrate diffusivity
C_INF = 5.0                  # Dirichlet boundary value (B8's flagged deviation
                              # from the paper's own near-zero Neumann influx)
SEED_ANGLES_DEG = (0.0, 120.0, 240.0)
SEED_RADIUS = 0.10
SEED_AMPLITUDE = 0.05

GRID_N = 28
DT_SLOW = 1.0
N_SLOW_STEPS = 15


def _make_coeffs(d_i: float) -> dict:
    """Hand-built coeffs dict approximating System (2.3), see module docstring."""
    # eps = r_max*L^2/D matches the paper's own Case-A value (1e4); Lambda for
    # the primary substrate is chosen as eps*Da with Da=1/Y (u_ref=c_ref=1),
    # i.e. Lambda_primary = R_MAIN/(D_SUBSTRATE*Y_MAIN) -- the identity nondim.py
    # documents (Lambda reduces to rhat*eps*Da for the substrate's own D).
    lam_primary = R_MAIN / (D_SUBSTRATE * Y_MAIN)
    lam_o2 = 1e-6 * lam_primary   # negligible O2 consumption -> O2 stays ~c_inf
    khat_o2 = 1e-8                # O2 saturates M(c;K)~1 for any c > 0

    Dhat = {sp: d_i for sp in SPECIES}
    Ahat = {sp: A_OVER_D_RATIO * d_i for sp in SPECIES}
    rhat = {sp: 1.0 for sp in SPECIES}
    bhat = {sp: B_MAIN for sp in SPECIES}

    Lambda = {
        "AOB": {"NH4": lam_primary, "O2": lam_o2},
        "NOB": {"NO2": lam_primary, "O2": lam_o2},
        "CMX": {"NH4": lam_primary, "O2": lam_o2},
    }
    LambdaProd = {sp: lam_primary for sp in SPECIES}   # unused: beta=0 below
    Khat = {
        "AOB": {"NH4": K_MAIN, "O2": khat_o2},
        "NOB": {"NO2": K_MAIN, "O2": khat_o2},
        "CMX": {"NH4": K_MAIN, "O2": khat_o2},
    }
    c_inf_hat = {"NH4": C_INF, "NO2": C_INF, "NO3": 0.0, "O2": C_INF}
    beta = {"AOB_to_NO2": 0.0, "NOB_to_NO3": 0.0, "CMX_to_NO3": 0.0}
    production = {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"),
                  "CMX": ("NO3", "CMX_to_NO3")}
    consumed = {"AOB": ("NH4", "O2"), "NOB": ("NO2", "O2"), "CMX": ("NH4", "O2")}

    return {"Lambda": Lambda, "LambdaProd": LambdaProd, "Khat": Khat,
            "rhat": rhat, "bhat": bhat, "Dhat": Dhat, "Ahat": Ahat,
            "c_inf_hat": c_inf_hat, "beta": beta,
            "consumed_substrates": consumed, "production": production,
            "eps": R_MAIN * 1.0 ** 2 / D_SUBSTRATE}


def _seed_profile(grid: Grid2D) -> dict:
    U0 = {sp: np.zeros(grid.Npts) for sp in SPECIES}
    cx, cy = 0.5, 0.5
    for sp, ang_deg in zip(SPECIES, SEED_ANGLES_DEG):
        ang = np.radians(ang_deg)
        x0 = cx + 0.15 * np.cos(ang)
        y0 = cy + 0.15 * np.sin(ang)
        r2 = (grid.X - x0) ** 2 + (grid.Y - y0) ** 2
        U0[sp] = (SEED_AMPLITUDE * np.exp(-r2 / SEED_RADIUS ** 2)).ravel()
    return U0


def _colony_radius(grid: Grid2D, rho: np.ndarray, frac_of_peak: float = 0.01) -> float:
    """Radius (from domain centre) at which radially-binned mean density falls
    below `frac_of_peak` of its own peak. Returns 0.0 if rho never rises above
    a negligible floor anywhere (no growth yet)."""
    r = grid.radius().ravel()
    if rho.max() < 1e-8:
        return 0.0
    nbins = 40
    edges = np.linspace(0, r.max(), nbins + 1)
    binned = np.zeros(nbins)
    for k in range(nbins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        binned[k] = rho[m].mean() if m.any() else 0.0
    peak = binned.max()
    if peak < 1e-8:
        return 0.0
    thresh = frac_of_peak * peak
    above = np.flatnonzero(binned > thresh)
    if above.size == 0:
        return 0.0
    return float(edges[above[-1] + 1])


def _sector_purity(grid: Grid2D, U: dict, r_colony: float, n_sectors: int = 6,
                    dominance_ratio: float = 2.0) -> float:
    """Fraction of angular sectors, within the active outer shell
    [0.5*r_colony, r_colony], where one species' mass exceeds `dominance_ratio`
    times the next-highest species' mass in that same sector."""
    if r_colony <= 1e-6:
        return 0.0
    r = grid.radius().ravel()
    theta = np.degrees(np.arctan2((grid.Y - 0.5).ravel(), (grid.X - 0.5).ravel()))
    shell = (r >= 0.5 * r_colony) & (r <= r_colony)
    if not shell.any():
        return 0.0

    edges = np.linspace(-180, 180, n_sectors + 1)
    clear = 0
    for k in range(n_sectors):
        sector_mask = shell & (theta >= edges[k]) & (theta < edges[k + 1])
        if not sector_mask.any():
            continue
        masses = sorted((float(np.sum(grid.Vflat[sector_mask] * U[sp][sector_mask]))
                          for sp in SPECIES), reverse=True)
        if masses[0] > 1e-10 and masses[0] > dominance_ratio * max(masses[1], 1e-12):
            clear += 1
    return clear / n_sectors


def run_one(d_i: float, grid_n: int = GRID_N, n_steps: int = N_SLOW_STEPS,
            dt_slow: float = DT_SLOW) -> dict:
    coeffs_kwargs = None  # placeholder for clarity; coeffs built below
    grid = Grid2D(Nx=grid_n, Ny=grid_n, Lx=1.0, Ly=1.0)
    U0 = _seed_profile(grid)

    # run_slow_loop_2d calls elliptic_coefficients(preset_name) internally, so
    # bypass it: replicate its inner loop directly against our hand-built coeffs.
    from nitrifiers.elliptic2d import solve_newton_2d
    from nitrifiers.parabolic2d import solve_parabolic_2d, total_mass_2d

    coeffs = _make_coeffs(d_i)
    U = {sp: U0[sp].copy() for sp in SPECIES}
    t0 = time.time()
    for _ in range(n_steps):
        C, hist, method = solve_newton_2d(coeffs, U, grid, bc_type="dirichlet",
                                           tol=1e-6, maxiter=100)
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt_slow, n_steps=1)
    dt = time.time() - t0

    rho = sum(U[sp] for sp in SPECIES)
    r_colony = _colony_radius(grid, rho)
    purity = _sector_purity(grid, U, r_colony)
    verdict = "SECTORS" if purity >= 4 / 6 else ("mixed" if purity > 0 else "UNIFORM")
    return {"d_i": d_i, "r_colony": r_colony, "purity": purity, "verdict": verdict,
            "elliptic_method": method, "elliptic_resid": hist[-1] if hasattr(hist, "__getitem__") else hist,
            "wall_s": dt}


if __name__ == "__main__":
    print("=" * 78)
    print("ITEM 4: redesigned B8(c) diffusion-length threshold sweep")
    print(f"(fixed resolution {GRID_N}x{GRID_N}, dt_slow={DT_SLOW}, "
          f"{N_SLOW_STEPS} steps, for EVERY point)")
    print("=" * 78)

    print("\n--- Coarse bracketing pass (log-spaced, same resolution throughout) ---")
    coarse_ds = [1e-6, 1e-4, 1e-2, 1.0]
    results = {}
    for d in coarse_ds:
        r = run_one(d)
        results[d] = r
        print(f"  d={d:.0e}: r_colony={r['r_colony']:.3f} purity={r['purity']:.2f} "
              f"-> {r['verdict']:8s} ({r['wall_s']:.1f}s, {r['elliptic_method']})")

    # find first flip SECTORS/mixed -> UNIFORM along increasing d
    ordered = sorted(results.items())
    lo, hi = None, None
    for (d1, r1), (d2, r2) in zip(ordered, ordered[1:]):
        if r1["verdict"] != "UNIFORM" and r2["verdict"] == "UNIFORM":
            lo, hi = d1, d2
            break

    if lo is None:
        print("\nNo SECTORS->UNIFORM flip found in the coarse bracket "
              "[1e-6, 1.0] at this resolution -- cannot bisect further; "
              "reporting the coarse pass as the full result.")
    else:
        print(f"\n--- Bisection refinement within [{lo:.0e}, {hi:.0e}] "
              f"(log-midpoint each iteration) ---")
        for it in range(4):
            mid = np.sqrt(lo * hi)   # geometric mean = log-space bisection
            r = run_one(mid)
            results[mid] = r
            print(f"  it={it} d={mid:.3e}: r_colony={r['r_colony']:.3f} "
                  f"purity={r['purity']:.2f} -> {r['verdict']:8s} ({r['wall_s']:.1f}s)")
            if r["verdict"] == "UNIFORM":
                hi = mid
            else:
                lo = mid
        print(f"\nBisected threshold bracket: d_i in [{lo:.3e}, {hi:.3e}] "
              f"(factor of {hi / lo:.1f} wide)")
        predicted = 1.0 ** 2 / (N_SLOW_STEPS * DT_SLOW)  # L^2/T at this sweep's T
        print(f"Predicted threshold scale d ~ L^2/T = {predicted:.3e} "
              f"(L=1, T={N_SLOW_STEPS * DT_SLOW:.0f})")
        in_range = lo <= predicted <= hi
        print(f"Predicted scale falls inside the bisected bracket: {in_range}")

    print("\n--- Full point list, sorted by d_i ---")
    for d, r in sorted(results.items()):
        print(f"  d={d:.3e}: r_colony={r['r_colony']:.3f} purity={r['purity']:.2f} "
              f"verdict={r['verdict']}")

    verdicts_by_d = [v["verdict"] for _, v in sorted(results.items())]
    rank = {"SECTORS": 2, "mixed": 1, "UNIFORM": 0}
    ranks = [rank[v] for v in verdicts_by_d]
    monotone = all(a >= b for a, b in zip(ranks, ranks[1:]))
    print(f"\nMonotonically non-increasing SECTORS->UNIFORM as d_i grows, "
          f"at this UNIFORM resolution: {monotone}")
