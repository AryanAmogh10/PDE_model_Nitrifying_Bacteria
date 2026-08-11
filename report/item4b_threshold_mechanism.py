"""
ITEM 4 follow-up: resolve (not just report) the ~100x gap between the naive
d << L^2/T scaling estimate and the bisected sector-formation threshold
(d_i in [4.2e-4, 5.6e-4], see report/item4_diffusion_threshold_sweep.py).

Reuses that script's coeffs/geometry/sector-purity machinery directly (this
is a continuation of the same investigation, not an independent
re-derivation -- ITEM 5 already covered independent re-derivation for a
different set of claims).

INVESTIGATION PLAN
  A. Compute Da = r_max*L^2/d at the bisected threshold for several candidate
     length scales L (domain size, inter-seed spacing, seed radius, grid
     spacing h) to see which gives an O(1)-ish value.
  B. Grid-resolution check: rerun the SAME coarse-bracket + bisection
     procedure at a finer grid (Nx=40 instead of 28). If the threshold in d_i
     shifts by roughly the h^2 ratio between the two resolutions, the
     "threshold" is a numerical-diffusion-resolution artifact, not physics
     tied to the seed geometry. If it stays fixed (in physical d_i, not in
     h-relative units), it reflects a real length scale in the problem
     (candidate: seed radius or inter-seed spacing).
  C. Advection-strength check: fix d_i near the (28x28) threshold and sweep
     A_OVER_D_RATIO across {3, 10, 30} to test whether cross-diffusion
     strength is a second parameter the single-parameter (d_i only) story is
     missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import item4_diffusion_threshold_sweep as base
from nitrifiers.grid2d import Grid2D
from nitrifiers.parabolic import SPECIES
from nitrifiers.elliptic2d import solve_newton_2d
from nitrifiers.parabolic2d import solve_parabolic_2d


def run_one_custom(d_i: float, a_over_d: float, grid_n: int, n_steps: int,
                    dt_slow: float = 1.0) -> dict:
    """Same as base.run_one but allows overriding A_OVER_D_RATIO (base.run_one
    hardcodes base.A_OVER_D_RATIO), and returns the colony-radius TIME SERIES
    (not just the final value) so a front speed can be measured directly."""
    grid = Grid2D(Nx=grid_n, Ny=grid_n, Lx=1.0, Ly=1.0)
    U0 = base._seed_profile(grid)

    coeffs = base._make_coeffs(d_i)
    coeffs["Ahat"] = {sp: a_over_d * d_i for sp in SPECIES}

    U = {sp: U0[sp].copy() for sp in SPECIES}
    radii, purities, times = [], [], []
    t0 = time.time()
    for step in range(n_steps):
        C, hist, method = solve_newton_2d(coeffs, U, grid, bc_type="dirichlet",
                                           tol=1e-6, maxiter=100)
        U, _ = solve_parabolic_2d(coeffs, C, U, grid, dt=dt_slow, n_steps=1)
        rho = sum(U[sp] for sp in SPECIES)
        r_col = base._colony_radius(grid, rho)
        purity = base._sector_purity(grid, U, r_col)
        radii.append(r_col)
        purities.append(purity)
        times.append((step + 1) * dt_slow)
    wall = time.time() - t0

    r_col_final = radii[-1]
    purity_final = purities[-1]
    verdict = "SECTORS" if purity_final >= 4 / 6 else ("mixed" if purity_final > 0 else "UNIFORM")
    return {"d_i": d_i, "a_over_d": a_over_d, "grid_n": grid_n,
            "times": times, "radii": radii, "purities": purities,
            "r_colony": r_col_final, "purity": purity_final, "verdict": verdict,
            "wall_s": wall}


def bisect_threshold(grid_n: int, n_steps: int = 15, dt_slow: float = 1.0,
                      coarse_ds=(1e-6, 1e-4, 1e-2, 1.0), rounds: int = 4,
                      a_over_d: float = 10.0):
    results = {}
    for d in coarse_ds:
        r = run_one_custom(d, a_over_d, grid_n, n_steps, dt_slow)
        results[d] = r
        print(f"    d={d:.0e}: r_colony={r['r_colony']:.3f} purity={r['purity']:.2f} "
              f"-> {r['verdict']:8s} ({r['wall_s']:.1f}s)")

    ordered = sorted(results.items())
    lo, hi = None, None
    for (d1, r1), (d2, r2) in zip(ordered, ordered[1:]):
        if r1["verdict"] != "UNIFORM" and r2["verdict"] == "UNIFORM":
            lo, hi = d1, d2
            break
    if lo is None:
        print("    No SECTORS->UNIFORM flip found in the coarse bracket.")
        return results, None, None

    for it in range(rounds):
        mid = np.sqrt(lo * hi)
        r = run_one_custom(mid, a_over_d, grid_n, n_steps, dt_slow)
        results[mid] = r
        print(f"    it={it} d={mid:.3e}: r_colony={r['r_colony']:.3f} "
              f"purity={r['purity']:.2f} -> {r['verdict']:8s} ({r['wall_s']:.1f}s)")
        if r["verdict"] == "UNIFORM":
            hi = mid
        else:
            lo = mid
    return results, lo, hi


if __name__ == "__main__":
    print("=" * 78)
    print("PART A: candidate Damkohler numbers Da = r_max*L^2/d at the")
    print("28x28-grid bisected threshold (already known: d in [4.2e-4, 5.6e-4])")
    print("=" * 78)
    r_max, L_domain = 1.0, 1.0
    d_mid_28 = np.sqrt(4.217e-4 * 5.623e-4)
    h28 = L_domain / 28
    seed_r = 0.10
    spacing = 2 * 0.15 * np.sin(np.radians(60))
    print(f"\nGeometric-mean bisected threshold (28x28 grid): d_mid = {d_mid_28:.4e}")
    print(f"sqrt(d_mid/r_max) [diffusion length in one reaction time]: "
          f"{np.sqrt(d_mid_28 / r_max):.4f}")
    for name, Lval in [("domain L=1", L_domain), ("inter-seed spacing", spacing),
                        ("seed radius", seed_r), ("grid spacing h (28x28)", h28)]:
        Da = r_max * Lval ** 2 / d_mid_28
        print(f"  L = {name:26s} = {Lval:.4f}  ->  Da = r*L^2/d = {Da:8.3f}")
    print("\nGrid spacing gives Da closest to O(1) -- this is a warning sign for a "
          "resolution artifact, not real physics tied to seed geometry. Testing "
          "directly in Part B.")

    print("\n" + "=" * 78)
    print("PART B: grid-resolution check -- does the threshold shift with h?")
    print("=" * 78)
    print("\n-- Resolution 28x28 (reproducing item4's original result) --")
    res28, lo28, hi28 = bisect_threshold(grid_n=28)
    print("\n-- Resolution 40x40 (finer: h shrinks from 1/28=0.0357 to 1/40=0.025) --")
    res40, lo40, hi40 = bisect_threshold(grid_n=40)

    if lo28 and lo40:
        mid28 = np.sqrt(lo28 * hi28)
        mid40 = np.sqrt(lo40 * hi40)
        h_ratio_sq = (1 / 28) ** 2 / (1 / 40) ** 2
        d_ratio = mid28 / mid40
        print(f"\nThreshold (28x28): d_mid = {mid28:.4e}")
        print(f"Threshold (40x40): d_mid = {mid40:.4e}")
        print(f"Observed threshold ratio d_28/d_40 = {d_ratio:.3f}")
        print(f"Predicted ratio if purely a grid (h^2) artifact: "
              f"(h_28/h_40)^2 = {h_ratio_sq:.3f}")
        print(f"Predicted ratio if threshold is resolution-INDEPENDENT "
              f"(real physics): 1.0")
        dist_to_artifact = abs(np.log(d_ratio) - np.log(h_ratio_sq))
        dist_to_physics = abs(np.log(d_ratio) - np.log(1.0))
        verdict = "GRID ARTIFACT (numerical)" if dist_to_artifact < dist_to_physics \
            else "RESOLUTION-INDEPENDENT (real physics)"
        print(f"\nObserved ratio is closer (in log-space) to: {verdict}")

    print("\n" + "=" * 78)
    print("PART C: does advection strength (A_OVER_D_RATIO) shift the threshold?")
    print("=" * 78)
    d_probe = np.sqrt(4.217e-4 * 5.623e-4)   # geometric mean of the 28x28 bracket
    print(f"\nFixed d_i = {d_probe:.3e} (28x28-grid bisected midpoint, right at "
          f"the transition for A_OVER_D_RATIO=10), sweeping A_OVER_D_RATIO:")
    for a_ratio in (3.0, 10.0, 30.0, 100.0):
        r = run_one_custom(d_probe, a_ratio, grid_n=28, n_steps=15, dt_slow=1.0)
        print(f"  A_OVER_D_RATIO={a_ratio:6.1f}: r_colony={r['r_colony']:.3f} "
              f"purity={r['purity']:.2f} -> {r['verdict']:8s} ({r['wall_s']:.1f}s)")
