"""Independent check of the substrate solver.

The substrate equations are written here directly in dimensional form (uM, m, day) from the model equations,
without using any function of the package except params.py for the numbers, and are solved with a different
method (collocation, scipy.integrate.solve_bvp, fourth order). The result is compared with the finite-volume
Newton solver of the package, which works in dimensionless variables. If the non-dimensionalisation, the
reaction terms, the boundary rows or the solver contained a mistake, the two would not agree.

    python tests/test_independent_reference.py        (prints the table of the thesis)
"""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nitrifiers.params import load_preset
from nitrifiers.nondim import elliptic_coefficients, nondimensionalize
from nitrifiers.one_dimensional.elliptic import Grid, solve_newton

P = load_preset("eloi")
L = P["domain"]["L"] * 1e-6
SP = P["species"]
D = {j: P["substrates"][j]["D"] for j in ("NH4", "NO2", "NO3", "O2")}
UREF = nondimensionalize("eloi").u_ref_uM
CASES = [(22.4, 250.0), (150.0, 250.0), (150.0, 40.0), (3.3, 100.0)]


def u_profile(s):
    """Prescribed biomass profile (uM) of the three species: dense in the middle, thin towards the boundary."""
    base = (0.05 + 0.3 * (1 - s ** 2)) * UREF
    return {"AOB": base, "NOB": 0.8 * base, "CMX": 0.6 * base}


def sources(c, u):
    """Right-hand sides S_j (uM/day): consumption (-) and production (+), written from the model equations."""
    c1, c2, c3, c4 = c
    mono = lambda x, K: x / (K + x)
    rate = {}
    rate["AOB"] = u["AOB"] * SP["AOB"]["r"] * mono(c1, SP["AOB"]["K"]["NH4"]) * mono(c4, SP["AOB"]["K"]["O2"])
    rate["NOB"] = u["NOB"] * SP["NOB"]["r"] * mono(c2, SP["NOB"]["K"]["NO2"]) * mono(c4, SP["NOB"]["K"]["O2"])
    rate["CMX"] = u["CMX"] * SP["CMX"]["r"] * mono(c1, SP["CMX"]["K"]["NH4"]) * mono(c4, SP["CMX"]["K"]["O2"])
    S1 = -rate["AOB"] / SP["AOB"]["Y"]["NH4"] - rate["CMX"] / SP["CMX"]["Y"]["NH4"]
    S2 = rate["AOB"] / SP["AOB"]["Y"]["NH4"] - rate["NOB"] / SP["NOB"]["Y"]["NO2"]
    S3 = rate["NOB"] / SP["NOB"]["Y"]["NO2"] + rate["CMX"] / SP["CMX"]["Y"]["NH4"]
    S4 = -rate["AOB"] / SP["AOB"]["Y"]["O2"] - rate["NOB"] / SP["NOB"]["Y"]["O2"] - rate["CMX"] / SP["CMX"]["Y"]["O2"]
    return [S1, S2, S3, S4]


def reference(nh4, o2):
    """Collocation solution on the slab 0<=s<=1: D_j/L^2 c_j'' + S_j = 0, c_j'(0)=0, c_j(1)=c_j^inf."""
    cinf = np.array([nh4, 0.0, 0.0, o2])
    Dl = np.array([D["NH4"], D["NO2"], D["NO3"], D["O2"]]) / L ** 2

    def f(s, y):
        c = np.maximum(y[:4], 0.0)
        S = np.array(sources(c, u_profile(s)))
        return np.vstack([y[4:], -S / Dl[:, None]])

    def bc(ya, yb):
        return np.concatenate([ya[4:], yb[:4] - cinf])

    # starting guess: any rough profile will do, the collocation solver checks the residual itself; we use a
    # coarse solution of the package on 100 intervals so that the stiff layer near c = 0 is resolved
    s, C, _ = package_solution(nh4, o2, 100)
    c0 = np.array([C[k] for k in ("NH4", "NO2", "NO3", "O2")])
    y0 = np.vstack([c0, np.gradient(c0, s, axis=1)])
    sol = solve_bvp(f, bc, s, y0, tol=1e-4, max_nodes=30000, bc_tol=1e-8)
    return sol, cinf, Dl


def package_solution(nh4, o2, N):
    coeffs = elliptic_coefficients("eloi")
    cref = nondimensionalize("eloi").c_ref_uM
    grid = Grid(N=N, geometry="slab")
    s = grid.r
    U = {k: v / UREF for k, v in u_profile(s).items()}
    specs = {"NH4": ("dirichlet", nh4 / cref), "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", o2 / cref)}
    C, hist, method = solve_newton(coeffs, U, grid, bc_specs=specs, tol=1e-10, maxiter=100)
    return s, {k: v * cref for k, v in C.items()}, hist[-1]


def compare(nh4, o2, Ns=(25, 50, 100, 200, 400)):
    sol, cinf, Dl = reference(nh4, o2)
    out = []
    for N in Ns:
        s, C, res = package_solution(nh4, o2, N)
        ref = sol.sol(s)[:4]
        err = max(np.max(np.abs(np.array([C[k] for k in ("NH4", "NO2", "NO3", "O2")])[j] - ref[j])) / cinf.max() for j in range(4))
        out.append((N, err))
    # global balance on the package solution at N=400: D c'(1) / L^2 + int S ds = 0 (symmetry at s=0)
    s, C, res = package_solution(nh4, o2, Ns[-1])
    c = np.array([C[k] for k in ("NH4", "NO2", "NO3", "O2")])
    S = np.array(sources(c, u_profile(s)))
    h = s[1] - s[0]
    flux = Dl * (3 * c[:, -1] - 4 * c[:, -2] + c[:, -3]) / (2 * h)
    integ = np.trapezoid(S, s, axis=1)
    bal = np.max(np.abs(flux + integ)) / np.max(np.abs(integ))
    return out, bal, sol.status


def test_package_matches_independent_reference():
    for nh4, o2 in CASES[1:3]:
        out, bal, status = compare(nh4, o2, Ns=(50, 100, 200))
        assert status == 0
        assert out[-1][1] < 4e-4
        assert out[-1][1] < out[0][1] / 8       # roughly second order
        assert bal < 5e-2


if __name__ == "__main__":
    for nh4, o2 in CASES:
        out, bal, status = compare(nh4, o2)
        print(f"NH4={nh4}, O2={o2}, solve_bvp status {status}")
        for N, e in out:
            print(f"   N={N:4d}  max difference / c_inf = {e:.2e}")
        print(f"   global balance (flux vs integral of reactions), relative: {bal:.2e}")
