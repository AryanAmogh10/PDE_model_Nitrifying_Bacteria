import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.elliptic import Grid, solve_newton, solve_picard, SUBSTRATES


def _uniform_u(Npts, value):
    return {sp: np.full(Npts, value) for sp in ("AOB", "NOB", "CMX")}


def test_mild_case_converges_quickly_and_stays_physical():
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=100, geometry="radial", p=2)
    U = _uniform_u(grid.N + 1, 1e-4)
    C, hist, method = solve_newton(coeffs, U, grid, bc_type="dirichlet", maxiter=50)
    assert hist[-1] < 1e-8
    assert method == "newton"
    assert len(hist) <= 10
    for sub in SUBSTRATES:
        assert np.all(C[sub] >= -1e-9)
        assert np.all(C[sub] <= coeffs["c_inf_hat"][sub] + 1e-9) or sub in ("NO2", "NO3")


def test_stiff_case_produces_anoxic_core_zonation():
    """With enough bacterial density, O2 should be fully depleted (rim-to-core
    zonation) while NH4 remains partially available -- this is the qualitative
    sanity check requested for Stage 3."""
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=100, geometry="radial", p=2)
    U = _uniform_u(grid.N + 1, 0.05)
    C, hist, method = solve_newton(coeffs, U, grid, bc_type="dirichlet", maxiter=300)
    assert hist[-1] < 1e-8
    assert method == "newton"

    # O2 must be non-negative and monotonically non-decreasing from centre to rim
    assert np.all(C["O2"] >= -1e-9)
    assert np.all(np.diff(C["O2"]) >= -1e-6)
    # a genuine anoxic core: O2 ~ 0 over an interior region
    assert C["O2"][0] < 1e-6
    assert C["O2"][-1] > 0.1  # rim stays oxygenated (bulk feed value)

    # NH4 also non-negative, decays toward centre, but not fully depleted
    assert np.all(C["NH4"] >= -1e-9)
    assert np.all(np.diff(C["NH4"]) >= -1e-6)
    assert C["NH4"][0] > 0.1

    # NO2/NO3 are pure by-products: zero at the rim (no external feed), positive inside
    assert abs(C["NO2"][-1]) < 1e-9
    assert abs(C["NO3"][-1]) < 1e-9
    assert C["NO2"][0] > 0
    assert C["NO3"][0] > 0


def test_newton_and_heavily_relaxed_picard_agree():
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=60, geometry="radial", p=2)
    U = _uniform_u(grid.N + 1, 0.05)
    C_n, _, _ = solve_newton(coeffs, U, grid, bc_type="dirichlet", maxiter=300)
    C_p, hist_p = solve_picard(coeffs, U, grid, bc_type="dirichlet",
                                maxiter=20000, tol=1e-6, relax=0.05)
    for sub in SUBSTRATES:
        assert np.max(np.abs(C_n[sub] - C_p[sub])) < 0.05, sub


if __name__ == "__main__":
    test_mild_case_converges_quickly_and_stays_physical()
    test_stiff_case_produces_anoxic_core_zonation()
    test_newton_and_heavily_relaxed_picard_agree()
    print("All Stage 3 elliptic solver sanity checks passed.")
