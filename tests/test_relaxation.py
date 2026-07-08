import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.params import list_presets
from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.elliptic import Grid, SUBSTRATES
from nitrifiers.relaxation import solve_relaxation, compare_with_elliptic


def _uniform_u(Npts, value):
    return {sp: np.full(Npts, value) for sp in ("AOB", "NOB", "CMX")}


def test_relaxation_converges_and_stays_physical():
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=100, geometry="radial", p=2)
    U = _uniform_u(grid.N + 1, 0.05)
    C, hist = solve_relaxation(coeffs, U, grid, bc_type="dirichlet",
                                dt0=1e-2, dt_growth=1.3, max_steps=3000)
    assert hist[-1] < 1e-8
    for sub in SUBSTRATES:
        assert np.all(C[sub] >= -1e-8)


def test_relaxation_matches_newton_across_densities_and_presets():
    for preset in list_presets():
        coeffs = elliptic_coefficients(preset)
        grid = Grid(N=60, geometry="radial", p=2)
        for u_val in (1e-4, 0.01, 0.05, 0.5):
            U = _uniform_u(grid.N + 1, u_val)
            res = compare_with_elliptic(coeffs, U, grid, bc_type="dirichlet",
                                         dt0=1e-2, dt_growth=1.3, max_steps=3000)
            assert max(res["max_diff"].values()) < 1e-6, (preset, u_val, res["max_diff"])


def test_relaxation_reproduces_anoxic_core():
    coeffs = elliptic_coefficients("toy")
    grid = Grid(N=100, geometry="radial", p=2)
    U = _uniform_u(grid.N + 1, 0.05)
    C, hist = solve_relaxation(coeffs, U, grid, bc_type="dirichlet",
                                dt0=1e-2, dt_growth=1.3, max_steps=3000)
    assert C["O2"][0] < 1e-6
    assert C["O2"][-1] > 0.1
    assert np.all(np.diff(C["O2"]) >= -1e-6)


if __name__ == "__main__":
    test_relaxation_converges_and_stays_physical()
    test_relaxation_matches_newton_across_densities_and_presets()
    test_relaxation_reproduces_anoxic_core()
    print("All Stage 4 relaxation solver sanity checks passed.")
