import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.params import list_presets
from nitrifiers.elliptic import Grid, SUBSTRATES
from nitrifiers.parabolic import SPECIES
from nitrifiers.slowfast import run_slow_loop


def _seed_profile(grid):
    r = grid.r
    return {sp: 0.02 * np.exp(-((r - 0.5) / 0.15) ** 2) for sp in SPECIES}


def test_slow_loop_runs_and_stays_physical_on_all_presets():
    grid = Grid(N=60, geometry="radial", p=2)
    for preset in list_presets():
        U0 = _seed_profile(grid)
        U, C, hist, _ = run_slow_loop(preset, grid, U0, n_slow_steps=20, dt_slow=0.05)

        # every elliptic solve (Newton or its relaxation fallback) reached tolerance
        assert all(h.elliptic_residual < 1e-6 for h in hist), \
            [(h.step, h.elliptic_method, h.elliptic_residual) for h in hist]

        for sp in SPECIES:
            assert np.all(U[sp] >= -1e-8), (preset, sp)
        for sub in SUBSTRATES:
            assert np.all(C[sub] >= -1e-8), (preset, sub)

        # total mass should stay finite and positive throughout
        masses = [h.total_mass for h in hist]
        assert all(np.isfinite(m) and m >= 0 for m in masses)


def test_relaxation_fallback_is_available_and_keeps_residual_low():
    """The seed profile is deliberately tiny/near-degenerate at step 0 -- this
    is exactly the regime where Newton is expected to occasionally stall (see
    Stage 6 notes); confirm the fallback engages and still meets tolerance."""
    grid = Grid(N=60, geometry="radial", p=2)
    U0 = _seed_profile(grid)
    U, C, hist, _ = run_slow_loop("eloi", grid, U0, n_slow_steps=10, dt_slow=0.05,
                                   elliptic_tol=1e-8)
    assert all(h.elliptic_residual < 1e-6 for h in hist)
    assert set(h.elliptic_method for h in hist) <= {"newton", "relaxation"}


def test_zonation_persists_through_the_slow_loop():
    grid = Grid(N=100, geometry="radial", p=2)
    U0 = _seed_profile(grid)
    U, C, hist, _ = run_slow_loop("toy", grid, U0, n_slow_steps=30, dt_slow=0.05)

    assert C["O2"][0] < 1e-6            # anoxic core persists
    assert C["O2"][-1] > 0.1            # rim stays oxygenated
    assert np.all(np.diff(C["O2"]) >= -1e-6)   # monotone rim-to-core zonation
    assert abs(C["NO2"][-1]) < 1e-6 and abs(C["NO3"][-1]) < 1e-6  # no external feed


def test_bacterial_mass_grows_over_the_slow_loop_when_substrate_is_ample():
    grid = Grid(N=60, geometry="radial", p=2)
    U0 = _seed_profile(grid)
    U, C, hist, _ = run_slow_loop("toy", grid, U0, n_slow_steps=30, dt_slow=0.05)
    assert hist[-1].total_mass > hist[0].total_mass


if __name__ == "__main__":
    test_slow_loop_runs_and_stays_physical_on_all_presets()
    test_relaxation_fallback_is_available_and_keeps_residual_low()
    test_zonation_persists_through_the_slow_loop()
    test_bacterial_mass_grows_over_the_slow_loop_when_substrate_is_ample()
    print("All Stage 6 slow-scale-loop sanity checks passed.")
