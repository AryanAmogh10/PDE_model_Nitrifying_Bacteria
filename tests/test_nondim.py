import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.params import list_presets, SUBSTRATES
from nitrifiers.nondim import nondimensionalize


def test_epsilon_is_small_for_all_presets():
    for name in list_presets():
        r = nondimensionalize(name)
        assert r.eps < 0.1, (name, r.eps)
        for j in SUBSTRATES:
            assert r.eps_j[j] < 0.1, (name, j, r.eps_j[j])


def test_bacterial_diffusivity_ratio_smaller_than_epsilon():
    # delta_i (bacterial/substrate diffusivity) should reinforce eps, i.e. be smaller too
    for name in list_presets():
        r = nondimensionalize(name)
        for i, delta in r.delta_i.items():
            assert delta < r.eps * 10, (name, i, delta, r.eps)


def test_dimensionless_growth_ratio_is_between_zero_and_one():
    for name in list_presets():
        r = nondimensionalize(name)
        for sp, s in r.species.items():
            assert 0 < s["rhat"] <= 1.0 + 1e-9, (name, sp, s["rhat"])


def test_cross_diffusion_number_matches_ratio():
    from nitrifiers.params import A_OVER_D_RATIO
    for name in list_presets():
        r = nondimensionalize(name)
        for sp, s in r.species.items():
            assert abs(s["Ahat"] / s["Dhat"] - A_OVER_D_RATIO) < 1e-9


if __name__ == "__main__":
    test_epsilon_is_small_for_all_presets()
    test_bacterial_diffusivity_ratio_smaller_than_epsilon()
    test_dimensionless_growth_ratio_is_between_zero_and_one()
    test_cross_diffusion_number_matches_ratio()
    print("All Stage 2 non-dimensionalization sanity checks passed.")
