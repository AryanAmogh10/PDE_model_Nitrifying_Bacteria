import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.params import load_preset, list_presets, SPECIES, SUBSTRATES, A_OVER_D_RATIO


def test_all_presets_have_required_fields():
    for name in list_presets():
        p = load_preset(name)
        for sp in SPECIES:
            assert sp in p["species"], f"{name}: missing species {sp}"
            spec = p["species"][sp]
            assert spec["r"] > 0
            assert spec["b"] >= 0
            assert spec["d"] > 0
            assert all(k > 0 for k in spec["K"].values())
            assert all(0 < y < 1 for y in spec["Y"].values()), (name, sp, spec["Y"])
        for sub in SUBSTRATES:
            assert sub in p["substrates"], f"{name}: missing substrate {sub}"
            assert p["substrates"][sub]["D"] > 0
            assert p["substrates"][sub]["c_inf"] >= 0


def test_beta_is_unit_molar_conversion():
    for name in list_presets():
        p = load_preset(name)
        assert p["beta"] == {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0}


def test_bacterial_advection_dominates_diffusion():
    # d_i << a_i, per the assumption in arXiv:2512.13156 Sec. 2.1 -- enforced by construction
    # via the fixed A_OVER_D_RATIO (a_i is derived from d_i, not stored independently, see note 7)
    assert A_OVER_D_RATIO >= 5


def test_eloi_oxygen_yields_match_stoichiometry():
    p = load_preset("eloi")
    aob, nob, cmx = p["species"]["AOB"], p["species"]["NOB"], p["species"]["CMX"]
    assert math.isclose(aob["Y"]["O2"], aob["Y"]["NH4"] / 1.5, rel_tol=1e-9)
    assert math.isclose(nob["Y"]["O2"], nob["Y"]["NO2"] / 0.5, rel_tol=1e-9)
    assert math.isclose(cmx["Y"]["O2"], cmx["Y"]["NH4"] / 2.0, rel_tol=1e-9)


def test_rebeca_yields_are_order_mol_fraction():
    # regression check for the /100 unit-cleaning fix (see params.py note 1)
    p = load_preset("rebeca")
    for sp in ("AOB", "NOB", "CMX"):
        for y in p["species"][sp]["Y"].values():
            assert 0 < y < 1


if __name__ == "__main__":
    test_all_presets_have_required_fields()
    test_beta_is_unit_molar_conversion()
    test_bacterial_advection_dominates_diffusion()
    test_eloi_oxygen_yields_match_stoichiometry()
    test_rebeca_yields_are_order_mol_fraction()
    print("All Stage 1 parameter sanity checks passed.")
