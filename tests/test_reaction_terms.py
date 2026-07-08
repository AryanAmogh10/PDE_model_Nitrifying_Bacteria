"""
Regression tests for reaction_and_jacobian() using an independently
hand-computed ground truth (not derived by calling any nitrifiers code),
built from a *synthetic* coefficient set with heterogeneous Lambda/LambdaProd/
Khat per species/substrate -- exactly the property the real 'eloi' preset has
and 'toy'/'rebeca' mostly don't, which is what let two real bugs slip past
the original test suite:

  1. Uptake_i was computed as rhat_i * M(...) * M(...) and then multiplied by
     Lambda[i][j] (which already contains the full r_i), double-counting the
     growth rate. Invisible on toy/rebeca (rhat_i ~ 1 or nearly equal across
     species); a ~13x distortion of CMX's consumption on eloi (r_CMX/r_AOB
     ~ 0.073).
  2. The production term (e.g. NO2 produced by AOB) reused the *consumption*
     Lambda (built with the SOURCE substrate's diffusivity, D_NH4) instead of
     a production-specific Lambda built with the PRODUCED substrate's
     diffusivity (D_NO2). Invisible when all substrates share one D
     (toy/rebeca); a real (if smaller, ~2.5%) error on eloi.

This test uses deliberately different Lambda/LambdaProd values per
species/substrate (so bug 2's D_source-vs-D_produced mixup would show up) and
a nontrivial 'rhat' entry that current code must NOT multiply in (so bug 1's
reappearance would also show up), and checks against values computed by an
independent from-scratch formula below.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.elliptic import reaction_and_jacobian, SUBSTRATES


def _monod(c, K):
    return c / (K + c)


def _dmonod(c, K):
    return K / (K + c) ** 2


def _synthetic_coeffs():
    return {
        "Lambda": {
            "AOB": {"NH4": 2.0, "O2": 3.0},
            "NOB": {"NO2": 5.0, "O2": 7.0},
            "CMX": {"NH4": 11.0, "O2": 13.0},
        },
        "LambdaProd": {"AOB": 17.0, "NOB": 19.0, "CMX": 23.0},
        "Khat": {
            "AOB": {"NH4": 0.1, "O2": 0.1},
            "NOB": {"NO2": 0.1, "O2": 0.1},
            "CMX": {"NH4": 0.1, "O2": 0.1},
        },
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "production": {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"),
                        "CMX": ("NO3", "CMX_to_NO3")},
        # deliberately nontrivial and heterogeneous: current code must NOT use
        # this for the reaction term (see bug 1 above); present so a
        # regression reintroducing rhat_i into Uptake_i would be caught.
        "rhat": {"AOB": 1.0, "NOB": 2.5, "CMX": 4.0},
    }


def _hand_computed_reference():
    C = {"NH4": np.array([0.5]), "NO2": np.array([0.3]),
         "NO3": np.array([0.2]), "O2": np.array([0.4])}
    U = {"AOB": np.array([1.0]), "NOB": np.array([1.0]), "CMX": np.array([1.0])}

    K = 0.1
    Mp_NH4 = _monod(0.5, K)
    Mp_NO2 = _monod(0.3, K)
    Mp_O2 = _monod(0.4, K)

    uptake_AOB = Mp_NH4 * Mp_O2
    uptake_NOB = Mp_NO2 * Mp_O2
    uptake_CMX = Mp_NH4 * Mp_O2  # same substrates as AOB

    Lambda = {"AOB": {"NH4": 2.0, "O2": 3.0}, "NOB": {"NO2": 5.0, "O2": 7.0},
              "CMX": {"NH4": 11.0, "O2": 13.0}}
    LambdaProd = {"AOB": 17.0, "NOB": 19.0, "CMX": 23.0}

    R_NH4 = -Lambda["AOB"]["NH4"] * uptake_AOB - Lambda["CMX"]["NH4"] * uptake_CMX
    R_O2 = (-Lambda["AOB"]["O2"] * uptake_AOB - Lambda["NOB"]["O2"] * uptake_NOB
            - Lambda["CMX"]["O2"] * uptake_CMX)
    R_NO2 = LambdaProd["AOB"] * uptake_AOB - Lambda["NOB"]["NO2"] * uptake_NOB
    R_NO3 = LambdaProd["NOB"] * uptake_NOB + LambdaProd["CMX"] * uptake_CMX

    # Jacobian entry dR_NH4/dc_NH4 (needed by both AOB and CMX terms, same
    # substrates/K for both in this synthetic case)
    dMp_dNH4 = _dmonod(0.5, K)
    duptake_dNH4 = dMp_dNH4 * Mp_O2  # d(uptake_AOB)/d(c_NH4) == d(uptake_CMX)/d(c_NH4)
    dR_NH4_dNH4 = -Lambda["AOB"]["NH4"] * duptake_dNH4 - Lambda["CMX"]["NH4"] * duptake_dNH4

    return C, U, {"NH4": R_NH4, "NO2": R_NO2, "NO3": R_NO3, "O2": R_O2}, dR_NH4_dNH4


def test_reaction_terms_match_independent_hand_computation():
    coeffs = _synthetic_coeffs()
    C, U, R_expected, dR_NH4_dNH4_expected = _hand_computed_reference()

    R, dR = reaction_and_jacobian(C, U, coeffs)

    for sub in SUBSTRATES:
        assert np.isclose(R[sub][0], R_expected[sub], rtol=1e-10), \
            (sub, R[sub][0], R_expected[sub])

    assert np.isclose(dR[("NH4", "NH4")][0], dR_NH4_dNH4_expected, rtol=1e-10)


def test_production_uses_produced_substrates_lambda_not_source_lambda():
    """Directly isolates bug 2: if LambdaProd is swapped out for Lambda[i][source]
    (the old, buggy behaviour), R_NO2/R_NO3 must change -- i.e. LambdaProd is
    actually load-bearing, not silently ignored."""
    coeffs = _synthetic_coeffs()
    C = {"NH4": np.array([0.5]), "NO2": np.array([0.3]),
         "NO3": np.array([0.2]), "O2": np.array([0.4])}
    U = {"AOB": np.array([1.0]), "NOB": np.array([1.0]), "CMX": np.array([1.0])}

    R_correct, _ = reaction_and_jacobian(C, U, coeffs)

    buggy_coeffs = _synthetic_coeffs()
    buggy_coeffs["LambdaProd"] = {
        "AOB": buggy_coeffs["Lambda"]["AOB"]["NH4"],
        "NOB": buggy_coeffs["Lambda"]["NOB"]["NO2"],
        "CMX": buggy_coeffs["Lambda"]["CMX"]["NH4"],
    }
    R_buggy, _ = reaction_and_jacobian(C, U, buggy_coeffs)

    assert not np.isclose(R_correct["NO2"][0], R_buggy["NO2"][0])
    assert not np.isclose(R_correct["NO3"][0], R_buggy["NO3"][0])


def test_reaction_terms_insensitive_to_rhat_entry():
    """Directly isolates bug 1: changing coeffs['rhat'] (which the reaction
    term must not use -- Lambda/LambdaProd already contain the full r_i)
    should not change R or dR at all."""
    coeffs = _synthetic_coeffs()
    C = {"NH4": np.array([0.5]), "NO2": np.array([0.3]),
         "NO3": np.array([0.2]), "O2": np.array([0.4])}
    U = {"AOB": np.array([1.0]), "NOB": np.array([1.0]), "CMX": np.array([1.0])}

    R1, dR1 = reaction_and_jacobian(C, U, coeffs)

    coeffs2 = _synthetic_coeffs()
    coeffs2["rhat"] = {"AOB": 99.0, "NOB": 0.001, "CMX": 42.0}
    R2, dR2 = reaction_and_jacobian(C, U, coeffs2)

    for sub in SUBSTRATES:
        assert np.allclose(R1[sub], R2[sub])
    for key in dR1:
        assert np.allclose(dR1[key], dR2[key])


if __name__ == "__main__":
    test_reaction_terms_match_independent_hand_computation()
    test_production_uses_produced_substrates_lambda_not_source_lambda()
    test_reaction_terms_insensitive_to_rhat_entry()
    print("All reaction-term regression tests passed.")
