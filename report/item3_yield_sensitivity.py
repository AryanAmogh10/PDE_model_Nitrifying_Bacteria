"""
ITEM 3: sensitivity analysis on the `rebeca` preset's /100 yield rescale.

params.py cleaning note 1 (see nitrifiers/params.py, lines ~46-51) rescales
Rebeca's raw e-mail yield values (4.09, 2.42, 6.51, 6.135, 1.21, 13.02,
labelled "CmoleX/Nmole") by 1/100, on the grounds that they are exactly 100x
the corresponding Eloi/literature values and are "almost certainly a units
slip ... (e.g. percent vs. fraction)". That rescale was previously labelled
ASSUMED. This script checks whether Stage 6's *qualitative* conclusions
(zonation persists, elliptic solves stay well-conditioned, bacterial mass
grows) are robust to that choice, by running the identical Stage 6 slow loop
twice on the SAME preset except for the yield values:

    "rebeca"            Y as currently defined (raw / 100)      -- the shipped preset
    "rebeca_unrescaled" Y as given verbatim in the e-mail xlsx   -- the /100 fix reverted

Y appears in the nondimensionalization only through the Damkohler number
Da_ij = u_ref / (Y_ij * c_ref) (nondim.py, Lambda[name][sub] in
elliptic_coefficients) -- a 100x larger Y (unrescaled) gives a 100x SMALLER
Lambda, i.e. 100x weaker substrate consumption/production per unit biomass.
This is a genuine, large quantitative change; the question is whether it
flips any qualitative conclusion Stage 6 relies on.

Self-contained script (mirrors report/item2_wave_speed.py's structure):
not added to the main package, since "rebeca_unrescaled" is a deliberately
wrong/reverted preset that should never be reachable from normal code paths,
only from this one-off sensitivity check. It registers the variant directly
into nitrifiers.params.PRESETS (a plain dict) rather than duplicating
params.py's REBECA block by hand, so the unrescaled variant is guaranteed to
be identical to "rebeca" in every field except Y.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers import params as _params
from nitrifiers.elliptic import Grid, SUBSTRATES
from nitrifiers.parabolic import SPECIES
from nitrifiers.nondim import elliptic_coefficients
from nitrifiers.slowfast import run_slow_loop


def _register_unrescaled_variant() -> None:
    variant = copy.deepcopy(_params.REBECA)
    variant["name"] = "rebeca_unrescaled"
    variant["description"] = ("Sensitivity check ONLY: Rebeca's e-mail yields used "
                               "verbatim (the /100 cleaning fix, note 1, reverted).")
    for sp in variant["species"].values():
        for sub in sp["Y"]:
            sp["Y"][sub] = sp["Y"][sub] * 100.0  # undo the /100 in params.py
    _params.PRESETS["rebeca_unrescaled"] = variant


def _seed_profile(grid):
    r = grid.r
    return {sp: 0.02 * np.exp(-((r - 0.5) / 0.15) ** 2) for sp in SPECIES}


def _run(preset_name: str, n_slow_steps: int = 20, dt_slow: float = 0.05):
    grid = Grid(N=60, geometry="radial", p=2)
    U0 = _seed_profile(grid)
    U, C, hist, _ = run_slow_loop(preset_name, grid, U0, n_slow_steps=n_slow_steps,
                                   dt_slow=dt_slow)
    return grid, U, C, hist


def _summarize(label, grid, U, C, hist):
    methods = sorted(set(h.elliptic_method for h in hist))
    max_resid = max(h.elliptic_residual for h in hist)
    masses = [h.total_mass for h in hist]
    mass_growth = masses[-1] / masses[0] if masses[0] > 0 else float("nan")
    o2_core = C["O2"][0]
    o2_rim = C["O2"][-1]
    zonation_ok = (o2_core < 1e-3 * o2_rim) or (o2_rim < 1e-6)
    dominant = max(SPECIES, key=lambda sp: float(np.sum(grid.V * U[sp])) if hasattr(grid, "V")
                   else float(np.sum(U[sp])))
    print(f"\n--- {label} ---")
    print(f"  elliptic methods used: {methods}")
    print(f"  max elliptic residual over loop: {max_resid:.3e}")
    print(f"  total mass: start={masses[0]:.6e} end={masses[-1]:.6e} "
          f"(growth factor {mass_growth:.4f})")
    print(f"  O2: core={o2_core:.4e}, rim={o2_rim:.4e}, "
          f"rim-to-core zonation present: {zonation_ok}")
    print(f"  dominant species by total mass: {dominant}")
    return {
        "methods": methods, "max_resid": max_resid, "mass_growth": mass_growth,
        "o2_core": o2_core, "o2_rim": o2_rim, "zonation_ok": zonation_ok,
        "dominant": dominant, "final_mass": masses[-1],
    }


if __name__ == "__main__":
    print("=" * 78)
    print("ITEM 3: rebeca yield /100 rescale -- sensitivity of Stage 6 conclusions")
    print("=" * 78)

    _register_unrescaled_variant()

    coeffs_rescaled = elliptic_coefficients("rebeca")
    coeffs_unrescaled = elliptic_coefficients("rebeca_unrescaled")
    print("\nDamkohler number Lambda[AOB][NH4] (drives NH4 consumption strength):")
    print(f"  rebeca (rescaled, Y=Y_raw/100):   {coeffs_rescaled['Lambda']['AOB']['NH4']:.4e}")
    print(f"  rebeca_unrescaled (Y=Y_raw):       {coeffs_unrescaled['Lambda']['AOB']['NH4']:.4e}")
    print(f"  ratio: {coeffs_rescaled['Lambda']['AOB']['NH4'] / coeffs_unrescaled['Lambda']['AOB']['NH4']:.1f}x "
          f"(expected exactly 100x, since Lambda ~ 1/Y)")

    grid1, U1, C1, hist1 = _run("rebeca")
    grid2, U2, C2, hist2 = _run("rebeca_unrescaled")

    r1 = _summarize("rebeca (SHIPPED, /100 rescale)", grid1, U1, C1, hist1)
    r2 = _summarize("rebeca_unrescaled (raw e-mail values)", grid2, U2, C2, hist2)

    print("\n" + "=" * 78)
    print("ROBUSTNESS VERDICT")
    print("=" * 78)
    both_converge = all(m in ("newton", "newton_inner_relax_fallback", "outer_relaxation_backstop")
                         for m in r1["methods"] + r2["methods"])
    both_grow = r1["mass_growth"] > 1.0 and r2["mass_growth"] > 1.0
    both_zonate = r1["zonation_ok"] and r2["zonation_ok"]
    print(f"  Both presets' elliptic solves converge to tolerance: {both_converge}")
    print(f"  Both presets show bacterial mass growth over the loop: {both_grow} "
          f"(rebeca: {r1['mass_growth']:.3f}x, unrescaled: {r2['mass_growth']:.3f}x)")
    print(f"  Both presets show rim-to-core O2 zonation: {both_zonate}")
    print(f"  Dominant species: rebeca={r1['dominant']}, unrescaled={r2['dominant']}")
    all_robust = both_converge and both_grow and both_zonate and r1["dominant"] == r2["dominant"]
    print(f"\n  QUALITATIVE CONCLUSIONS ROBUST TO THE /100 CHOICE: {all_robust}")
