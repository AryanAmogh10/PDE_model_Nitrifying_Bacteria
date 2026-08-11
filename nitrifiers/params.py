"""
Parameter presets for the 3-species (AOB, NOB, CMX) / 4-substrate
(NH4+, NO2-, NO3-, O2) nitrification model of arXiv:2512.13156 (eq. 2.5).

Model structure (dimensional, before non-dimensionalization in Stage 2):

    d/dt u_i = div(d_i grad(u_i) + a_i u_i grad(rho)) + u_i f_i(u_i, c),   i in {AOB, NOB, CMX}
    d/dt c_j = D_j Delta(c_j) + c_j g_j(u, c),                             j in {NH4, NO2, NO3, O2}

    rho = u_AOB + u_NOB + u_CMX

    f_AOB(c) = r_AOB * M(c_NH4; K[AOB][NH4]) * M(c_O2; K[AOB][O2]) - b_AOB
    f_NOB(c) = r_NOB * M(c_NO2; K[NOB][NO2]) * M(c_O2; K[NOB][O2]) - b_NOB
    f_CMX(c) = r_CMX * M(c_NH4; K[CMX][NH4]) * M(c_O2; K[CMX][O2]) - b_CMX
    where M(c; K) = c / (K + c)   (Monod term)

    g_NH4 = -(1/Y[AOB][NH4]) u_AOB (f_AOB+b_AOB) - (1/Y[CMX][NH4]) u_CMX (f_CMX+b_CMX)   [/ c_NH4, see note]
    g_NO2 = +beta['AOB_to_NO2'] * (Y-scaled AOB production) - (1/Y[NOB][NO2]) u_NOB (f_NOB+b_NOB)
    g_NO3 = +beta['NOB_to_NO3'] * (...) + beta['CMX_to_NO3'] * (...)
    g_O2  = -(1/Y[AOB][O2]) u_AOB(...) - (1/Y[NOB][O2]) u_NOB(...) - (1/Y[CMX][O2]) u_CMX(...)

    (Exact factored-by-c_j form of g_j is assembled in Stage 2/3; only the
    parameter values are fixed here.)

Units used throughout this module (already cleaned/converted):
    r_i, b_i             : 1/day               (specific growth / decay rates)
    K[i][j]               : micromolar (uM)     (Monod half-saturation concentrations)
    Y[i][j]               : mol biomass-C / mol substrate  (dimensionless yield)
    D_j                    : m^2/day            (substrate molecular diffusivity)
    d_i                    : m^2/day            (bacterial "diffusion" coefficient)
    a_i                    : m^2/day            (cross-diffusion / advection coefficient "A")
    c_inf[j]               : micromolar (uM)     (bulk/feed reference concentration)
    beta[...]              : mol product / mol substrate consumed (dimensionless, molar N-conversion)

Sources:
    [Rebeca]  Rough parameter e-mail from R. Gonzalez-Cabaleiro (parameters_nitrifiers.xlsx, sheet 1).
    [Eloi]    E. Martinez-Rabert et al., thermodynamics-derived kinetics (parameters_nitrifiers.xlsx,
              sheet "Eloi Supplementary Table"), cross-checked against:
                - Martinez-Rabert et al. 2023, ISME Communications 3:91 (supplementary docx),
                  Table S4 (growth kinetics) and Table S2 (diffusion coefficients, ref. 24 = CRC Handbook).
                - Martinez-Rabert et al. 2021, supplementary docx (bit28045), Tables S1-S6
                  (literature ranges for mu_max, affinities, yields across many AOB/NOB/CMX strains).
    [arXiv]   Freingruber, Gonzalez-Cabaleiro, Yoldas, arXiv:2512.13156, Table 1 (order-of-magnitude
              values of d_i, a_i used in the paper's own non-dimensional toy simulations).

Cleaning decisions (flagged explicitly so they can be revisited):
    1. Rebeca's yield values in the xlsx (4.09, 2.42, 6.51, 6.135, 1.21, 13.02, all labelled
       "CmoleX/Nmole") are exactly 100x the corresponding Eloi/literature values (0.0409, 0.0242,
       0.0651, ...). This is almost certainly a units slip in the original e-mail (e.g. percent vs.
       fraction). We rescale Rebeca's yields by 1/100 so Y is a dimensionless mol/mol fraction < 1,
       consistent with Eloi's set and with literature (bit28045 Table S5/S6: Y_X/NH3 ~ 0.01-0.1).
    2. Rebeca gives no decay/maintenance rate b_i and no O2 half-saturation K_O2. We assume
       b_i = 0.1 * r_i (10% of max growth rate), following the assumption used in Martinez-Rabert
       2023 (ref. 40, Bodegom 2007), and borrow K_O2 = 3.13 uM (2023 docx Table S4, used for all
       three metabolisms there) for all three species in the Rebeca preset.
    3. Eloi's set gives two variants for NOB (Nitrobacter-type vs Nitrospira-type kinetics). We
       default to the Nitrospira-type NOB kinetics (r2alt, lower K_NO2, higher O2 affinity) because
       CMX in this model is comammox Nitrospira, so a Nitrospira-type NOB is the ecologically
       relevant direct competitor; the Nitrobacter-type numbers are kept alongside for reference.
    4. Eloi's set has no reported O2 yields; we derive them from the 1:1 stoichiometric reactions
       given in arXiv:2512.13156 (eq. between (2.5) and Fig. 2):
           AOB:  NH4+ + 1.5 O2 -> NO2- + 2H+ + H2O   =>  Y_O2,AOB = Y_NH4,AOB / 1.5
           NOB:  NO2-  + 0.5 O2 -> NO3-              =>  Y_O2,NOB = Y_NO2,NOB / 0.5
           CMX:  NH4+ + 2   O2 -> NO3- + 2H+ + H2O   =>  Y_O2,CMX = Y_NH4,CMX / 2
    5. Because all three reactions above convert exactly 1 mol of N-substrate into 1 mol of
       N-product, the metabolic conversion factors beta (mol product / mol substrate) are all 1
       for every preset -- this is a modelling simplification, not a fitted parameter.
    6. Eloi's set has no reported feed/bulk concentrations c_inf ("no info" in the sheet). We use
       representative values from the ammonia-feeding scenarios discussed in Martinez-Rabert 2023
       (NH4 = 500 uM, O2 = 93.8 uM, aerobic case); flagged as a literature-representative choice,
       not a fitted/measured value.
    7. Bacterial diffusion d_i and cross-diffusion a_i ("A") are not reported in any of the
       parameter sources (Rebeca e-mail, Eloi table, or the two supplementary docx files) --
       they describe individual-based/finite-volume substrate diffusion models, not this
       continuum cross-diffusion PDE. We anchor d_i to the order-of-magnitude value used in
       arXiv:2512.13156 Table 1 for the (non-dimensional) toy simulations (d_i ~ 1e-6, in the
       same units as D_j here, m^2/day). For a_i we only fix the *ratio* a_i/d_i = 10 (matching
       that paper's d_i << a_i assumption, Sec. 2.1); the dimensional value of a_i itself is
       derived in nondim.py, because dimensionally a_i must carry units of d_i / concentration
       (the flux terms d_i*grad(u_i) and a_i*u_i*grad(rho) must match: [a_i] = [d_i]/[rho]).
       Storing a_i in the same units as d_i, as an earlier version of this file did, was a
       units bug -- fixed by keeping only the target ratio A_OVER_D_RATIO here and deriving the
       dimensional a_i from it once a bacterial reference density u_ref is chosen (Stage 2).
       A_OVER_D_RATIO = 10 status: VERIFIED (not just ASSUMED) -- Table 1, Case (A) of
       arXiv:2512.13156, pulled directly from the paper's own PDF, gives d_i=1e-6, a_i=1e-5,
       i.e. a_i/d_i = 10 exactly (see report/audit_checklist.md, B8). This is still a borrowed
       constant (this project's a_i/d_i was not independently re-derived from first principles),
       but it now matches the paper's own stated value rather than being an unchecked guess.

    Labeling status of cleaning decisions above, per report/audit_checklist.md D6:
    note 1 (rebeca /100 yield rescale) -- ASSUMED, and shown (ITEM 3,
    report/item3_yield_sensitivity.py) to be qualitatively load-bearing: it determines whether
    Stage 6's rebeca run develops an anoxic core, while solver convergence, mass-growth
    direction, and dominant species are unaffected by the choice.
    note 7 (A_OVER_D_RATIO) -- VERIFIED, see above.
"""

from __future__ import annotations
import json
from pathlib import Path

SPECIES = ("AOB", "NOB", "CMX")
SUBSTRATES = ("NH4", "NO2", "NO3", "O2")

# Target ratio a_i/d_i (dimensionless), anchored to arXiv:2512.13156 Table 1 toy simulations
# (Sec. 2.1: "a natural assumption is d_i << a_i", i.e. bacterial spreading is
# advection/pressure-dominated, not diffusion-dominated). Not a fitted value -- see note 7.
A_OVER_D_RATIO = 10.0

# Packed biomass density and molecular weight, used in Stage 2 to build a physically grounded
# bacterial reference concentration u_ref. Source: Martinez-Rabert et al. 2023 (ISME Comm.),
# Table S4-equivalent: density of biomass (refs 26,27) and molecular weight of biomass
# CH1.8O0.5N0.2 (ref. 29).
BIOFILM_DENSITY_KG_M3 = 500.0
BIOMASS_MW_G_MOL = 24.6

# molar (1:1 N-conversion) stoichiometry -> conversion factors are all 1, see note 5 above.
BETA = {
    "AOB_to_NO2": 1.0,   # AOB: NH4+ + 1.5 O2 -> NO2- + 2H+ + H2O
    "NOB_to_NO3": 1.0,   # NOB: NO2- + 0.5 O2 -> NO3-
    "CMX_to_NO3": 1.0,   # CMX: NH4+ + 2 O2   -> NO3- + 2H+ + H2O
}


def _day_from_hour(x_per_hour: float) -> float:
    return x_per_hour * 24.0


def _m2_per_day_from_m2_per_hour(x: float) -> float:
    return x * 24.0


def _m2_per_day_from_m2_per_sec(x: float) -> float:
    return x * 86400.0


# ---------------------------------------------------------------------------
# Preset 1: "toy" -- clean round numbers for fast sanity checks / unit tests.
# ---------------------------------------------------------------------------
TOY = {
    "name": "toy",
    "description": "Synthetic round-number preset for solver development and unit tests.",
    "species": {
        "AOB": {"r": 1.0, "b": 0.1,
                 "K": {"NH4": 1.0, "O2": 1.0},
                 "Y": {"NH4": 0.1, "O2": 0.1},
                 "d": 1.0e-6},
        "NOB": {"r": 1.0, "b": 0.1,
                 "K": {"NO2": 1.0, "O2": 1.0},
                 "Y": {"NO2": 0.1, "O2": 0.1},
                 "d": 1.0e-6},
        "CMX": {"r": 1.0, "b": 0.1,
                 "K": {"NH4": 1.0, "O2": 1.0},
                 "Y": {"NH4": 0.1, "O2": 0.1},
                 "d": 1.0e-6},
    },
    "substrates": {
        "NH4": {"D": 1.0e-4, "c_inf": 500.0},
        "NO2": {"D": 1.0e-4, "c_inf": 0.0},
        "NO3": {"D": 1.0e-4, "c_inf": 0.0},
        "O2":  {"D": 1.0e-4, "c_inf": 200.0},
    },
    "beta": dict(BETA),
    "domain": {"L": 500.0},  # reference length scale, micrometres (biofilm/granule radius)
}

# ---------------------------------------------------------------------------
# Preset 2: "rebeca" -- rough e-mail estimate (parameters_nitrifiers.xlsx, sheet1 top block).
# ---------------------------------------------------------------------------
_DIFF_GENERIC = round(_m2_per_day_from_m2_per_sec(1e-9), 8)  # 8.64e-5 m^2/day, all substrates

REBECA = {
    "name": "rebeca",
    "description": "Rough estimate from R. Gonzalez-Cabaleiro's e-mail (xlsx sheet1, rows 3-20).",
    "species": {
        "AOB": {"r": 0.5, "b": 0.05,      # b assumed = 0.1*r, see cleaning note 2
                 "K": {"NH4": 1.0, "O2": 3.13},   # K_O2 borrowed from Eloi/2023-docx set, note 2
                 "Y": {"NH4": 4.09 / 100, "O2": 6.135 / 100},   # /100 unit fix, note 1
                 "d": 1.0e-6},
        "NOB": {"r": 1.5, "b": 0.15,
                 "K": {"NO2": 1.0, "O2": 3.13},
                 "Y": {"NO2": 2.42 / 100, "O2": 1.21 / 100},
                 "d": 1.0e-6},
        "CMX": {"r": 0.375, "b": 0.0375,
                 "K": {"NH4": 1.0, "O2": 3.13},
                 "Y": {"NH4": 6.51 / 100, "O2": 13.02 / 100},
                 "d": 1.0e-6},
    },
    "substrates": {
        "NH4": {"D": _DIFF_GENERIC, "c_inf": 554.0},
        "NO2": {"D": _DIFF_GENERIC, "c_inf": 0.0},
        "NO3": {"D": _DIFF_GENERIC, "c_inf": 0.0},
        "O2":  {"D": _DIFF_GENERIC, "c_inf": 187.5},
    },
    "beta": dict(BETA),
    "domain": {"L": 500.0},
}

# ---------------------------------------------------------------------------
# Preset 3: "eloi" -- Martinez-Rabert thermodynamics/literature-derived set.
# ---------------------------------------------------------------------------
# Diffusion coefficients from Martinez-Rabert 2023 (ISME Comm.) Table S4-equivalent, ref. 24
# (CRC Handbook), reported in m^2/h -> converted to m^2/day.
_D_NH4 = round(_m2_per_day_from_m2_per_hour(7.05e-6), 8)
_D_NO2 = round(_m2_per_day_from_m2_per_hour(6.88e-6), 8)
_D_NO3 = round(_m2_per_day_from_m2_per_hour(6.85e-6), 8)
_D_O2 = round(_m2_per_day_from_m2_per_hour(7.56e-6), 8)

ELOI = {
    "name": "eloi",
    "description": ("Martinez-Rabert thermodynamics-derived kinetics (T=25C, pH=7) "
                     "+ literature diffusion coefficients. NOB defaults to Nitrospira-type "
                     "kinetics (see note 3); Nitrobacter-type kept under 'NOB_nitrobacter_alt'."),
    "species": {
        "AOB": {"r": _day_from_hour(0.03953884054665725), "b": 0.1 * _day_from_hour(0.03953884054665725),
                 "K": {"NH4": 1.453, "O2": 3.606},
                 "Y": {"NH4": 0.0409, "O2": 0.0409 / 1.5},   # Y_O2 derived, note 4
                 "d": 1.0e-6},
        "NOB": {"r": _day_from_hour(0.02095688325507988), "b": 0.1 * _day_from_hour(0.02095688325507988),
                 # K_NO2 = 8.894e-5 uM (~89 pM) is a genuine literature value for
                 # Nitrospira's nitrite affinity (traced correctly to the xlsx/bit28045
                 # source), but it is an extraordinarily high affinity -- the Monod
                 # term saturates to ~1 at essentially any positive NO2 and its
                 # derivative K/(K+c)^2 vanishes away from c~0. Flagged as a likely
                 # contributor to elliptic-solver stiffness at "realistic" density;
                 # worth confirming with Viktoria/Rebeca whether this literal value or
                 # a more typical/rounded K is intended for numerical development.
                 "K": {"NO2": 8.894e-5, "O2": 2.594},
                 "Y": {"NO2": 0.0242, "O2": 0.0242 / 0.5},
                 "d": 1.0e-6},
        "CMX": {"r": _day_from_hour(0.002884638212810753), "b": 0.1 * _day_from_hour(0.002884638212810753),
                 "K": {"NH4": 0.063, "O2": 3.13},
                 "Y": {"NH4": 0.0651, "O2": 0.0651 / 2.0},
                 "d": 1.0e-6},
        "NOB_nitrobacter_alt": {
                 "r": _day_from_hour(0.03327322922275781), "b": 0.1 * _day_from_hour(0.03327322922275781),
                 "K": {"NO2": 0.004447, "O2": 7.813},
                 "Y": {"NO2": 0.0242, "O2": 0.0242 / 0.5},
                 "d": 1.0e-6},
    },
    "substrates": {
        "NH4": {"D": _D_NH4, "c_inf": 500.0},   # c_inf: literature-representative, note 6
        "NO2": {"D": _D_NO2, "c_inf": 0.0},
        "NO3": {"D": _D_NO3, "c_inf": 0.0},
        "O2":  {"D": _D_O2, "c_inf": 93.8},
    },
    "beta": dict(BETA),
    "domain": {"L": 500.0},
}

PRESETS = {"toy": TOY, "rebeca": REBECA, "eloi": ELOI}


def list_presets() -> list[str]:
    return list(PRESETS.keys())


def load_preset(name: str) -> dict:
    """Return a deep-enough copy of the named preset ('toy', 'rebeca', or 'eloi')."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list_presets()}")
    return json.loads(json.dumps(PRESETS[name]))  # cheap deep copy, all-JSON-safe values


def export_json(path: str | Path = "parameters_nitrifiers_clean.json") -> Path:
    path = Path(path)
    path.write_text(json.dumps(PRESETS, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = export_json()
    print(f"Wrote cleaned parameter presets to {out.resolve()}")
    for name in list_presets():
        print(f"\n=== {name} ===")
        print(json.dumps(load_preset(name), indent=2))
