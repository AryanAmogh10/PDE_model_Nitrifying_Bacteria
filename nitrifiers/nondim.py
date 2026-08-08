"""
Non-dimensionalization of the 3-species / 4-substrate nitrification model
(arXiv:2512.13156, eq. 2.5) using the cleaned parameter presets from params.py.

-----------------------------------------------------------------------------
Reference scales
-----------------------------------------------------------------------------
    L      reference length      = preset['domain']['L'] (micrometres -> metres)
                                    the granule/biofilm radius.
    c_ref  reference substrate   = c_inf of NH4 (ammonia feed concentration).
           concentration           Ammonia is the substrate every species in this
                                    system either consumes directly (AOB, CMX) or
                                    is downstream of (NOB via NO2), so it is the
                                    natural single reference concentration for all
                                    four substrates (NH4, NO2, NO3, O2 all measured
                                    in units of c_ref).
    u_ref  reference bacterial   = physically-packed biomass density BIOFILM_DENSITY_KG_M3,
           density                 converted to a molar (uM) concentration via
                                    BIOMASS_MW_G_MOL. This is deliberately the *packed*
                                    biofilm density (an upper-bound physical scale), not a
                                    yield-derived guess (u_ref = Y*c_ref would be ~1e4-1e5
                                    times smaller and would make the substrate reaction
                                    term negligible compared to diffusion, which contradicts
                                    the expected rim-to-core substrate zonation). Using the
                                    packed biomass density gives Damkohler numbers Da_ij
                                    (below) that are O(1)-to-large, consistent with a
                                    genuine reaction-diffusion balance in the elliptic
                                    substrate problem, matching the modelling intent
                                    described in SlowFast_Nitrifiers.pdf.
    tau_slow = 1 / r_max          reference slow time, r_max = max_i(r_i) over AOB/NOB/CMX
                                    (using the default species only, not the
                                    NOB_nitrobacter_alt variant).
    tau_fast = L^2 / D_ref        reference substrate-diffusion time, D_ref = D_NH4.

-----------------------------------------------------------------------------
The slow-fast parameter epsilon
-----------------------------------------------------------------------------
We report two dimensionless numbers, both of which must be small for the
quasi-steady-state reduction in SlowFast_Nitrifiers.pdf to be justified:

    eps_j := tau_fast_j / tau_slow = r_max * L^2 / D_j       (one per substrate j)

        This is the ratio of "how long it takes substrate j to diffuse across
        the domain" to "how long it takes the bacteria to double". It is the
        actual slow-fast time-scale-separation parameter of the PDE (the PDF's
        epsilon): small eps_j means substrate profiles equilibrate long before
        the bacterial population changes appreciably.

    delta_i := d_i / D_ref                                    (per species i)

        The ratio of bacterial "diffusion" to substrate diffusion. This is a
        second, independent confirmation of scale separation (bacteria barely
        move by diffusion at all compared to how fast substrate spreads);
        delta_i << eps_j is expected and checked below.

We take eps := max_j(eps_j) as the controlling (worst-case) value: the
quasi-steady approximation is only as good as its least-separated substrate.

-----------------------------------------------------------------------------
Dimensionless groups and rewritten PDEs
-----------------------------------------------------------------------------
With x = L*xhat, t = t_hat/r_max, u_i = u_ref*uhat_i, c_j = c_ref*chat_j,
rhohat = uhat_AOB + uhat_NOB + uhat_CMX:

    Species PDE (i in AOB, NOB, CMX; using substrate 1 = NH4/NO2 (species-specific)
    and substrate 4 = O2):

        d(uhat_i)/d(that) = Dhat_i * Lap(uhat_i)
                             + Ahat_i * div(uhat_i * grad(rhohat))
                             + uhat_i * [ rhat_i * M(chat_p; Khat_ip) * M(chat_O2; Khat_iO2) - bhat_i ]

        where p = NH4 for AOB/CMX, NO2 for NOB, and M(c;K) := c/(K+c).

    Substrate PDE (j in NH4, NO2, NO3, O2):

        eps_j * d(chat_j)/d(that) = Lap(chat_j) + eps_j * chat_j * ghat_j(uhat, chat)

        i.e. after non-dimensionalizing with tau_slow, diffusion appears at
        order 1/eps_j (fast) while both the substrate time-derivative and the
        reaction term appear at order eps_j -- consistent with eps_j -> 0
        giving the leading-order quasi-steady elliptic problem
        0 = Lap(chat_j) + chat_j * ghat_j(uhat, chat) [after dividing by eps_j],
        matching SlowFast_Nitrifiers.pdf eq. (6).

    Dimensionless numbers:
        Dhat_i  = d_i / (r_max * L^2)                 bacterial diffusion number
        Ahat_i  = A_OVER_D_RATIO * Dhat_i              cross-diffusion number (see params.py note 7)
        rhat_i  = r_i / r_max,   bhat_i = b_i / r_max   growth/decay ratios
        Khat_ij = K_ij / c_ref                          Monod half-saturation ratios
        Da_ij   = u_ref / (Y_ij * c_ref)                Damkohler number for substrate j
                                                         consumption/production by species i
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .params import (
    load_preset, list_presets, SPECIES, SUBSTRATES,
    A_OVER_D_RATIO, BIOFILM_DENSITY_KG_M3, BIOMASS_MW_G_MOL,
)

# species -> substrates it actually consumes (Monod-limiting), (primary, secondary).
# This is the single source of truth for PRIMARY/SECONDARY: elliptic.py imports
# them from here rather than hardcoding a second copy (that duplication -- found
# during an audit, alongside an identical one between elliptic.py and
# parabolic.py -- was the same "same fact defined twice" pattern that caused the
# Neumann-target and volume-normalisation bugs elsewhere in this project).
CONSUMED_SUBSTRATES = {"AOB": ("NH4", "O2"), "NOB": ("NO2", "O2"), "CMX": ("NH4", "O2")}
PRIMARY = {name: pair[0] for name, pair in CONSUMED_SUBSTRATES.items()}
SECONDARY = {name: pair[1] for name, pair in CONSUMED_SUBSTRATES.items()}


def _u_ref_uM() -> float:
    """Packed biomass density converted to a molar (uM) concentration scale."""
    mol_per_m3 = (BIOFILM_DENSITY_KG_M3 * 1000.0) / BIOMASS_MW_G_MOL  # kg/m3 -> g/m3 -> mol/m3
    # 1 mol/m3 = 1 mmol/L = 1000 uM
    return mol_per_m3 * 1000.0


@dataclass
class NondimResult:
    preset_name: str
    L_m: float
    c_ref_uM: float
    u_ref_uM: float
    r_max_per_day: float
    tau_slow_day: float
    D_ref_m2_day: float
    tau_fast_day: float
    eps_j: dict = field(default_factory=dict)
    eps: float = 0.0
    delta_i: dict = field(default_factory=dict)
    species: dict = field(default_factory=dict)   # per-species dimensionless numbers


def nondimensionalize(preset_name: str) -> NondimResult:
    p = load_preset(preset_name)
    species = {k: v for k, v in p["species"].items() if k in SPECIES}

    L_m = p["domain"]["L"] * 1e-6  # micrometres -> metres
    c_ref = p["substrates"]["NH4"]["c_inf"]
    if c_ref <= 0:
        raise ValueError(f"{preset_name}: NH4 c_inf must be > 0 to use as c_ref")
    u_ref = _u_ref_uM()

    r_max = max(spec["r"] for spec in species.values())
    tau_slow = 1.0 / r_max

    D_ref = p["substrates"]["NH4"]["D"]
    tau_fast = L_m ** 2 / D_ref

    eps_j = {j: r_max * L_m ** 2 / p["substrates"][j]["D"] for j in SUBSTRATES}
    eps = max(eps_j.values())
    delta_i = {i: spec["d"] / D_ref for i, spec in species.items()}

    species_out = {}
    for name, spec in species.items():
        primary = PRIMARY[name]
        Dhat = spec["d"] / (r_max * L_m ** 2)
        Ahat = A_OVER_D_RATIO * Dhat
        Khat = {sub: k / c_ref for sub, k in spec["K"].items()}
        Da = {sub: u_ref / (y * c_ref) for sub, y in spec["Y"].items()}
        species_out[name] = {
            "primary_substrate": primary,
            "rhat": spec["r"] / r_max,
            "bhat": spec["b"] / r_max,
            "Dhat": Dhat,
            "Ahat": Ahat,
            "Khat": Khat,
            "Da": Da,
        }

    return NondimResult(
        preset_name=preset_name, L_m=L_m, c_ref_uM=c_ref, u_ref_uM=u_ref,
        r_max_per_day=r_max, tau_slow_day=tau_slow, D_ref_m2_day=D_ref,
        tau_fast_day=tau_fast, eps_j=eps_j, eps=eps, delta_i=delta_i,
        species=species_out,
    )


# which consumed substrate produces which downstream substrate, and the (molar) beta key
PRODUCTION = {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")}


def elliptic_coefficients(preset_name: str) -> dict:
    """
    Assemble the dimensionless coefficients needed to write down the leading-order
    (epsilon -> 0) quasi-steady elliptic substrate problem

        0 = Lap(chat_j) + Rhat_j(uhat, chat),    j in NH4, NO2, NO3, O2

    where, writing M(c;K) := c/(K+c) and Uptake_i(chat) := M(chat_p;Khat_ip) * M(chat_O2;Khat_iO2)
    (NOTE: no rhat_i factor here -- Lambda[i][j] below already contains the full,
    un-normalised r_i; multiplying by rhat_i again was a bug caught in review,
    fixed together with the point below) for species i with primary substrate p
    (NH4 for AOB/CMX, NO2 for NOB):

        Rhat_NH4 = -Lambda[AOB][NH4] * uhat_AOB * Uptake_AOB - Lambda[CMX][NH4] * uhat_CMX * Uptake_CMX
        Rhat_NO2 = +beta['AOB_to_NO2'] * LambdaProd[AOB] * uhat_AOB * Uptake_AOB
                   -Lambda[NOB][NO2] * uhat_NOB * Uptake_NOB
        Rhat_NO3 = +beta['NOB_to_NO3'] * LambdaProd[NOB] * uhat_NOB * Uptake_NOB
                   +beta['CMX_to_NO3'] * LambdaProd[CMX] * uhat_CMX * Uptake_CMX
        Rhat_O2  = -Lambda[AOB][O2] * uhat_AOB * Uptake_AOB
                   -Lambda[NOB][O2] * uhat_NOB * Uptake_NOB
                   -Lambda[CMX][O2] * uhat_CMX * Uptake_CMX

    Note: the production terms reuse the species' consumption-side reaction rate
    (r_i, Y_ij of the SOURCE substrate) -- not a separately fitted beta*raw-growth
    term as in the literal arXiv eq. 2.5 -- to enforce exact 1:1 molar N mass
    balance (beta == 1 for all three conversions here, see params.py note 5).
    However, since the production term sits in the *produced* substrate's own
    equation (derived by dividing through by D_(produced) c_ref/L^2, not
    D_(source) c_ref/L^2), its dimensionless coefficient LambdaProd[i] must use
    the produced substrate's diffusivity, not the source substrate's -- reusing
    Lambda[i][source] directly (which carries D_source) was a second bug caught
    in review, fixed here by defining LambdaProd separately:

    Lambda[i][j]     := L^2 * u_ref * r_i / (D_j * c_ref * Y_ij)              j = a substrate species i CONSUMES
    LambdaProd[i]     := L^2 * u_ref * r_i / (D_prod * c_ref * Y_i,source)     D_prod = diffusivity of the PRODUCED substrate,
                                                                                Y_i,source = yield on the species' own source substrate
    (both reduce to rhat_i * eps_j * Da_ij for the respective D_j, see module docstring)
    """
    r = nondimensionalize(preset_name)
    p = load_preset(preset_name)
    species = {k: v for k, v in p["species"].items() if k in SPECIES}
    c_ref = r.c_ref_uM

    Lambda = {}
    LambdaProd = {}
    Khat = {}
    rhat = {}
    bhat = {}
    Dhat = {}
    Ahat = {}
    for name, spec in species.items():
        Lambda[name] = {}
        for sub in CONSUMED_SUBSTRATES[name]:
            D_j = p["substrates"][sub]["D"]
            Y_ij = spec["Y"][sub]
            Lambda[name][sub] = (r.L_m ** 2 * r.u_ref_uM * spec["r"]) / (D_j * c_ref * Y_ij)
        source_sub = PRIMARY[name]
        produced_sub, _ = PRODUCTION[name]
        D_prod = p["substrates"][produced_sub]["D"]
        Y_source = spec["Y"][source_sub]
        LambdaProd[name] = (r.L_m ** 2 * r.u_ref_uM * spec["r"]) / (D_prod * c_ref * Y_source)
        Khat[name] = {sub: spec["K"][sub] / c_ref for sub in CONSUMED_SUBSTRATES[name]}
        rhat[name] = spec["r"] / r.r_max_per_day
        bhat[name] = spec["b"] / r.r_max_per_day
        Dhat[name] = r.species[name]["Dhat"]
        Ahat[name] = r.species[name]["Ahat"]

    c_inf_hat = {sub: p["substrates"][sub]["c_inf"] / c_ref for sub in SUBSTRATES}

    return {
        "Lambda": Lambda,
        "LambdaProd": LambdaProd,
        "Khat": Khat,
        "rhat": rhat,
        "bhat": bhat,
        "Dhat": Dhat,
        "Ahat": Ahat,
        "c_inf_hat": c_inf_hat,
        "beta": dict(p["beta"]),
        "consumed_substrates": CONSUMED_SUBSTRATES,
        "production": PRODUCTION,
        "eps": r.eps,
    }


def _fmt(x: float) -> str:
    return f"{x:.4e}"


def report(preset_name: str) -> str:
    r = nondimensionalize(preset_name)
    lines = [f"=== {preset_name} ===",
             f"L = {r.L_m:.3e} m, c_ref (NH4 feed) = {r.c_ref_uM:g} uM, "
             f"u_ref (packed biomass) = {r.u_ref_uM:.3e} uM",
             f"r_max = {r.r_max_per_day:.4g} 1/day -> tau_slow = {r.tau_slow_day:.4g} day",
             f"D_ref (D_NH4) = {r.D_ref_m2_day:.3e} m^2/day -> tau_fast = {r.tau_fast_day:.3e} day",
             f"eps_j per substrate: " + ", ".join(f"{j}={r.eps_j[j]:.3e}" for j in SUBSTRATES),
             f"eps (= max_j eps_j) = {r.eps:.3e}  {'[OK: eps << 1]' if r.eps < 0.1 else '[WARNING: not << 1]'}",
             f"delta_i (d_i/D_ref) per species: " + ", ".join(f"{i}={r.delta_i[i]:.3e}" for i in r.delta_i),
             ""]
    for name, s in r.species.items():
        lines.append(f"  {name}: rhat={s['rhat']:.3f}, bhat={s['bhat']:.3f}, "
                      f"Dhat={s['Dhat']:.3e}, Ahat={s['Ahat']:.3e}")
        lines.append(f"        Khat={ {k: round(v,4) for k,v in s['Khat'].items()} }")
        lines.append(f"        Da  ={ {k: _fmt(v) for k,v in s['Da'].items()} }")
    return "\n".join(lines)


if __name__ == "__main__":
    for name in list_presets():
        print(report(name))
        print()
