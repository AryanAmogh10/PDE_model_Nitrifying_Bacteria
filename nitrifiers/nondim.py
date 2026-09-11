"""Non-dimensionalization of the nitrification model (arXiv:2512.13156, eq. 2.5)."""

from __future__ import annotations
from dataclasses import dataclass, field

from .params import (
    load_preset, list_presets, SPECIES, SUBSTRATES,
    A_OVER_D_RATIO, BIOFILM_DENSITY_KG_M3, BIOMASS_MW_G_MOL,
)

CONSUMED_SUBSTRATES = {"AOB": ("NH4", "O2"), "NOB": ("NO2", "O2"), "CMX": ("NH4", "O2")}
PRIMARY = {name: pair[0] for name, pair in CONSUMED_SUBSTRATES.items()}
SECONDARY = {name: pair[1] for name, pair in CONSUMED_SUBSTRATES.items()}


def _u_ref_uM() -> float:
    mol_per_m3 = (BIOFILM_DENSITY_KG_M3 * 1000.0) / BIOMASS_MW_G_MOL
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
    species: dict = field(default_factory=dict)


def nondimensionalize(preset_name: str) -> NondimResult:
    p = load_preset(preset_name)
    species = {k: v for k, v in p["species"].items() if k in SPECIES}

    L_m = p["domain"]["L"] * 1e-6
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


PRODUCTION = {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")}


def elliptic_coefficients(preset_name: str) -> dict:
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
