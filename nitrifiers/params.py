"""Parameter presets for the 3-species / 4-substrate nitrification model (arXiv:2512.13156)."""

from __future__ import annotations
import json
from pathlib import Path

SPECIES = ("AOB", "NOB", "CMX")
SUBSTRATES = ("NH4", "NO2", "NO3", "O2")

A_OVER_D_RATIO = 10.0

BIOFILM_DENSITY_KG_M3 = 500.0
BIOMASS_MW_G_MOL = 24.6

BETA = {
    "AOB_to_NO2": 1.0,
    "NOB_to_NO3": 1.0,
    "CMX_to_NO3": 1.0,
}


def _day_from_hour(x_per_hour: float) -> float:
    return x_per_hour * 24.0


def _m2_per_day_from_m2_per_hour(x: float) -> float:
    return x * 24.0


def _m2_per_day_from_m2_per_sec(x: float) -> float:
    return x * 86400.0


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
    "domain": {"L": 500.0},
}

_DIFF_GENERIC = round(_m2_per_day_from_m2_per_sec(1e-9), 8)

REBECA = {
    "name": "rebeca",
    "description": "Rough estimate from R. Gonzalez-Cabaleiro's e-mail (xlsx sheet1, rows 3-20).",
    "species": {
        "AOB": {"r": 0.5, "b": 0.05,
                 "K": {"NH4": 1.0, "O2": 3.13},
                 "Y": {"NH4": 4.09 / 100, "O2": 6.135 / 100},
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

_D_NH4 = round(_m2_per_day_from_m2_per_hour(7.05e-6), 8)
_D_NO2 = round(_m2_per_day_from_m2_per_hour(6.88e-6), 8)
_D_NO3 = round(_m2_per_day_from_m2_per_hour(6.85e-6), 8)
_D_O2 = round(_m2_per_day_from_m2_per_hour(7.56e-6), 8)

ELOI = {
    "name": "eloi",
    "description": ("Martinez-Rabert thermodynamics-derived kinetics (T=25C, pH=7) "
                     "+ literature diffusion coefficients. NOB defaults to Nitrospira-type "
                     "kinetics; Nitrobacter-type kept under 'NOB_nitrobacter_alt'."),
    "species": {
        "AOB": {"r": _day_from_hour(0.03953884054665725), "b": 0.1 * _day_from_hour(0.03953884054665725),
                 "K": {"NH4": 1.453, "O2": 3.606},
                 "Y": {"NH4": 0.0409, "O2": 0.0409 / 1.5},
                 "d": 1.0e-6},
        "NOB": {"r": _day_from_hour(0.02095688325507988), "b": 0.1 * _day_from_hour(0.02095688325507988),
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
        "NH4": {"D": _D_NH4, "c_inf": 500.0},
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
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list_presets()}")
    return json.loads(json.dumps(PRESETS[name]))


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
