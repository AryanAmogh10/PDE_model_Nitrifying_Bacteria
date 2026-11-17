"""Sensitivity runs for the AOB-CMX study (combined system, 100x100 nodes, 50 days = 47.5 solver units).
Variants: outer (CMX starts in the outer ring of the initial disc, AOB and NOB inside), b05 / b2 (death
coefficients times 0.5 / 2), k13 (CMX half-saturation for ammonium times 13.3, free ammonia -> total ammonium),
and grid200 (200x200 nodes, base case, one condition only).
    python run_sensitivity.py            -> sensitivity.json"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM)); sys.path.insert(0, str(SIM.parent))
import common as c
from common import SPECIES

PRESET = "eloi"
NH4 = np.round(np.geomspace(0.5, 1000.0, 9), 3)
O2 = np.array([5.0, 15.0, 40.0, 70.0, 100.0, 175.0, 250.0])
CASES = [(4, 6), (6, 6), (6, 2), (8, 6), (2, 4)]       # (NH4 index, O2 index)
T_END = 47.5

_orig_make, _orig_init = c.make_coeffs, c.initial_biomass


def outer_init(grid, system):
    X, Y, circles = c.circle_layout(grid.Nx + 1)
    support = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.2 ** 2
    inner = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.12 ** 2
    pops = {s: np.zeros_like(X) for s in SPECIES}
    for cx, cy, k in circles:
        pops[SPECIES[0 if k == 2 else k]] += ((X - cx) ** 2 + (Y - cy) ** 2 <= 0.05 ** 2).astype(float)
    tot = pops["AOB"] + pops["NOB"]; tot[tot == 0] = 1.0
    U = {"AOB": np.where(inner, pops["AOB"] / tot, 0.0), "NOB": np.where(inner, pops["NOB"] / tot, 0.0),
         "CMX": np.where(support & ~inner, 1.0, 0.0)}
    U = {s: v.ravel() for s, v in U.items()}
    mass = {s: float(np.sum(grid.Vflat * U[s])) for s in SPECIES}
    M0 = float(np.mean(list(mass.values())))
    return {s: U[s] * M0 / mass[s] for s in SPECIES}


def run(job):
    variant, i, j, N = job
    c.make_coeffs, c.initial_biomass = _orig_make, _orig_init
    if variant == "outer":
        c.initial_biomass = outer_init
    elif variant in ("b05", "b2", "k13"):
        def mk(preset, v=variant):
            co = _orig_make(preset)
            if v == "b05":
                co["bhat"] = {s: x * 0.5 for s, x in co["bhat"].items()}
            elif v == "b2":
                co["bhat"] = {s: x * 2.0 for s, x in co["bhat"].items()}
            else:
                co["Khat"]["CMX"]["NH4"] *= 0.84 / 0.063
            return co
        c.make_coeffs = mk
    u = c.Units(PRESET)
    h, info = c.run(PRESET, "all", NH4[i] / u.c_ref_uM, O2[j] / u.c_ref_uM, N=N, T_max=T_END, dt=0.5)
    M = {s: [float(u.biomass_ug_per_mm(m)) for m in h["M"][s]] for s in SPECIES}
    return {"variant": variant, "i": i, "j": j, "N": N, "NH4": float(NH4[i]), "O2": float(O2[j]), "t_day": [float(u.days(t)) for t in h["t"]],
            "M": M, "touch": info["boundary_touch_t"]}


if __name__ == "__main__":
    jobs = [(v, i, j, 99) for (i, j) in CASES for v in ("base", "outer", "b05", "b2", "k13")] + [("base", 6, 6, 199)]
    jobs.sort(key=lambda x: -x[3])
    out = []
    with Pool(6) as pool:
        for r in pool.imap_unordered(run, jobs):
            out.append(r); print("done", r["variant"], r["NH4"], r["O2"], r["N"], flush=True)
            (HERE / "sensitivity.json").write_text(json.dumps(out))
