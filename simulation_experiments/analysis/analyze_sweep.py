"""Derived quantities of the AOB-CMX sweep from the stored results (no new simulations):
mass time series, per-species limiting factors, uptake shares and profiles at the 47-day snapshot.
Writes sweep_derived.pkl next to this file. Run before render_results.py."""
import sys, json, pickle
from pathlib import Path
import numpy as np

CODES = Path(__file__).resolve().parents[2]
SIM = CODES / "simulation_experiments"
sys.path.insert(0, str(CODES))
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "heatmap"))
import common as c
from common import SPECIES, SUBSTRATES
from nitrifiers.nondim import elliptic_coefficients, PRIMARY, SECONDARY
from nitrifiers.one_dimensional.elliptic import monod
from nitrifiers.two_dimensional.grid2d import Grid2D

OUT = Path(__file__).resolve().parent / "sweep_derived.pkl"
RES_ALL = SIM / "heatmap" / "results_all_N99_T150"
RES_CMX = SIM / "heatmap" / "results_cmx_N99_T150"
NH4 = np.round(np.geomspace(0.5, 1000.0, 9), 3)
O2 = np.array([5.0, 15.0, 40.0, 70.0, 100.0, 175.0, 250.0])
NI, NJ = len(NH4), len(O2)
u = c.Units("eloi")
coeffs = c.make_coeffs("eloi")
grid = Grid2D(Nx=99, Ny=99, Lx=1.0, Ly=1.0)
V = grid.Vflat
N = 99
K_COMMON = 95          # t = 47.5 solver units = 50.1 days (all runs reach it)
SNAP = 45              # snapshot time used for field-based quantities (47.4 days)

D = {"NH4_um": NH4, "O2_um": O2, "tau_day": float(u.tau_day)}


def load_point(res, i, j):
    return json.loads((res / f"point_nh4_{i}_o2_{j}.json").read_text())


# ------------------------------------------------------------ time series
M = {s: np.full((NI, NJ, K_COMMON + 1), np.nan) for s in SPECIES}      # combined, solver units
Mc = np.full((NI, NJ, K_COMMON + 1), np.nan)                           # CMX alone
rate = {s: np.full((NI, NJ, K_COMMON + 1), np.nan) for s in SUBSTRATES}
stock = {s: np.full((NI, NJ, K_COMMON + 1), np.nan) for s in SUBSTRATES}
t_stop = np.zeros((NI, NJ)); k_stop = np.zeros((NI, NJ), int)
Mstop = {s: np.zeros((NI, NJ)) for s in SPECIES}
for i in range(NI):
    for j in range(NJ):
        r = load_point(RES_ALL, i, j)
        k = K_COMMON + 1
        for s in SPECIES:
            M[s][i, j] = np.array(r["M"][s][:k])
        for s in SUBSTRATES:
            rate[s][i, j] = np.array(r["rate"][s][:k])
            stock[s][i, j] = np.array(r["stock"][s][:k])
        t_stop[i, j] = r["t_stop"]; k_stop[i, j] = r["k_stop"]
        for s in SPECIES:
            Mstop[s][i, j] = r["M"][s][r["k_stop"]]
        rc = load_point(RES_CMX, i, j)
        Mc[i, j] = np.array(rc["M"]["CMX"][:k])
D.update(t_solver=np.arange(K_COMMON + 1) * 0.5, t_day=np.arange(K_COMMON + 1) * 0.5 * u.tau_day)
D["M_ug"] = {s: u.biomass_ug_per_mm(M[s]) for s in SPECIES}
D["Mc_ug"] = u.biomass_ug_per_mm(Mc)
D["t_stop_day"] = u.days(t_stop)
D["Mstop_ug"] = {s: u.biomass_ug_per_mm(Mstop[s]) for s in SPECIES}
D["rate"] = rate; D["stock"] = stock

# ------------------------------------------------------------ field-based quantities at SNAP
keys = ["fN", "fO", "anoxic", "noN", "share_NH4_CMX", "spec_NH4_AOB", "spec_NH4_CMX",
        "O2share_AOB", "O2share_NOB", "O2share_CMX", "NH4rate_AOB", "NH4rate_CMX",
        "O2rate_AOB", "O2rate_NOB", "O2rate_CMX", "NO2rate_NOB", "area_frac", "meanrho"]
F = {s: {k: np.full((NI, NJ), np.nan) for k in ("fN", "fO", "anoxic", "noN", "M", "both", "grow")} for s in SPECIES}
G = {k: np.full((NI, NJ), np.nan) for k in keys[4:]}
profiles = {}
cref = u.c_ref_uM


def per_species_rates(Cd, Ud):
    out = {}
    for sp in SPECIES:
        p_sub, s_sub = PRIMARY[sp], SECONDARY[sp]
        Mp = monod(Cd[p_sub], coeffs["Khat"][sp][p_sub]); Ms = monod(Cd[s_sub], coeffs["Khat"][sp][s_sub])
        w = Ud[sp]
        out[sp] = {"Mp": Mp, "Ms": Ms,
                   p_sub: coeffs["Lambda"][sp][p_sub] * np.sum(V * w * Mp * Ms),
                   s_sub: coeffs["Lambda"][sp][s_sub] * np.sum(V * w * Mp * Ms)}
    return out


def to_umol_day(sub, x):
    return float(u.amount_umol_per_mm(sub, x) / u.tau_day)


calib = None
for i in range(NI):
    for j in range(NJ):
        d = RES_ALL / f"fields_nh4_{i}_o2_{j}"
        Cd = {s: np.load(d / f"t{SNAP}_c_{s}.npy").astype(np.float64) for s in SUBSTRATES}
        Ud = {s: np.load(d / f"t{SNAP}_u_{s}.npy").astype(np.float64) for s in SPECIES}
        pr = per_species_rates(Cd, Ud)
        o2uM = Cd["O2"] * cref; nh4uM = Cd["NH4"] * cref
        for sp in SPECIES:
            mass = np.sum(V * Ud[sp])
            F[sp]["M"][i, j] = mass
            if mass > 0:
                w = V * Ud[sp] / mass
                F[sp]["fN"][i, j] = np.sum(w * pr[sp]["Mp"])
                F[sp]["fO"][i, j] = np.sum(w * pr[sp]["Ms"])
                F[sp]["anoxic"][i, j] = np.sum(w * (o2uM < 1.0))
                F[sp]["noN"][i, j] = np.sum(w * (nh4uM < 0.1)) if sp != "NOB" else np.sum(w * (Cd["NO2"] * cref < 0.01))
                nav = (nh4uM > 0.1) if sp != "NOB" else (Cd["NO2"] * cref > 0.01)
                F[sp]["both"][i, j] = np.sum(w * ((o2uM > 1.0) & nav))
                rho_ = Ud["AOB"] + Ud["NOB"] + Ud["CMX"]
                F[sp]["grow"][i, j] = np.sum(w * ((coeffs["rhat"][sp] * pr[sp]["Mp"] * pr[sp]["Ms"] - coeffs["bhat"][sp] * rho_) > 0))
        nh4_a = to_umol_day("NH4", pr["AOB"]["NH4"]); nh4_c = to_umol_day("NH4", pr["CMX"]["NH4"])
        o2_a = to_umol_day("O2", pr["AOB"]["O2"]); o2_n = to_umol_day("O2", pr["NOB"]["O2"]); o2_c = to_umol_day("O2", pr["CMX"]["O2"])
        G["NH4rate_AOB"][i, j] = nh4_a; G["NH4rate_CMX"][i, j] = nh4_c
        G["O2rate_AOB"][i, j] = o2_a; G["O2rate_NOB"][i, j] = o2_n; G["O2rate_CMX"][i, j] = o2_c
        G["NO2rate_NOB"][i, j] = to_umol_day("NO2", pr["NOB"]["NO2"])
        tot_n = nh4_a + nh4_c
        G["share_NH4_CMX"][i, j] = nh4_c / tot_n if tot_n > 0 else np.nan
        tot_o = o2_a + o2_n + o2_c
        for sp, v in (("AOB", o2_a), ("NOB", o2_n), ("CMX", o2_c)):
            G[f"O2share_{sp}"][i, j] = v / tot_o if tot_o > 0 else np.nan
        mA, mC = F["AOB"]["M"][i, j], F["CMX"]["M"][i, j]
        G["spec_NH4_AOB"][i, j] = nh4_a / u.biomass_ug_per_mm(mA) if mA > 0 else np.nan   # umol per ug per day
        G["spec_NH4_CMX"][i, j] = nh4_c / u.biomass_ug_per_mm(mC) if mC > 0 else np.nan
        G["area_frac"][i, j] = np.sum(V * ((Ud["AOB"] + Ud["NOB"] + Ud["CMX"]) > 1e-3))
        rho = Ud["AOB"] + Ud["NOB"] + Ud["CMX"]
        G["meanrho"][i, j] = np.sum(V * rho) / max(np.sum(V * (rho > 1e-3)), 1e-12)
        # calibration against the stored total NH4 rate (k=90 corresponds to t=45)
        if calib is None and i == 8 and j == 6:
            R, _ = __import__("nitrifiers.one_dimensional.elliptic", fromlist=["x"]).reaction_and_jacobian(Cd, Ud, coeffs)
            calib = (to_umol_day("NH4", float(np.sum(V * R["NH4"]))), to_umol_day("NH4", pr["AOB"]["NH4"] + pr["CMX"]["NH4"]),
                     to_umol_day("NH4", float(rate["NH4"][i, j, 90])))
        if (i, j) in {(8, 6), (7, 6), (6, 6), (4, 6), (8, 2), (6, 2), (4, 4), (3, 2), (2, 4), (0, 0), (8, 0), (4, 0)}:
            n = N + 1
            row = lambda a: a.reshape(n, n).T[50, :]
            profiles[(i, j)] = {"x_um": np.linspace(0, u.L_um, n),
                                "u": {s: row(Ud[s]) for s in SPECIES},
                                "c": {s: row(Cd[s]) * cref for s in SUBSTRATES}}
print("calibration (sum R_NH4 from fields, sum of species parts, stored rate k=90):", calib)
D["F"] = F; D["G"] = G; D["profiles"] = profiles
D["snap_day"] = float(SNAP * u.tau_day)
pickle.dump(D, open(OUT, "wb"))
print("saved", OUT, "snapshot day", D["snap_day"])
