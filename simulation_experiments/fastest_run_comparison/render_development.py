"""Biomass development of every combined-system run against the first run
to stop (NH4 1000 uM, O2 250 uM), compared at T* = its stop time.
  development_timeseries.png   mass and net specific growth rate per species
  development_at_Tstar.png     specific growth rate at T*, mean rate over
                               0..T*, colony footprint and total growth rate
    python render_development.py [T_max of the sweep, default 150]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).parent.parent / "heatmap"))
from run_heatmap import NH4_UM, o2_values, PRESET, point_file
import common as c
from common import SPECIES

HERE = Path(__file__).parent
T_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
N = 99
O2_UM = o2_values(T_MAX)
nI, nJ = len(NH4_UM), len(O2_UM)
U_FOOT = 1e-3   # volume fraction above which a node counts as colony
u = c.Units(PRESET)
recs = {(i, j): json.loads(point_file(i, j, "all", N, T_MAX).read_text()) for i in range(nI) for j in range(nJ)}
FAST = min(recs, key=lambda ij: recs[ij]["t_stop"])
T_STAR = recs[FAST]["t_stop"]
K = int(np.searchsorted(recs[FAST]["t"], T_STAR))
T_FOOT = 5.0 * np.floor(T_STAR / 5.0)   # last snapshot before T*
V_UM2 = c.Grid2D(Nx=N, Ny=N).Vflat * u.L_um ** 2


def mass(r, s):
    return u.biomass_ug_per_mm(np.asarray(r["M"][s]))


def specific(r, s):
    """Net specific growth rate d ln M / dt [1/day]."""
    M = np.asarray(r["M"][s])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(np.log(np.where(M > 0, M, np.nan)), np.asarray(r["t"])) / u.tau_day


def footprint(i, j, t):
    d = point_file(i, j, "all", N, T_MAX).parent / f"fields_nh4_{i}_o2_{j}"
    tot = sum(np.load(d / f"t{t:g}_u_{s}.npy").astype(np.float64) for s in SPECIES)
    return float(np.sum(V_UM2[tot > U_FOOT]))


cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for (i, j), r in recs.items():
    if (i, j) == FAST:
        continue
    k = c.last_valid_index(r, r["info"])
    days = u.days(r["t"][:k + 1])
    for col, s in enumerate(SPECIES):
        kw = dict(color=cmap(j / (nJ - 1)), lw=0.7, alpha=0.6)
        ax[0, col].plot(days, mass(r, s)[:k + 1], **kw)
        ax[1, col].plot(days, specific(r, s)[:k + 1], **kw)
r = recs[FAST]
for col, s in enumerate(SPECIES):
    days = u.days(r["t"][:K + 1])
    ax[0, col].plot(days, mass(r, s)[:K + 1], "k", lw=2.5, label="first to stop (NH4 1000, O2 250 uM)")
    ax[1, col].plot(days, specific(r, s)[:K + 1], "k", lw=2.5)
    ax[0, col].set_yscale("log")
    ax[0, col].set_title(f"{s} biomass [ug per mm depth]", fontsize=10)
    ax[1, col].set_title(f"{s} net specific growth rate d ln M / dt [1/day]", fontsize=10)
    ax[1, col].axhline(0, color="grey", lw=0.8)
    for a in ax[:, col]:
        a.axvline(u.days(T_STAR), color="r", ls="--", lw=1)
        a.set_xlabel("time [days]")
for j in range(nJ):
    ax[0, 0].plot([], [], color=cmap(j / (nJ - 1)), label=f"O2 = {O2_UM[j]:g} uM")
ax[0, 0].plot([], [], "r--", label=f"T* = {u.days(T_STAR):.1f} days")
ax[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle(f"Biomass development of all {len(recs)} combined runs ({PRESET}, {N + 1}x{N + 1} grid), each up to its own stop time; "
             "thin lines coloured by O2_BC", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "development_timeseries.png", dpi=130); plt.close(fig)
print("saved development_timeseries.png")


def grid_of(fn):
    G = np.full((nJ, nI), np.nan)
    for (i, j), rr in recs.items():
        G[j, i] = fn(i, j, rr)
    return G


mu_T = {s: grid_of(lambda i, j, rr, s=s: specific(rr, s)[K]) for s in SPECIES}
mu_mean = {s: grid_of(lambda i, j, rr, s=s: np.log(rr["M"][s][K] / rr["M"][s][0]) / u.days(T_STAR)) for s in SPECIES}
foot = grid_of(lambda i, j, rr: footprint(i, j, T_FOOT) / footprint(i, j, 0))
dM_T = grid_of(lambda i, j, rr: sum(np.gradient(mass(rr, s), u.days(np.asarray(rr["t"])))[K] for s in SPECIES))


def cells(a, G, title, cbar, fmt, cmap, vmin, vmax):
    im = a.pcolormesh(np.arange(nI + 1), np.arange(nJ + 1), G, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    for j in range(nJ):
        for i in range(nI):
            a.text(i + 0.5, j + 0.5, fmt.format(G[j, i]), ha="center", va="center", fontsize=6.5)
    a.add_patch(Rectangle((FAST[0], FAST[1]), 1, 1, fill=False, ec="r", lw=2.5))
    a.set_xticks(np.arange(nI) + 0.5); a.set_xticklabels([f"{v:g}" for v in NH4_UM], rotation=45, fontsize=7)
    a.set_yticks(np.arange(nJ) + 0.5); a.set_yticklabels([f"{v:g}" for v in O2_UM], fontsize=7)
    a.set_xlabel("NH4 at boundary [uM]"); a.set_ylabel("O2 at boundary [uM]")
    a.set_title(title, fontsize=9)
    plt.colorbar(im, ax=a, label=cbar)


lim = max(np.nanmax(np.abs(G)) for G in list(mu_T.values()) + list(mu_mean.values()))
fig, ax = plt.subplots(2, 4, figsize=(24, 9.5))
for col, s in enumerate(SPECIES):
    cells(ax[0, col], mu_T[s], f"{s} net specific growth rate at T* [1/day]", "1/day", "{:+.3f}", "RdBu", -lim, lim)
    cells(ax[1, col], mu_mean[s], f"{s} mean net specific growth rate 0..T*\nln(M(T*) / M(0)) / T* [1/day]", "1/day",
          "{:+.3f}", "RdBu", -lim, lim)
cells(ax[0, 3], foot, f"colony footprint at {u.days(T_FOOT):.1f} days / at 0\n(area with total volume fraction > {U_FOOT:g})",
      "[-]", "{:.2f}", "Greens", 0, np.nanmax(foot))
cells(ax[1, 3], dM_T, "total biomass growth rate at T*\ndM/dt, all species [ug/mm/day]", "ug/mm/day", "{:+.3f}", "RdBu",
      -np.nanmax(np.abs(dM_T)), np.nanmax(np.abs(dM_T)))
fig.suptitle(f"Combined system at T* = {u.days(T_STAR):.1f} days, the stop time of the first run (red box); "
             f"{PRESET}, {N + 1}x{N + 1} grid; blue grows, red declines", fontsize=12)
fig.tight_layout(); fig.savefig(HERE / "development_at_Tstar.png", dpi=120); plt.close(fig)
print("saved development_at_Tstar.png")


def tolist(G, nd=5):
    return [[None if not np.isfinite(v) else round(float(v), nd) for v in row] for row in G]


summary = {"T_star_units": T_STAR, "T_star_days": round(float(u.days(T_STAR)), 2),
           "first_to_stop": {"NH4_uM": float(NH4_UM[FAST[0]]), "O2_uM": float(O2_UM[FAST[1]])},
           "NH4_uM": NH4_UM.tolist(), "O2_uM": O2_UM.tolist(), "rows_are": "O2_uM", "cols_are": "NH4_uM",
           "specific_growth_at_Tstar_per_day": {s: tolist(mu_T[s]) for s in SPECIES},
           "mean_specific_growth_0_Tstar_per_day": {s: tolist(mu_mean[s]) for s in SPECIES},
           "footprint_ratio": {"t_days": round(float(u.days(T_FOOT)), 2), "threshold": U_FOOT, "values": tolist(foot, 3)},
           "total_growth_rate_at_Tstar_ug_per_mm_per_day": tolist(dM_T)}
(HERE / "development_summary.json").write_text(json.dumps(summary, indent=1))
for name, G in [("mu_T", mu_T), ("mu_mean", mu_mean)]:
    print(name, {s: round(float(G[s][FAST[1], FAST[0]]), 4) for s in SPECIES},
          "max over map", {s: round(float(np.nanmax(G[s])), 4) for s in SPECIES})
print("foot fast", round(foot[FAST[1], FAST[0]], 2), "dM fast", round(dM_T[FAST[1], FAST[0]], 3))
