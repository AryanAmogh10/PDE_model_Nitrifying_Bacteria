"""Nitrogen turnover of every combined-system run against the first run to
stop (NH4 1000 uM, O2 250 uM), from the stored volume-integrated reaction
rates. Compared at T* = its stop time, when no run has reached the boundary.
  rates_timeseries.png   NH4 consumption, net NO2 and NO3 production over time
  rates_at_Tstar.png     the same at T* and accumulated up to T*, and the
                         NO3 share of the oxidised N
    python render_rates.py [T_max of the sweep, default 150]"""
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

HERE = Path(__file__).parent
T_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
N = 99
O2_UM = o2_values(T_MAX)
nI, nJ = len(NH4_UM), len(O2_UM)
u = c.Units(PRESET)
recs = {(i, j): json.loads(point_file(i, j, "all", N, T_MAX).read_text()) for i in range(nI) for j in range(nJ)}
FAST = min(recs, key=lambda ij: recs[ij]["t_stop"])
T_STAR = recs[FAST]["t_stop"]
K = int(np.searchsorted(recs[FAST]["t"], T_STAR))
SIGN = {"NH4": -1.0, "NO2": 1.0, "NO3": 1.0}   # consumption shown positive
LABEL = {"NH4": "NH4 consumption", "NO2": "net NO2 production", "NO3": "NO3 production"}


def per_day(r, sub):
    """Volume-integrated rate in umol per mm depth per day."""
    return SIGN[sub] * u.amount_umol_per_mm(sub, np.asarray(r["rate"][sub])) / u.tau_day


def total(r, sub, k):
    return SIGN[sub] * float(u.amount_umol_per_mm(sub, c.cumulative(r, sub)[k]))


cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
for (i, j), r in recs.items():
    k = c.last_valid_index(r, r["info"])
    days = u.days(r["t"][:k + 1])
    for a, sub in zip(ax, SIGN):
        if (i, j) == FAST:
            continue
        a.plot(days, per_day(r, sub)[:k + 1], color=cmap(j / (nJ - 1)), lw=0.7, alpha=0.6)
for a, sub in zip(ax, SIGN):
    r = recs[FAST]
    a.plot(u.days(r["t"][:K + 1]), per_day(r, sub)[:K + 1], "k", lw=2.5, label="first to stop (NH4 1000, O2 250 uM)")
    a.axvline(u.days(T_STAR), color="r", ls="--", lw=1, label=f"T* = {u.days(T_STAR):.1f} days")
    a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel("time [days]"); a.set_ylabel("umol per mm depth per day")
    a.set_title(LABEL[sub] + " (whole domain)", fontsize=10)
for j in range(nJ):
    ax[0].plot([], [], color=cmap(j / (nJ - 1)), label=f"O2 = {O2_UM[j]:g} uM")
ax[0].legend(fontsize=7, ncol=2)
fig.suptitle(f"Nitrogen turnover of all {len(recs)} combined runs ({PRESET}, {N + 1}x{N + 1} grid), each up to its own stop time; "
             "thin lines coloured by O2_BC", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "rates_timeseries.png", dpi=130); plt.close(fig)
print("saved rates_timeseries.png")


def grid_of(fn):
    G = np.full((nJ, nI), np.nan)
    for (i, j), r in recs.items():
        G[j, i] = fn(r)
    return G


rate = {s: grid_of(lambda r, s=s: per_day(r, s)[K]) for s in SIGN}
cum = {s: grid_of(lambda r, s=s: total(r, s, K)) for s in SIGN}
share_rate = rate["NO3"] / (rate["NO3"] + rate["NO2"])
share_cum = cum["NO3"] / (cum["NO3"] + cum["NO2"])


def cells(a, G, title, cbar, fmt, cmap, log=False, vmin=None, vmax=None):
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(np.nanmin(G[G > 0]), 1e-4), vmax=np.nanmax(G)) if log else None
    im = a.pcolormesh(np.arange(nI + 1), np.arange(nJ + 1), np.where(G > 0, G, np.nan) if log else G,
                      cmap=cmap, norm=norm, vmin=None if log else vmin, vmax=None if log else vmax, shading="flat")
    for j in range(nJ):
        for i in range(nI):
            a.text(i + 0.5, j + 0.5, fmt(G[j, i]), ha="center", va="center", fontsize=6.5)
    a.add_patch(Rectangle((FAST[0], FAST[1]), 1, 1, fill=False, ec="r", lw=2.5))
    a.set_xticks(np.arange(nI) + 0.5); a.set_xticklabels([f"{v:g}" for v in NH4_UM], rotation=45, fontsize=7)
    a.set_yticks(np.arange(nJ) + 0.5); a.set_yticklabels([f"{v:g}" for v in O2_UM], fontsize=7)
    a.set_xlabel("NH4 at boundary [uM]"); a.set_ylabel("O2 at boundary [uM]")
    a.set_title(title, fontsize=9)
    plt.colorbar(im, ax=a, label=cbar)


def num(v):
    return "0" if v == 0 else (f"{v:.2g}" if abs(v) < 10 else f"{v:.0f}")


fig, ax = plt.subplots(2, 4, figsize=(24, 9.5))
for k, s in enumerate(SIGN):
    cells(ax[0, k], rate[s], f"{LABEL[s]} at T* [umol/mm/day]", "umol/mm/day", num, "YlOrBr", log=True)
    cells(ax[1, k], cum[s], f"{LABEL[s]}, total up to T* [umol/mm]", "umol/mm", num, "YlOrBr", log=True)
cells(ax[0, 3], share_rate, "NO3 share of oxidised N at T*\nNO3 / (NO2 + NO3)", "[-]", lambda v: f"{v:.2f}", "RdYlGn", vmin=0, vmax=1)
cells(ax[1, 3], share_cum, "NO3 share of oxidised N, up to T*", "[-]", lambda v: f"{v:.2f}", "RdYlGn", vmin=0, vmax=1)
fig.suptitle(f"Combined system at T* = {u.days(T_STAR):.1f} days, the stop time of the first run (red box); "
             f"{PRESET}, {N + 1}x{N + 1} grid", fontsize=12)
fig.tight_layout(); fig.savefig(HERE / "rates_at_Tstar.png", dpi=120); plt.close(fig)
print("saved rates_at_Tstar.png")


def tolist(G, nd=4):
    return [[None if not np.isfinite(v) else round(float(v), nd) for v in row] for row in G]


summary = {"T_star_units": T_STAR, "T_star_days": round(float(u.days(T_STAR)), 2),
           "first_to_stop": {"NH4_uM": float(NH4_UM[FAST[0]]), "O2_uM": float(O2_UM[FAST[1]])},
           "NH4_uM": NH4_UM.tolist(), "O2_uM": O2_UM.tolist(), "rows_are": "O2_uM", "cols_are": "NH4_uM",
           "units": {"rate": "umol per mm depth per day", "total": "umol per mm depth", "NH4": "consumption, positive"},
           "rate_at_Tstar": {s: tolist(rate[s]) for s in SIGN}, "total_to_Tstar": {s: tolist(cum[s]) for s in SIGN},
           "NO3_share_rate": tolist(share_rate, 3), "NO3_share_total": tolist(share_cum, 3)}
(HERE / "rates_summary.json").write_text(json.dumps(summary, indent=1))
print({s: (round(rate[s][FAST[1], FAST[0]], 3), round(cum[s][FAST[1], FAST[0]], 2)) for s in SIGN})
