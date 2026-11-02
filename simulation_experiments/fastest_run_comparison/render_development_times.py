"""Development of every combined-system run at several times up to T*, the
stop time of the first run (NH4 1000 uM, O2 250 uM; red box): net specific
growth rate of each species, and the colony footprint relative to t = 0.
  development_over_time.png, development_over_time.json
    python render_development_times.py [T_max of the sweep, default 150]"""
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
U_FOOT = 1e-3
u = c.Units(PRESET)
recs = {(i, j): json.loads(point_file(i, j, "all", N, T_MAX).read_text()) for i in range(nI) for j in range(nJ)}
FAST = min(recs, key=lambda ij: recs[ij]["t_stop"])
T_STAR = recs[FAST]["t_stop"]
TIMES = [10.0, 20.0, 30.0, 40.0, T_STAR]                     # growth rates
FOOT_TIMES = [10.0, 20.0, 30.0, 40.0, 5.0 * np.floor(T_STAR / 5.0)]   # snapshot times
V_UM2 = c.Grid2D(Nx=N, Ny=N).Vflat * u.L_um ** 2


def specific(r, s, t):
    M, tt = np.asarray(r["M"][s]), np.asarray(r["t"])
    k = int(np.searchsorted(tt, t))
    return float(np.gradient(np.log(M), tt)[k] / u.tau_day)


def area(i, j, t):
    d = point_file(i, j, "all", N, T_MAX).parent / f"fields_nh4_{i}_o2_{j}"
    tot = sum(np.load(d / f"t{t:g}_u_{s}.npy").astype(np.float64) for s in SPECIES)
    return float(np.sum(V_UM2[tot > U_FOOT]))


def grid_of(fn):
    G = np.full((nJ, nI), np.nan)
    for (i, j), r in recs.items():
        G[j, i] = fn(i, j, r)
    return G


mu = {s: [grid_of(lambda i, j, r, s=s, t=t: specific(r, s, t)) for t in TIMES] for s in SPECIES}
a0 = grid_of(lambda i, j, r: area(i, j, 0))
foot = [grid_of(lambda i, j, r, t=t: area(i, j, t)) / a0 for t in FOOT_TIMES]


def cells(a, G, fmt, cmap, vmin, vmax):
    im = a.pcolormesh(np.arange(nI + 1), np.arange(nJ + 1), G, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    for j in range(nJ):
        for i in range(nI):
            a.text(i + 0.5, j + 0.5, fmt.format(G[j, i]), ha="center", va="center", fontsize=5.5)
    a.add_patch(Rectangle((FAST[0], FAST[1]), 1, 1, fill=False, ec="r", lw=2))
    a.set_xticks(np.arange(nI) + 0.5); a.set_xticklabels([f"{v:g}" for v in NH4_UM], rotation=45, fontsize=6)
    a.set_yticks(np.arange(nJ) + 0.5); a.set_yticklabels([f"{v:g}" for v in O2_UM], fontsize=6)
    return im


lim = max(np.nanmax(np.abs(G)) for s in SPECIES for G in mu[s])
fmax = max(np.nanmax(G) for G in foot)
rows = [(f"{s}\nnet specific growth\nrate [1/day]", mu[s], TIMES, "{:+.3f}", "RdBu", -lim, lim) for s in SPECIES]
rows.append(("colony footprint\n/ footprint at 0 [-]", foot, FOOT_TIMES, "{:.2f}", "Greens", 1.0, fmax))
fig, ax = plt.subplots(len(rows), len(TIMES), figsize=(4.2 * len(TIMES), 3.4 * len(rows)))
fig.subplots_adjust(top=0.94, hspace=0.45, wspace=0.25)
for r, (label, Gs, times, fmt, cmap, vmin, vmax) in enumerate(rows):
    for k, (G, t) in enumerate(zip(Gs, times)):
        im = cells(ax[r, k], G, fmt, cmap, vmin, vmax)
        ax[r, k].set_title(f"t = {u.days(t):.1f} days", fontsize=9)
        if k == 0:
            ax[r, k].set_ylabel(f"{label}\n\nO2 at boundary [uM]", fontsize=8)
        if r == len(rows) - 1:
            ax[r, k].set_xlabel("NH4 at boundary [uM]", fontsize=8)
    fig.colorbar(im, ax=list(ax[r]), fraction=0.015, pad=0.01)
fig.suptitle(f"Development of all combined runs up to T* = {u.days(T_STAR):.1f} days, the stop time of the first run "
             f"(red box); {PRESET}, {N + 1}x{N + 1} grid; blue grows, red declines; footprint: total volume fraction > {U_FOOT:g}",
             fontsize=11)
fig.savefig(HERE / "development_over_time.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("saved development_over_time.png")


def tolist(G, nd=5):
    return [[None if not np.isfinite(v) else round(float(v), nd) for v in row] for row in G]


out = {"NH4_uM": NH4_UM.tolist(), "O2_uM": O2_UM.tolist(), "rows_are": "O2_uM", "cols_are": "NH4_uM",
       "first_to_stop": {"NH4_uM": float(NH4_UM[FAST[0]]), "O2_uM": float(O2_UM[FAST[1]])},
       "specific_growth_per_day": {s: {f"{u.days(t):.1f}": tolist(G) for t, G in zip(TIMES, mu[s])} for s in SPECIES},
       "footprint_ratio": {f"{u.days(t):.1f}": tolist(G, 3) for t, G in zip(FOOT_TIMES, foot)}}
(HERE / "development_over_time.json").write_text(json.dumps(out, indent=1))
for s in SPECIES:
    print(s, "fast:", [round(G[FAST[1], FAST[0]], 4) for G in mu[s]], "map max:", [round(float(np.nanmax(G)), 4) for G in mu[s]])
print("foot fast:", [round(G[FAST[1], FAST[0]], 2) for G in foot], "map min:", [round(float(np.nanmin(G)), 2) for G in foot])
