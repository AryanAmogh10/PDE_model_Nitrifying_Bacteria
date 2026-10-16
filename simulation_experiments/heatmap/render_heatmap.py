"""Heatmaps over (NH4_BC, O2_BC):
  heatmap_F_cmx.png        F_CMX = M_CMX / (M_AOB + M_CMX) at stop time, F_CMX
                           against NH4 per O2 with the transitions, stop time
  heatmap_growth.png       growth of AOB and CMX in the combined system, and
                           CMX in the combined system relative to CMX alone
  heatmap_resolution.png   F_CMX on 50x50 against 100x100
    python render_heatmap.py [N]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as c
from run_heatmap import NH4_UM, O2_UM, PRESET, point_file

N = int(sys.argv[1]) if len(sys.argv) > 1 else 99
u = c.Units(PRESET)
nI, nJ = len(NH4_UM), len(O2_UM)


def load(system, n):
    recs = {}
    for i in range(nI):
        for j in range(nJ):
            f = point_file(i, j, system, n)
            if f.exists():
                recs[(i, j)] = json.loads(f.read_text())
    return recs


def grid_of(recs, fn):
    G = np.full((nJ, nI), np.nan)
    for (i, j), r in recs.items():
        G[j, i] = fn(r)
    return G


def mass_at(r, sp, t):
    return float(np.interp(t, r["t"], r["M"][sp]))


comb = load("all", N)
alone = load("cmx", N)
F = grid_of(comb, lambda r: r["F_CMX"])
T = grid_of(comb, lambda r: u.days(r["t_stop"]))
touched = grid_of(comb, lambda r: r["info"]["boundary_touch_t"] is not None).astype(bool) if comb else np.zeros((nJ, nI), bool)
gA = grid_of(comb, lambda r: r["M"]["AOB"][r["k_stop"]] / r["M"]["AOB"][0])
gC = grid_of(comb, lambda r: r["M"]["CMX"][r["k_stop"]] / r["M"]["CMX"][0])
ratio = np.full((nJ, nI), np.nan)
gC_alone = np.full((nJ, nI), np.nan)
for (i, j), r in comb.items():
    if (i, j) in alone:
        a = alone[(i, j)]
        t = min(r["t_stop"], a["t"][c.last_valid_index(a, a["info"])])
        ratio[j, i] = mass_at(r, "CMX", t) / mass_at(a, "CMX", t)
        gC_alone[j, i] = mass_at(a, "CMX", t) / a["M"]["CMX"][0]


def transition(G, j):
    out = []
    for i in range(nI - 1):
        a, b = G[j, i], G[j, i + 1]
        if np.isfinite(a) and np.isfinite(b) and (a - 0.5) * (b - 0.5) < 0:
            w = (0.5 - a) / (b - a)
            out.append(float(np.exp(np.log(NH4_UM[i]) + w * np.log(NH4_UM[i + 1] / NH4_UM[i]))))
    return out


def cells(ax, G, fmt, cmap, vmin, vmax, title, cbar, mark=None, txtcolor="k"):
    if not np.isfinite(G).any():
        raise SystemExit(f"no data for panel '{title.splitlines()[0]}': run the missing sweep first")
    im = ax.pcolormesh(np.arange(nI + 1), np.arange(nJ + 1), G, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    for j in range(nJ):
        for i in range(nI):
            if np.isfinite(G[j, i]):
                s = fmt.format(G[j, i]) + ("*" if mark is not None and not mark[j, i] else "")
                ax.text(i + 0.5, j + 0.5, s, ha="center", va="center", fontsize=7, color=txtcolor)
    ax.set_xticks(np.arange(nI) + 0.5); ax.set_xticklabels([f"{v:g}" for v in NH4_UM], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(nJ) + 0.5); ax.set_yticklabels([f"{v:g}" for v in O2_UM])
    ax.set_xlabel("NH4 at boundary [uM]"); ax.set_ylabel("O2 at boundary [uM]")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=cbar)


grid_label = f"{N + 1}x{N + 1} grid"
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
cells(ax[0], F, "{:.2f}", "PRGn_r", 0, 1,
      "F_CMX = M_CMX / (M_AOB + M_CMX) at stop time\n(purple: more CMX, green: more AOB; starts at 0.5)", "F_CMX [-]", touched)
for j in range(nJ):
    ok = np.isfinite(F[j])
    line, = ax[1].semilogx(NH4_UM[ok], F[j, ok], "o-", label=f"O2 = {O2_UM[j]:g} uM")
    for x in transition(F, j):
        ax[1].axvline(x, color=line.get_color(), lw=0.7, ls=":")
ax[1].axhline(0.5, color="grey", lw=0.8, ls="--")
ax[1].set_ylim(0, 1); ax[1].set_xlabel("NH4 at boundary [uM]"); ax[1].set_ylabel("F_CMX [-]")
ax[1].set_title("F_CMX against NH4 at fixed O2\n(dotted: transition, F_CMX = 0.5)", fontsize=10); ax[1].legend(fontsize=8)
cells(ax[2], T, "{:.0f}", "viridis", None, None,
      "Stop time [days]: just before reaching the boundary\n(* in the left panel: never reached it, stopped at 63 days)", "days", txtcolor="w")
fig.suptitle(f"AOB vs CMX ({PRESET}), combined system, equal initial AOB and CMX mass, NOB ignored in F, {grid_label}", fontsize=12)
fig.tight_layout(); fig.savefig(HERE / "heatmap_F_cmx.png", dpi=130); plt.close(fig)
print("saved heatmap_F_cmx.png")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
cells(ax[0], gA, "{:.2f}", "RdBu", 0, 2, "AOB growth in the combined system\nM_AOB(stop) / M_AOB(0)  (>1 grows, <1 declines)", "[-]")
cells(ax[1], gC, "{:.2f}", "RdBu", 0, 2, "CMX growth in the combined system\nM_CMX(stop) / M_CMX(0)", "[-]")
cells(ax[2], ratio, "{:.2f}", "magma", 0, 1, "CMX in the combined system relative to CMX alone\nM_CMX(combined) / M_CMX(alone) at the same time", "[-]",
      txtcolor="w")
fig.suptitle(f"Growth and competitive loss ({PRESET}, {grid_label})", fontsize=12)
fig.tight_layout(); fig.savefig(HERE / "heatmap_growth.png", dpi=130); plt.close(fig)
print("saved heatmap_growth.png")

coarse = load("all", 49)
if coarse and N != 49:
    Fc = grid_of(coarse, lambda r: r["F_CMX"])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    cells(ax[0], Fc, "{:.2f}", "PRGn_r", 0, 1, "F_CMX, 50x50 grid", "F_CMX [-]")
    cells(ax[1], F, "{:.2f}", "PRGn_r", 0, 1, f"F_CMX, {grid_label}", "F_CMX [-]")
    cells(ax[2], F - Fc, "{:+.2f}", "coolwarm", -0.3, 0.3, f"difference ({grid_label} minus 50x50)", "[-]")
    fig.suptitle("Grid resolution check (each point at its own stop time)", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / "heatmap_resolution.png", dpi=130); plt.close(fig)
    print("saved heatmap_resolution.png")


def tolist(G, nd=4):
    return [[None if not np.isfinite(v) else round(float(v), nd) for v in row] for row in G]


summary = {"grid": grid_label, "NH4_uM": NH4_UM.tolist(), "O2_uM": O2_UM.tolist(), "rows_are": "O2_uM", "cols_are": "NH4_uM",
           "F_CMX": tolist(F), "stop_days": tolist(T, 2), "touched_boundary": touched.tolist(),
           "AOB_growth": tolist(gA, 3), "CMX_growth": tolist(gC, 3), "CMX_alone_growth": tolist(gC_alone, 3),
           "CMX_combined_over_alone": tolist(ratio, 3),
           "stalled_steps": tolist(grid_of(comb, lambda r: r["info"]["stalled"]), 0),
           "transition_NH4_uM": {f"{O2_UM[j]:g}": [round(x, 2) for x in transition(F, j)] for j in range(nJ)}}
if coarse and N != 49:
    summary["F_CMX_50x50"] = tolist(grid_of(coarse, lambda r: r["F_CMX"]))
    summary["transition_NH4_uM_50x50"] = {f"{O2_UM[j]:g}": [round(x, 2) for x in transition(grid_of(coarse, lambda r: r["F_CMX"]), j)] for j in range(nJ)}
(HERE / "heatmap_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary["transition_NH4_uM"]))
