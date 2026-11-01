"""Growth M_i(T*) / M_i(0) of one species (AOB, CMX or NOB) at one common time T* for every
(NH4_BC, O2_BC) point of the 100x100, 60-unit combined-system sweep. T* is
the stop time of the point that reaches the boundary first (top right:
NH4 = 1000 uM, O2 = 250 uM), so every point is compared at the same time and
before any colony touches the boundary.
    python render_common_time.py [AOB|CMX|NOB]"""
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

N, T_MAX = 99, 60.0
SP = sys.argv[1] if len(sys.argv) > 1 else "AOB"
u = c.Units(PRESET)
nI, nJ = len(NH4_UM), len(O2_UM)
recs = {(i, j): json.loads(point_file(i, j, "all", N, T_MAX).read_text()) for i in range(nI) for j in range(nJ)}

T_star = min(r["t_stop"] for r in recs.values())
assert all(r["t_stop"] >= T_star for r in recs.values())
G = np.full((nJ, nI), np.nan)
for (i, j), r in recs.items():
    G[j, i] = np.interp(T_star, r["t"], r["M"][SP]) / r["M"][SP][0]

fig, ax = plt.subplots(figsize=(9, 5.4))
im = ax.pcolormesh(np.arange(nI + 1), np.arange(nJ + 1), G, cmap="RdBu", vmin=0, vmax=2, shading="flat")
for j in range(nJ):
    for i in range(nI):
        ax.text(i + 0.5, j + 0.5, f"{G[j, i]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(np.arange(nI) + 0.5); ax.set_xticklabels([f"{v:g}" for v in NH4_UM], rotation=45, fontsize=8)
ax.set_yticks(np.arange(nJ) + 0.5); ax.set_yticklabels([f"{v:g}" for v in O2_UM])
ax.set_xlabel("NH4 at boundary [uM]"); ax.set_ylabel("O2 at boundary [uM]")
ax.set_title(f"{SP} growth M_{SP}(t) / M_{SP}(0) at t = {u.days(T_star):.1f} days\n"
             f"(combined system, {PRESET}, {N + 1}x{N + 1} grid; >1 grows, <1 declines)", fontsize=10)
plt.colorbar(im, ax=ax, label=f"M_{SP}(t) / M_{SP}(0) [-]")
fig.tight_layout()
fig.savefig(HERE / f"heatmap_{SP}_growth_common_time.png", dpi=130)
plt.close(fig)

out = {"T_star_solver": T_star, "T_star_days": round(float(u.days(T_star)), 2),
       "NH4_uM": NH4_UM.tolist(), "O2_uM": O2_UM.tolist(), "rows_are": "O2_uM",
       f"{SP}_growth": [[round(float(v), 4) for v in row] for row in G]}
(HERE / f"heatmap_{SP}_growth_common_time.json").write_text(json.dumps(out, indent=1))
print(f"T* = {T_star} solver units = {u.days(T_star):.2f} days")
for j in reversed(range(nJ)):
    print(f"O2 {O2_UM[j]:5g}: " + " ".join(f"{v:5.2f}" for v in G[j]))
