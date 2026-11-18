"""Figure of the boundary-condition regime scan (run_bc_scan.py must have been run)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from matplotlib.colors import LinearSegmentedColormap as _LSC
def wmap(cmap):
    """Lowest value pure white (diverging maps: pure white in the middle)."""
    base = plt.get_cmap(cmap)
    name = base.name
    if name in ("RdBu", "RdBu_r", "PRGn", "PRGn_r"):
        cols = [base(x) for x in np.linspace(0, 0.42, 120)] + [(1, 1, 1, 1)] + [base(x) for x in np.linspace(0.58, 1, 120)]
        return _LSC.from_list(name + "_w", cols, N=256)
    xs_ = np.linspace(0, 1, 256)
    w = np.clip(xs_ / 0.35, 0, 1)[:, None]
    cols = (1 - w) * np.ones((256, 4)) + w * np.array([base(x) for x in xs_])
    cols[:, 3] = 1.0
    return _LSC.from_list(name + "_w", cols, N=256)


HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\amogh\OneDrive\Desktop\Thesis\Overleaf_export\figures")
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "figure.dpi": 200, "savefig.dpi": 220})
r = json.load(open(HERE / "bc_scan.json"))
nm = sorted([x for x in r if x["kind"] == "neumann"], key=lambda x: x["par"])
BI = sorted({x["par"][0] for x in r if x["kind"] == "robin"})
CI = sorted({x["par"][1] for x in r if x["kind"] == "robin"})
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
q = np.array([x["par"] for x in nm]); m = np.array([x["m_rel"] for x in nm]); st = np.array([x["stalled"] > 0 for x in nm])
ax.loglog(q, m, "-", color="0.5", lw=1)
ax.loglog(q[~st], m[~st], "o", color="#1b6e2f", ms=4, label="all substrate solves converge")
ax.loglog(q[st], m[st], "s", color="#c0392b", ms=4, label="some solves fail (no steady state)")
ax.axhline(1, color="k", lw=0.6, ls=":")
ax.set_xlabel("supplied flux $|\\hat g|$"); ax.set_ylabel("total biomass at $t=20$ / initial")
ax.legend(frameon=False, loc="upper left", fontsize=6.5); ax.set_title("(a) Neumann condition")
ax = axes[1]
Z = np.full((len(CI), len(BI)), np.nan)
for x in r:
    if x["kind"] == "robin":
        Z[CI.index(x["par"][1]), BI.index(x["par"][0])] = x["m_rel"]
im = ax.imshow(Z, origin="lower", aspect="auto", cmap=wmap("YlGn"), norm=LogNorm(1, 80))
ax.set_xticks(range(len(BI))); ax.set_xticklabels([("%g" % b) for b in BI]); ax.set_yticks(range(len(CI))); ax.set_yticklabels([("%g" % c) for c in CI])
for i, c in enumerate(CI):
    for j, b in enumerate(BI):
        ax.text(j, i, "%.1f" % Z[i, j] if Z[i, j] < 10 else "%d" % round(Z[i, j]), ha="center", va="center", fontsize=6.3, color="white" if Z[i, j] > 15 else "black")
ax.set_xlabel("exchange number Bi"); ax.set_ylabel("outer concentration $\\hat c^\\infty$"); ax.set_title("(b) Robin condition, biomass at $t=20$ / initial")
plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
fig.tight_layout(); fig.savefig(OUT / "r_bc_regimes.png", bbox_inches="tight")
