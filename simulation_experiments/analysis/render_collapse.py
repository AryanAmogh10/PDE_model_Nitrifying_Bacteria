"""Collapse of the outcomes onto the supply ratio R = D4 c4 / (2 D1 c1) (Chapter 8)."""
import json, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\amogh\OneDrive\Desktop\Thesis\Overleaf_export\figures")
D = pickle.load(open(HERE / "sweep_derived.pkl", "rb"))
NH4, O2 = D["NH4_um"], D["O2_um"]
M = D["M_ug"]; k = len(D["t_day"]) - 1
rs = json.load(open(HERE.parent / "fastest_run_comparison" / "rates_summary.json"))
share = np.array(rs["NO3_share_total"])          # [O2, NH4]
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "figure.dpi": 200, "savefig.dpi": 220})
fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.2))
cm = plt.get_cmap("viridis")
panels = [("$F_{\\mathrm{CMX}}$", lambda i, j: M["CMX"][i, j, k] / (M["AOB"][i, j, k] + M["CMX"][i, j, k])),
          ("AOB mass / initial", lambda i, j: M["AOB"][i, j, k] / M["AOB"][i, j, 0]),
          ("NOB mass / initial", lambda i, j: M["NOB"][i, j, k] / M["NOB"][i, j, 0]),
          ("nitrate share", lambda i, j: share[j, i])]
for ax, (t, f) in zip(axes, panels):
    for j in range(len(O2)):
        R = 1.81 * O2[j] / (2 * 1.69 * NH4)
        y = [f(i, j) for i in range(len(NH4))]
        ax.plot(R, y, "o-", ms=2.5, lw=0.7, color=cm(j / (len(O2) - 1)))
    ax.axvline(1, color="k", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("$R$")
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, len(O2) - 1))
cb = fig.colorbar(sm, ax=axes, pad=0.01, fraction=0.025, ticks=range(len(O2)))
cb.ax.set_yticklabels(["%g" % o for o in O2]); cb.set_label("O$_2$ [$\\mu$M]")
fig.savefig(OUT / "r_collapse.png", bbox_inches="tight")
