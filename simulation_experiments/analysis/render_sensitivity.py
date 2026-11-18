"""Figure for the sensitivity runs: CMX mass and AOB mass after 50 days relative to the start, per condition and variant.
    python render_sensitivity.py [output_dir]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE.parent / "sensitivity" / "sensitivity.json"))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
out.mkdir(exist_ok=True, parents=True)

conds = sorted({(r["NH4"], r["O2"]) for r in d})
conds = sorted(conds, key=lambda x: (x[0], x[1]))
variants = [("base", "base case"), ("b05", "death $\\times0.5$"), ("b2", "death $\\times2$"), ("k13", "$K_{\\rm CMX}$ ammonium $\\times13$"), ("outer", "CMX in outer ring")]
cols = ["#555555", "#4c9a6a", "#c0504d", "#8c6bb1", "#e08a1e"]
get = lambda v, c, s: next(r for r in d if r["variant"] == v and r["N"] == 99 and (r["NH4"], r["O2"]) == c)["M"][s]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for k, (sp, lab) in enumerate([("CMX", "CMX mass after 50 d / start"), ("AOB", "AOB mass after 50 d / start")]):
    for vi, (v, name) in enumerate(variants):
        y = [get(v, c, sp)[-1] / get(v, c, sp)[0] for c in conds]
        ax[k].bar(np.arange(len(conds)) + (vi - 2) * 0.16, y, 0.16, color=cols[vi], label=name)
    ax[k].axhline(1, color="k", lw=0.8, ls="--")
    ax[k].set_xticks(range(len(conds)))
    ax[k].set_xticklabels(["%g / %g" % c for c in conds], fontsize=8)
    ax[k].set_xlabel("ammonium / oxygen at the boundary ($\\mu$M)")
    ax[k].set_ylabel(lab)
ax[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out / "r_sensitivity.pdf")
fig.savefig(out / "r_sensitivity.png", dpi=200)
print("ok")
