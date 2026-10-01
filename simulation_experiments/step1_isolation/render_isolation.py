"""Renders step 1 for one preset: AOB system vs CMX system.
    python render_isolation.py <preset>
Reads results/<preset>_aob and results/<preset>_cmx; writes
<preset>_comparison.png and <preset>_{aob,cmx}_fields.png."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PRESET = sys.argv[1]
HERE = Path(__file__).parent
RES = {s: HERE / "results" / f"{PRESET}_{s}" for s in ("aob", "cmx")}
SPECIES = ["AOB", "NOB", "CMX"]
SUBSTRATES = ["NH4", "NO2", "NO3", "O2"]
CMAP = {"AOB": LinearSegmentedColormap.from_list("w_g", ["white", "darkgreen"]),
        "NOB": LinearSegmentedColormap.from_list("w_o", ["white", "darkorange"]),
        "CMX": LinearSegmentedColormap.from_list("w_p", ["white", "purple"])}
CMAP_C = LinearSegmentedColormap.from_list("w_b", ["white", "royalblue"])
COLOR = {"AOB": "darkgreen", "NOB": "darkorange", "CMX": "purple"}

hist = {s: json.loads((RES[s] / "history.json").read_text()) for s in RES}
info = {s: json.loads((RES[s] / "run_info.json").read_text()) for s in RES}

# compare both systems only up to the last time before EITHER touches the boundary
DT = info["aob"]["dt"]
T_CMN = min(info[s]["T_end"] - (DT if info[s]["boundary_touch_t"] is not None else 0.0) for s in info)
for s in hist:
    keep = [i for i, t in enumerate(hist[s]["t"]) if t <= T_CMN + 1e-9]
    for key in ("t", "boundary_mass", "cut_mass"):
        hist[s][key] = [hist[s][key][i] for i in keep]
    for grp in ("M", "stock", "rate"):
        hist[s][grp] = {k: [v[i] for i in keep] for k, v in hist[s][grp].items()}


def cumulative(t, rate):
    t, rate = np.asarray(t), np.asarray(rate)
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])


fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
for sysname, ls in (("aob", "-"), ("cmx", "--")):
    h = hist[sysname]
    t = np.asarray(h["t"])
    tot = np.sum([h["M"][s] for s in SPECIES], axis=0)
    label_sys = "AOB system" if sysname == "aob" else "CMX system"
    for s in SPECIES:
        if max(h["M"][s]) > 0:
            ax[0, 0].plot(t, h["M"][s], ls, color=COLOR[s], label=f"{s} ({label_sys})")
    ax[0, 0].plot(t, tot, ls, color="k", lw=0.8, label=f"total ({label_sys})")
    nh4 = -cumulative(t, h["rate"]["NH4"])
    o2 = -cumulative(t, h["rate"]["O2"])
    no3 = cumulative(t, h["rate"]["NO3"])
    no2 = cumulative(t, h["rate"]["NO2"])
    ax[0, 1].plot(t, nh4, ls, color="tab:blue", label=f"NH4 consumed ({label_sys})")
    ax[0, 1].plot(t, o2, ls, color="tab:red", label=f"O2 consumed ({label_sys})")
    ax[1, 0].plot(t, no3, ls, color="tab:purple", label=f"NO3 produced ({label_sys})")
    ax[1, 0].plot(t, no2, ls, color="tab:orange", label=f"net NO2 produced ({label_sys})")
    ax[1, 1].plot(t, tot / tot[0], ls, color="k", label=f"M_total(t) / M_total(0) ({label_sys})")

ax[0, 0].set_title("Total biomass $M_i(t)=\\int u_i\\,dx$"); ax[0, 0].set_xlabel("t")
ax[0, 1].set_title("Cumulative consumption (solver units; compare ratios only)"); ax[0, 1].set_xlabel("t")
ax[1, 0].set_title("Cumulative production (solver units; compare ratios only)"); ax[1, 0].set_xlabel("t")
ax[1, 1].set_title("Relative total biomass"); ax[1, 1].set_xlabel("t"); ax[1, 1].axhline(1, color="grey", lw=0.5)
for a in ax.ravel():
    a.legend(fontsize=7)
i = info["aob"]
fig.suptitle(f"Step 1, {PRESET}: AOB system (solid) vs CMX system (dashed) -- "
             f"NH4_BC={i['NH4_BC']:g}, O2_BC={i['O2_BC']:g}, N={i['N']}, compared up to t={T_CMN:g}", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / f"{PRESET}_comparison.png", dpi=120)
plt.close(fig)
print("saved", HERE / f"{PRESET}_comparison.png")


def fields(sysname):
    res = RES[sysname]
    N = int(np.load(res / "grid_shape.npy")[0])
    shape = (N + 1, N + 1)
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    times = sorted({float(p.name.split("_")[0][1:]) for p in res.glob("t*_u_AOB.npy")})
    times = [t for t in times if t <= T_CMN + 1e-9]
    times = times[:: max(1, len(times) // 6)]
    sp = [s for s in SPECIES if max(hist[sysname]["M"][s]) > 0]
    rows = [("u", s) for s in sp] + [("c", s) for s in SUBSTRATES]
    fig, axes = plt.subplots(len(rows), len(times), figsize=(2.0 * len(times), 2.0 * len(rows)), squeeze=False)
    for i, (kind, name) in enumerate(rows):
        data = [np.load(res / f"t{t:g}_{kind}_{name}.npy").reshape(shape) for t in times]
        vmax = max(float(d.max()) for d in data) or 1.0
        levels = np.linspace(0, vmax, 11)
        cmap = CMAP[name] if kind == "u" else CMAP_C
        for j, (t, d) in enumerate(zip(times, data)):
            a = axes[i, j]
            cf = a.contourf(X, Y, d, levels=levels, cmap=cmap, extend="max")
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(f"t={t:g}", fontsize=9)
            if j == 0:
                a.set_ylabel(f"{name}\n(peak {vmax:.3g})", fontsize=8)
        fig.colorbar(cf, ax=axes[i, :].tolist(), fraction=0.012, pad=0.01)
    label = "AOB system (AOB+NOB)" if sysname == "aob" else "CMX system"
    fig.suptitle(f"Step 1, {PRESET}: {label}", fontsize=12)
    out = HERE / f"{PRESET}_{sysname}_fields.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


fields("aob")
fields("cmx")
