"""Plots and a summary table for the base case, in physical units."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import common as c

PRESET = "eloi"
u = c.Units(PRESET)
SYSTEMS = {"all": "combined (AOB+NOB+CMX)", "aob": "AOB system (AOB+NOB)", "cmx": "CMX system"}
STYLE = {"all": "-", "aob": "--", "cmx": ":"}
COLOR = {"AOB": "darkgreen", "NOB": "darkorange", "CMX": "purple"}
CMAP = {"AOB": LinearSegmentedColormap.from_list("w_g", ["white", "darkgreen"]),
        "NOB": LinearSegmentedColormap.from_list("w_o", ["white", "darkorange"]),
        "CMX": LinearSegmentedColormap.from_list("w_p", ["white", "purple"])}
CMAP_C = LinearSegmentedColormap.from_list("w_b", ["white", "royalblue"])

hist, info, last = {}, {}, {}
for s in SYSTEMS:
    hist[s] = json.loads((HERE / "results" / s / "history.json").read_text())
    info[s] = json.loads((HERE / "results" / s / "run_info.json").read_text())
    last[s] = c.last_valid_index(hist[s], info[s])
# compare all systems up to the last time before ANY of them touches the boundary
T_CMN = min(hist[s]["t"][last[s]] for s in SYSTEMS)


def series(s, key, sub=None):
    t = np.asarray(hist[s]["t"])
    keep = t <= T_CMN + 1e-9
    if key == "M":
        return u.days(t[keep]), u.biomass_ug_per_mm(np.asarray(hist[s]["M"][sub])[keep])
    cum = c.cumulative(hist[s], sub)
    sign = -1.0 if key == "consumed" else 1.0
    return u.days(t[keep]), u.amount_umol_per_mm(sub, sign * cum[keep])


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for s, label in SYSTEMS.items():
    for sp in ("AOB", "NOB", "CMX"):
        if hist[s]["M"][sp][0] > 0:
            ax[0, 0].plot(*series(s, "M", sp), STYLE[s], color=COLOR[sp], label=f"{sp}, {label}")
ax[0, 0].set_title("Biomass of each species"); ax[0, 0].set_ylabel("biomass [ug per mm depth]")

for sp in ("AOB", "CMX"):
    for s in ("all", "aob" if sp == "AOB" else "cmx"):
        ax[0, 1].plot(*series(s, "M", sp), STYLE[s], color=COLOR[sp], label=f"{sp}, {SYSTEMS[s]}")
ax[0, 1].set_title("AOB and CMX: isolated vs combined"); ax[0, 1].set_ylabel("biomass [ug per mm depth]")

t_all = np.asarray(hist["all"]["t"]); keep = t_all <= T_CMN + 1e-9
F = [c.f_cmx(hist["all"], i) for i in range(len(t_all))]
ax[0, 2].plot(u.days(t_all[keep]), np.asarray(F)[keep], color="purple")
ax[0, 2].axhline(0.5, color="grey", lw=0.6, ls="--")
ax[0, 2].set_ylim(0, 1); ax[0, 2].set_title("Combined system: F_CMX = M_CMX / (M_AOB + M_CMX)")
ax[0, 2].set_ylabel("F_CMX [-]")

for s, label in SYSTEMS.items():
    ax[1, 0].plot(*series(s, "produced", "NO3"), STYLE[s], color="k", label=label)
    ax[1, 1].plot(*series(s, "consumed", "NH4"), STYLE[s], color="tab:blue", label=f"NH4, {label}")
    ax[1, 1].plot(*series(s, "consumed", "O2"), STYLE[s], color="tab:red", label=f"O2, {label}")
    ax[1, 2].plot(*series(s, "produced", "NO2"), STYLE[s], color="tab:orange", label=label)
ax[1, 0].set_title("Cumulative NO3 produced"); ax[1, 0].set_ylabel("NO3 [umol per mm depth]")
ax[1, 1].set_title("Cumulative NH4 and O2 consumed"); ax[1, 1].set_ylabel("amount [umol per mm depth]")
ax[1, 2].set_title("Cumulative net NO2 produced (leaves through the boundary)"); ax[1, 2].set_ylabel("NO2 [umol per mm depth]")
for a in ax.ravel():
    a.set_xlabel("time [days]")
    if a is not ax[0, 2]:
        a.legend(fontsize=7)
i0 = info["all"]
fig.suptitle(f"Base case ({PRESET}): NH4_BC = {u.uM(i0['NH4_BC']):.0f} uM, O2_BC = {u.uM(i0['O2_BC']):.0f} uM, "
             f"{i0['N'] + 1}x{i0['N'] + 1} grid, compared up to {u.days(T_CMN):.1f} days (before any system reaches the boundary)",
             fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "base_case_comparison.png", dpi=120)
plt.close(fig)
print("saved", HERE / "base_case_comparison.png")


def fields(s):
    res = HERE / "results" / s
    N = int(np.load(res / "grid_shape.npy")[0])
    shape = (N + 1, N + 1)
    x_um = np.linspace(0, u.L_um, N + 1)
    X, Y = np.meshgrid(x_um, x_um, indexing="ij")
    times = sorted({float(p.name.split("_")[0][1:]) for p in res.glob("t*_u_AOB.npy")})
    times = [t for t in times if t <= T_CMN + 1e-9]
    times = times[:: max(1, len(times) // 6)]
    sp = [k for k in ("AOB", "NOB", "CMX") if hist[s]["M"][k][0] > 0]
    rows = [("u", k) for k in sp] + [("c", k) for k in ("NH4", "NO2", "NO3", "O2")]
    fig, axes = plt.subplots(len(rows), len(times), figsize=(2.1 * len(times), 2.0 * len(rows)), squeeze=False)
    for i, (kind, name) in enumerate(rows):
        conv = (lambda a: a * c.BIOFILM_DENSITY_KG_M3) if kind == "u" else u.uM
        unit = "kg/m^3" if kind == "u" else "uM"
        data = [conv(np.load(res / f"t{t:g}_{kind}_{name}.npy").reshape(shape)) for t in times]
        vmax = max(float(d.max()) for d in data) or 1.0
        levels = np.linspace(0, vmax, 11)
        for j, (t, d) in enumerate(zip(times, data)):
            a = axes[i, j]
            cf = a.contourf(X, Y, d, levels=levels, cmap=CMAP[name] if kind == "u" else CMAP_C, extend="max")
            a.set_aspect("equal")
            a.set_xticks([0, u.L_um]); a.set_yticks([0, u.L_um])
            a.tick_params(labelsize=6)
            if i == 0:
                a.set_title(f"{u.days(t):.1f} days", fontsize=9)
            if j == 0:
                a.set_ylabel(f"{name} [{unit}]\ny [um]", fontsize=8)
            if i == len(rows) - 1:
                a.set_xlabel("x [um]", fontsize=8)
        fig.colorbar(cf, ax=axes[i, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"Base case ({PRESET}): {SYSTEMS[s]}; biomass as density [kg/m^3], substrates [uM]", fontsize=12)
    out = HERE / f"base_case_{s}_fields.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


for s in SYSTEMS:
    fields(s)

# summary table at T_CMN
rows = []
for s, label in SYSTEMS.items():
    t = np.asarray(hist[s]["t"]); i = int(np.argmin(np.abs(t - T_CMN)))
    M = {k: u.biomass_ug_per_mm(hist[s]["M"][k][i]) for k in ("AOB", "NOB", "CMX")}
    M0 = {k: u.biomass_ug_per_mm(hist[s]["M"][k][0]) for k in ("AOB", "NOB", "CMX")}
    no3 = u.amount_umol_per_mm("NO3", c.cumulative(hist[s], "NO3")[i])
    no2 = u.amount_umol_per_mm("NO2", c.cumulative(hist[s], "NO2")[i])
    nh4 = u.amount_umol_per_mm("NH4", -c.cumulative(hist[s], "NH4")[i])
    o2 = u.amount_umol_per_mm("O2", -c.cumulative(hist[s], "O2")[i])
    rows.append({"system": label, "M0": M0, "M": M, "NO3": no3, "NO2": no2, "NH4": nh4, "O2": o2,
                 "F_CMX": c.f_cmx(hist[s], i) if s == "all" else None,
                 "stalled": info[s]["stalled"], "touch_days": None if info[s]["boundary_touch_t"] is None else float(u.days(info[s]["boundary_touch_t"]))})
summary = {"T_compare_days": float(u.days(T_CMN)), "rows": rows}
(HERE / "base_case_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=1, default=float))
