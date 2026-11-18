"""Figures of the AOB-CMX results section (Chapter 8), drawn from sweep_derived.pkl and the stored sweep.
Run analyze_sweep.py first. Figures are written to ./figures."""
import sys, json, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize

SCR = Path(__file__).resolve().parent
OUT = SCR / "figures"
OUT.mkdir(exist_ok=True)
D = pickle.load(open(SCR / "sweep_derived.pkl", "rb"))
NH4, O2 = D["NH4_um"], D["O2_um"]
NI, NJ = len(NH4), len(O2)
td = D["t_day"]
M = D["M_ug"]; Mc = D["Mc_ug"]
F = D["F"]; G = D["G"]
KEND = len(td) - 1
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "figure.dpi": 200, "savefig.dpi": 220,
                     "axes.linewidth": 0.6})
COL = {"AOB": "#1b6e2f", "NOB": "#d9730d", "CMX": "#7b1fa2"}
D1, D4 = 1.69, 1.81     # diffusion coefficients (1e-4 m2/day) of NH4 and O2

# ----------------------------------------------------------------- helpers
x_nh4 = np.log10(NH4)
def thr_line():
    """Oxygen supply = demand of complete nitrification: D4 c4 = 2 D1 c1, as polyline in index space."""
    th = D4 * O2 / (2 * D1)
    xs = np.interp(np.log10(th), x_nh4, np.arange(NI))
    return xs, np.arange(NJ)

def lab(v, fmt, strip=True, zero=None):
    if zero is not None and abs(v) < zero:
        return "0"
    s = fmt.format(v)
    if s == "1.00":
        return "1"
    if strip and s.startswith("0."):
        s = s[1:]
    elif strip and s.startswith("-0."):
        s = "-" + s[2:]
    return s

from matplotlib.colors import LinearSegmentedColormap
DIVERGING = ("RdBu", "RdBu_r", "PRGn", "PRGn_r")

def wmap(name):
    """Colour map whose lowest value is pure white (diverging maps: pure white in the middle)."""
    base = plt.get_cmap(name)
    if name in DIVERGING:
        cols = [base(x) for x in np.linspace(0, 0.42, 120)] + [(1, 1, 1, 1)] + [base(x) for x in np.linspace(0.58, 1, 120)]
    else:
        xs_ = np.linspace(0, 1, 256)
        w = np.clip(xs_ / 0.35, 0, 1)[:, None]
        cols = (1 - w) * np.ones((256, 4)) + w * np.array([base(x) for x in xs_])
        cols[:, 3] = 1.0
    return LinearSegmentedColormap.from_list(name + "_w", cols, N=256)

def heat(ax, a, title, cmap, norm, fmt="{:.2f}", zero=None, line=False, xlabel=True, ylabel=True, fs=5.8, nanlab="", box=None):
    """a has shape [NI, NJ]; shown with O2 on the vertical and NH4 on the horizontal axis."""
    arr = np.asarray(a, float).T
    shown = np.where(np.isfinite(arr), arr, np.nan)
    if isinstance(norm, LogNorm):
        shown = np.where(np.isfinite(shown), np.maximum(shown, norm.vmin), np.nan)
    cm = wmap(cmap).copy(); cm.set_bad("#e8e8e8")
    im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cm, norm=norm)
    ax.set_xticks(range(NI)); ax.set_xticklabels([("%g" % round(x, 1) if x < 10 else "%d" % round(x)) for x in NH4], rotation=60, ha="right")
    ax.set_yticks(range(NJ)); ax.set_yticklabels(["%d" % y for y in O2])
    if xlabel: ax.set_xlabel("NH$_4$ at the boundary [$\\mu$M]")
    if ylabel: ax.set_ylabel("O$_2$ at the boundary [$\\mu$M]")
    ax.set_title(title)
    rgba = im.cmap(im.norm(shown))
    for j in range(NJ):
        for i in range(NI):
            v = arr[j, i]
            if not np.isfinite(v):
                txt = nanlab; col = "black"
            else:
                lum = 0.299 * rgba[j, i, 0] + 0.587 * rgba[j, i, 1] + 0.114 * rgba[j, i, 2]
                txt = lab(v, fmt, zero=zero); col = "white" if lum < 0.45 else "black"
            ax.text(i, j, txt, ha="center", va="center", fontsize=fs, color=col, zorder=5)
    if line:
        xs, ys = thr_line()
        ax.plot(xs, ys, color="black", lw=1.0, ls=(0, (3, 2)), zorder=2)
    if box is not None:
        ax.add_patch(plt.Rectangle((box[0] - .5, box[1] - .5), 1, 1, fill=False, ec="red", lw=1.4))
    ax.set_xlim(-0.5, NI - 0.5); ax.set_ylim(-0.5, NJ - 0.5)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.ax.tick_params(labelsize=6)
    return im

def fcmx_t(k):
    return M["CMX"][:, :, k] / (M["AOB"][:, :, k] + M["CMX"][:, :, k])

# ----------------------------------------------------------------- R0: F_CMX map with the line (replaces t_sweep_fcmx)
def fig_fcmx():
    h = json.load(open(SCR.parent / "heatmap" / "heatmap_summary_N99_T150.json"))
    Fs = np.array(h["F_CMX"]); stop = np.array(h["stop_days"]); touched = np.array(h["touched_boundary"], dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    arr = Fs.T
    heat(axes[0], Fs.T.T if False else np.asarray(h["F_CMX"]).T, "$F_{\\mathrm{CMX}}$ at the stop time", "Purples", Normalize(0, 1), fmt="{:.2f}")
    # the summary json is stored as [O2, NH4]; heat() expects [NH4, O2]
    plt.close(fig)
    Fs2 = np.asarray(h["F_CMX"]); st2 = np.asarray(h["stop_days"]); to2 = np.asarray(h["touched_boundary"], dtype=bool)
    if Fs2.shape == (NJ, NI):
        Fs2, st2, to2 = Fs2.T, st2.T, to2.T
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    heat(axes[0], Fs2, "$F_{\\mathrm{CMX}}$ at the stop time", "Purples", Normalize(0, 1), fmt="{:.2f}")
    heat(axes[1], st2, "Stop time [days]", "viridis", Normalize(40, 160), fmt="{:.0f}", ylabel=False, strip=False) if False else None
    ax = axes[1]
    heat(ax, st2, "Stop time [days]", "viridis", Normalize(40, 160), fmt="{:.0f}", ylabel=False)
    # stars for runs that never reached the boundary
    for j in range(NJ):
        for i in range(NI):
            if not to2[i, j]:
                axes[0].text(i + 0.38, j + 0.28, "*", fontsize=7, ha="center", va="center", color="white" if Fs2[i, j] < 0.3 or Fs2[i, j] > 0.8 else "black")
    fig.tight_layout()
    fig.savefig(OUT / "r_fcmx_stop.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------- R1: kinetics
def fig_kinetics():
    r = {"AOB": 0.949, "CMX": 0.0692}
    KN = {"AOB": 1.453, "CMX": 0.063}; KO = {"AOB": 3.606, "CMX": 3.13}
    N = np.logspace(-2, 3, 400)
    mu = lambda s, n, o: r[s] * n / (KN[s] + n) * o / (KO[s] + o)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    ax = axes[0]
    for o, ls in ((250, "-"), (5, "--")):
        for s in ("AOB", "CMX"):
            ax.plot(N, mu(s, N, o), color=COL[s], ls=ls, lw=1.3, label=f"{s}, O$_2$ = {o} $\\mu$M")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("NH$_4$ [$\\mu$M]"); ax.set_ylabel("growth rate $r\\,M$ [1/day]"); ax.legend(loc="upper left", frameon=False)
    ax.axvspan(0.5, 1000, color="0.92", zorder=0)
    ax.set_title("(a) growth rates without competition")
    ax = axes[1]
    cross = {}
    for o, col in ((250, "#444444"), (40, "#2a7fbf"), (5, "#c0392b")):
        ratio = mu("CMX", N, o) / mu("AOB", N, o)
        ax.plot(N, ratio, color=col, lw=1.3, label=f"O$_2$ = {o} $\\mu$M", ls=("-" if o != 40 else ":"))
        k = np.argmin(np.abs(np.log(ratio)))
        cross[o] = N[k]
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("NH$_4$ [$\\mu$M]"); ax.set_ylabel("$\\mu_{\\mathrm{CMX}}/\\mu_{\\mathrm{AOB}}$")
    ax.axvspan(0.5, 1000, color="0.92", zorder=0)
    ax.text(2.5, 0.55, "shaded: boundary values\nused in the sweep", fontsize=6, color="0.3")
    ax.legend(frameon=False, loc="lower left"); ax.set_title("(b) ratio CMX to AOB")
    fig.tight_layout(); fig.savefig(OUT / "r_kinetics.png", bbox_inches="tight"); plt.close(fig)
    print("kinetic crossover NH4 (uM):", {k: round(float(v), 3) for k, v in cross.items()})

# ----------------------------------------------------------------- R2: time courses
def fig_timecourses():
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.9), sharex=True)
    cmap = plt.get_cmap("viridis")
    for row, jo in enumerate((6, 2)):
        Fk = M["CMX"][:, jo, :] / (M["AOB"][:, jo, :] + M["CMX"][:, jo, :])
        for i in range(NI):
            col = cmap(i / (NI - 1))
            axes[row, 0].plot(td, Fk[i], color=col, lw=1.1)
            axes[row, 1].plot(td, M["AOB"][i, jo] / M["AOB"][i, jo, 0], color=col, lw=1.1)
            axes[row, 2].plot(td, M["CMX"][i, jo] / M["CMX"][i, jo, 0], color=col, lw=1.1)
        axes[row, 0].axhline(0.5, color="k", lw=0.5, ls=":")
        axes[row, 0].set_ylabel(f"O$_2$ = {O2[jo]:g} $\\mu$M")
        axes[row, 1].axhline(1, color="k", lw=0.5, ls=":")
        axes[row, 2].axhline(1, color="k", lw=0.5, ls=":")
    axes[0, 0].set_title("$F_{\\mathrm{CMX}}$"); axes[0, 1].set_title("AOB mass / initial"); axes[0, 2].set_title("CMX mass / initial")
    for a in axes[1]: a.set_xlabel("time [days]")
    axes[0, 1].set_ylim(0, 3.4); axes[1, 1].set_ylim(0, 3.4)
    axes[0, 2].set_ylim(0.6, 1.1); axes[1, 2].set_ylim(0.6, 1.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, NI - 1))
    cb = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.03, ticks=[0, 2, 4, 6, 8])
    cb.ax.set_yticklabels(["%.2g" % NH4[k] if NH4[k] < 10 else "%d" % round(NH4[k]) for k in (0, 2, 4, 6, 8)])
    cb.set_label("NH$_4$ at the boundary [$\\mu$M]")
    fig.savefig(OUT / "r_timecourses.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------- R3: dependence on the time of observation
def crossing(col):
    for i in range(NI - 1):
        a, b = col[i], col[i + 1]
        if (a - 0.5) * (b - 0.5) < 0:
            return 10 ** (x_nh4[i] + (0.5 - a) / (b - a) * (x_nh4[i + 1] - x_nh4[i]))
    return np.nan

def fig_crossing():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    cm = plt.get_cmap("plasma")
    times = [(19, "10 d"), (38, "20 d"), (57, "30 d"), (76, "40 d"), (95, "50 d")]
    for n, (k, lbl) in enumerate(times):
        ax.plot(NH4, fcmx_t(k)[:, 6], color=cm(n / 4.6), marker="o", ms=2.5, lw=1.2, label=lbl)
    ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xscale("log"); ax.set_ylim(0.1, 1.0)
    ax.set_xlabel("NH$_4$ at the boundary [$\\mu$M]"); ax.set_ylabel("$F_{\\mathrm{CMX}}$"); ax.legend(frameon=False, ncol=2, title="O$_2$ = 250 $\\mu$M, at", title_fontsize=6.5)
    ax.set_title("(a) $F_{\\mathrm{CMX}}$ at fixed times")
    # (b) day on which AOB first outweigh CMX (F_CMX falls below 0.5 after its maximum), full runs to the stop time
    RES = SCR.parent / "heatmap" / "results_all_N99_T150"
    tdom = np.full((NI, NJ), np.nan)
    for i in range(NI):
        for j in range(NJ):
            r = json.loads((RES / f"point_nh4_{i}_o2_{j}.json").read_text())
            ks = r["k_stop"]
            a = np.array(r["M"]["AOB"][:ks + 1]); c_ = np.array(r["M"]["CMX"][:ks + 1])
            f = c_ / (a + c_); km = int(np.argmax(f))
            below = np.where(f[km:] < 0.5)[0]
            if len(below):
                tdom[i, j] = r["t"][km + below[0]] * D["tau_day"]
    np.save(SCR / "tdom.npy", tdom)
    ax = axes[1]
    heat(ax, tdom, "(b) day on which $F_{\\mathrm{CMX}}$ falls below 0.5", "viridis_r", Normalize(10, 150), fmt="{:.0f}", nanlab="-", ylabel=False)
    fig.tight_layout(); fig.savefig(OUT / "r_crossing.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------- R4: growth factors at 50 d
def fig_growth():
    k = KEND
    nrm = TwoSlopeNorm(1.0, 0.0, 3.3)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))
    ga = {s: M[s][:, :, k] / M[s][:, :, 0] for s in M}
    gc = Mc[:, :, k] / Mc[:, :, 0]
    heat(axes[0, 0], ga["AOB"], "(a) AOB, combined", "RdBu", nrm, xlabel=False, fmt="{:.2f}", fs=5.3)
    heat(axes[0, 1], ga["NOB"], "(b) NOB, combined", "RdBu", nrm, xlabel=False, ylabel=False, fmt="{:.2f}", fs=5.3)
    heat(axes[0, 2], ga["CMX"], "(c) CMX, combined", "RdBu", nrm, xlabel=False, ylabel=False, fmt="{:.2f}", fs=5.3)
    heat(axes[1, 0], gc, "(d) CMX, alone", "RdBu", nrm, fmt="{:.2f}", fs=5.3)
    heat(axes[1, 1], ga["CMX"] / gc, "(e) CMX combined / alone", "viridis", Normalize(0, 1), ylabel=False, fmt="{:.2f}", fs=5.3)
    tmin = np.full((NI, NJ), np.nan)
    for i in range(NI):
        for j in range(NJ):
            m = M["AOB"][i, j]; km = int(np.argmin(m))
            if 0 < km < KEND: tmin[i, j] = td[km]
    heat(axes[1, 2], tmin, "(f) day of lowest AOB mass", "viridis_r", Normalize(0, 40), ylabel=False, fmt="{:.0f}", nanlab="-")
    fig.tight_layout(); fig.savefig(OUT / "r_growth.png", bbox_inches="tight"); plt.close(fig)
    return ga, gc, tmin

# ----------------------------------------------------------------- R5: limiting substrates
def fig_limit():
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))
    for c, sp in enumerate(("AOB", "NOB", "CMX")):
        heat(axes[0, c], F[sp]["fN"], f"{sp}: ammonium factor" if sp != "NOB" else "NOB: nitrite factor", "YlGn", Normalize(0, 1), xlabel=False, ylabel=(c == 0), fmt="{:.2f}", zero=0.005)
        heat(axes[1, c], F[sp]["fO"], f"{sp}: oxygen factor", "YlGn", Normalize(0, 1), ylabel=(c == 0), fmt="{:.2f}", zero=0.005)
    fig.tight_layout(); fig.savefig(OUT / "r_limit.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------- R6: uptake
def fig_uptake():
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))
    heat(axes[0, 0], G["share_NH4_CMX"], "(a) CMX share of NH$_4$ uptake", "Purples", Normalize(0, 0.5), xlabel=False, fmt="{:.2f}", zero=0.005)
    heat(axes[0, 1], G["spec_NH4_AOB"] * 1e3, "(b) NH$_4$ per AOB biomass", "YlOrBr", LogNorm(0.05, 60), xlabel=False, ylabel=False, fmt="{:.2g}", zero=0.01)
    heat(axes[0, 2], G["spec_NH4_CMX"] * 1e3, "(c) NH$_4$ per CMX biomass", "YlOrBr", LogNorm(0.05, 60), xlabel=False, ylabel=False, fmt="{:.1g}", zero=0.005)
    heat(axes[1, 0], G["O2share_AOB"], "(d) AOB share of O$_2$ uptake", "Greens", Normalize(0, 1), fmt="{:.2f}", zero=0.005)
    heat(axes[1, 1], G["O2share_NOB"], "(e) NOB share of O$_2$ uptake", "Oranges", Normalize(0, 0.5), ylabel=False, fmt="{:.2f}", zero=0.005)
    heat(axes[1, 2], F["AOB"]["both"], "(f) AOB biomass with NH$_4$ and O$_2$", "Greens", Normalize(0, 0.3), ylabel=False, fmt="{:.2f}", zero=0.005)
    fig.tight_layout(); fig.savefig(OUT / "r_uptake.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------- R7: profiles
def fig_profiles():
    P = D["profiles"]
    cases = [((4, 6), "(a)"), ((6, 6), "(b)"), ((6, 2), "(c)"), ((8, 6), "(d)")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharex=True)
    for ax, ((i, j), tag) in zip(axes.ravel(), cases):
        p = P[(i, j)]
        for s in ("AOB", "NOB", "CMX"):
            ax.plot(p["x_um"], p["u"][s], color=COL[s], lw=1.3, label=s)
        ax.set_ylim(0, 1.0 if p["u"]["AOB"].max() > 0.5 else 0.5)
        ax.set_title(f"{tag} NH$_4$ = {(NH4[i] if NH4[i] < 10 else round(NH4[i])):g} $\\mu$M, O$_2$ = {O2[j]:g} $\\mu$M")
        ax2 = ax.twinx()
        ax2.plot(p["x_um"], p["c"]["NH4"] / NH4[i], color="#2a7fbf", lw=1.0, ls="--", label="NH$_4$")
        ax2.plot(p["x_um"], p["c"]["O2"] / O2[j], color="#c0392b", lw=1.0, ls="--", label="O$_2$")
        ax2.set_ylim(0, 1.05)
        if ax in axes[:, 1]: ax2.set_ylabel("concentration / boundary value")
        else: ax2.set_yticklabels([])
        if ax in axes[:, 0]: ax.set_ylabel("volume fraction")
        if ax in axes[1]: ax.set_xlabel("position along the middle line [$\\mu$m]")
        if ax is axes[0, 0]:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.savefig(OUT / "r_profiles.png", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    what = sys.argv[1:] or ["fcmx", "kin", "time", "cross", "growth", "limit", "uptake", "prof"]
    if "fcmx" in what: fig_fcmx()
    if "kin" in what: fig_kinetics()
    if "time" in what: fig_timecourses()
    if "cross" in what: fig_crossing()
    if "growth" in what: fig_growth()
    if "limit" in what: fig_limit()
    if "uptake" in what: fig_uptake()
    if "prof" in what: fig_profiles()
    print("done", what)
