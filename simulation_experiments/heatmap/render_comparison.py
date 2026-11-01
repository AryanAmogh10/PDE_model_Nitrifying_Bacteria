"""Spatial comparison of the first run to stop (top right of the heatmap,
NH4 1000 uM, O2 250 uM) with the bottom-left corner (NH4 0.5 uM, O2 5 uM).
The stop time T* (47.5 units) is not a snapshot time, so the state at T*
is recomputed for both runs from their stored t = 45 fields (5 steps).
  comparison_progression.png   biomass composition from t = 0 to T*
  comparison_at_stop.png       each species and NH4, O2 at T*
    python render_comparison.py [T_max of the sweep, default 150]"""
import sys, json, pickle, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from run_heatmap import results_dir, point_file, PRESET, DT
import common as c
from common import SPECIES, SUBSTRATES

HERE = Path(__file__).parent
T_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
N = 99
CASES = [((8, 6), "top right (first to stop)"), ((0, 0), "bottom left")]
COL = {"AOB": (0.10, 0.60, 0.20), "CMX": (0.50, 0.10, 0.60), "NOB": (0.95, 0.50, 0.10)}
u = c.Units(PRESET)
L = u.L_um


def fields_dir(i, j):
    return results_dir("all", N, T_MAX) / f"fields_nh4_{i}_o2_{j}"


def load(i, j, t):
    d = fields_dir(i, j)
    U = {s: np.load(d / f"t{t:g}_u_{s}.npy").astype(np.float64) for s in SPECIES}
    C = {s: np.load(d / f"t{t:g}_c_{s}.npy").astype(np.float64) for s in SUBSTRATES}
    return U, C


def recompute(i, j, t_from, t_to, rec):
    """Rerun from the stored t_from fields to t_to; returns (U, C) at t_to."""
    hist = json.loads((fields_dir(i, j) / "history.json").read_text())
    k = int(round(t_from / DT))
    for key in ("t", "boundary_mass"):
        hist[key] = hist[key][:k + 1]
    for key in ("M", "stock", "rate"):
        hist[key] = {s: v[:k + 1] for s, v in hist[key].items()}
    hist["newton_iters"] = hist["newton_iters"][:k]
    U, C = load(i, j, t_from)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / "state.ckpt", "wb") as fh:
            pickle.dump({"U": U, "C": C, "hist": hist, "k": k, "elapsed": 0.0}, fh)
        c.run(PRESET, "all", rec["NH4_uM"] / u.c_ref_uM, rec["O2_uM"] / u.c_ref_uM, N=N, T_max=t_to, dt=DT,
              checkpoint=tmp / "state.ckpt", out=tmp)
        return load_from(tmp, t_to)


def load_from(d, t):
    return ({s: np.load(d / f"t{t:g}_u_{s}.npy") for s in SPECIES},
            {s: np.load(d / f"t{t:g}_c_{s}.npy") for s in SUBSTRATES})


def grid2(v):
    return np.asarray(v).reshape(N + 1, N + 1).T   # [x, y] -> image rows = y


def composite(U):
    img = np.ones((N + 1, N + 1, 3))
    for s in SPECIES:
        img -= grid2(U[s])[..., None] * (1.0 - np.array(COL[s]))
    return np.clip(img, 0, 1)


def mass(U, s):
    return float(u.biomass_ug_per_mm(np.sum(c.Grid2D(Nx=N, Ny=N).Vflat * U[s])))


recs = {ij: json.loads(point_file(*ij, "all", N, T_MAX).read_text()) for ij, _ in CASES}
T_STAR = recs[CASES[0][0]]["t_stop"]
t_prev = 5.0 * np.floor(T_STAR / 5.0)
at_stop = {ij: recompute(*ij, t_prev, T_STAR, recs[ij]) for ij, _ in CASES}
times = [0, 10, 20, 30, 40, T_STAR]

fig, ax = plt.subplots(2, len(times), figsize=(3.1 * len(times), 7.0))
for r, (ij, name) in enumerate(CASES):
    rec = recs[ij]
    for k, t in enumerate(times):
        U = at_stop[ij][0] if t == T_STAR else load(*ij, t)[0]
        a = ax[r, k]
        a.imshow(composite(U), origin="lower", extent=[0, L, 0, L])
        a.set_title(f"t = {u.days(t):.1f} days", fontsize=10)
        a.set_xticks([0, L / 2, L]); a.set_yticks([0, L / 2, L])
        if k == 0:
            a.set_ylabel(f"{name}\nNH4 {rec['NH4_uM']:g} uM, O2 {rec['O2_uM']:g} uM\ny [um]", fontsize=9)
        if r == 1:
            a.set_xlabel("x [um]")
fig.legend(handles=[Patch(color=COL[s], label=s) for s in SPECIES], loc="lower center", ncol=3, fontsize=10)
fig.suptitle(f"Biomass composition up to T* = {u.days(T_STAR):.1f} days, when the first run (top right) stops "
             f"({PRESET}, {N + 1}x{N + 1} grid; colour = species, intensity = volume fraction)", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(HERE / "comparison_progression.png", dpi=130); plt.close(fig)
print("saved comparison_progression.png")

cols = [("u", s) for s in SPECIES] + [("c", "NH4"), ("c", "O2")]
fig, ax = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 6.8))
summary = {"T_star_units": T_STAR, "T_star_days": round(float(u.days(T_STAR)), 2), "cases": {}}
for r, (ij, name) in enumerate(CASES):
    U, C = at_stop[ij]
    rec = recs[ij]
    M = {s: mass(U, s) for s in SPECIES}
    M0 = {s: float(u.biomass_ug_per_mm(rec["M"][s][0])) for s in SPECIES}
    summary["cases"][name] = {"NH4_uM": rec["NH4_uM"], "O2_uM": rec["O2_uM"],
                              "biomass_ug_per_mm": {s: round(M[s], 3) for s in SPECIES},
                              "growth": {s: round(M[s] / M0[s], 3) for s in SPECIES},
                              "F_CMX": round(M["CMX"] / (M["AOB"] + M["CMX"]), 3),
                              "min_uM": {s: round(float(u.uM(C[s].min())), 3) for s in ("NH4", "O2")}}
    for k, (kind, s) in enumerate(cols):
        a = ax[r, k]
        if kind == "u":
            im = a.imshow(grid2(U[s]), origin="lower", extent=[0, L, 0, L], cmap="Greys", vmin=0, vmax=1)
            a.set_title(f"{s}: {M[s]:.2f} ug/mm ({M[s] / M0[s]:.2f}x start)", fontsize=9)
            plt.colorbar(im, ax=a, fraction=0.046, label="volume fraction [-]")
        else:
            im = a.imshow(grid2(u.uM(C[s])), origin="lower", extent=[0, L, 0, L], cmap="viridis")
            a.set_title(f"{s} [uM]", fontsize=9)
            plt.colorbar(im, ax=a, fraction=0.046, label="uM")
        a.set_xticks([0, L / 2, L]); a.set_yticks([0, L / 2, L])
        if k == 0:
            a.set_ylabel(f"{name}\nNH4 {rec['NH4_uM']:g} uM, O2 {rec['O2_uM']:g} uM\ny [um]", fontsize=9)
        if r == 1:
            a.set_xlabel("x [um]")
fig.suptitle(f"State at T* = {u.days(T_STAR):.1f} days: F_CMX top right "
             f"{summary['cases'][CASES[0][1]]['F_CMX']:.2f}, bottom left {summary['cases'][CASES[1][1]]['F_CMX']:.2f}", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "comparison_at_stop.png", dpi=130); plt.close(fig)
(HERE / "comparison_summary.json").write_text(json.dumps(summary, indent=1))
print("saved comparison_at_stop.png")
print(json.dumps(summary, indent=1))
