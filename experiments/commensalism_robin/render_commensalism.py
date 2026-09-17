"""Renders bacteria-density and substrate-concentration grids (every 5 steps,
0..50) for one commensalism regime. Usage: python render_commensalism.py {dirichlet,neumann,robin}"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REGIME = "robin"  # hardcoded for this folder
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent

SPECIES = ["u1", "u2", "u3"]
SUBSTRATES = ["c1", "c2", "c3"]
STEPS = list(range(0, 51, 5))
cmap_u1 = LinearSegmentedColormap.from_list("white_to_darkgreen",  ["white", "darkgreen"])
cmap_u2 = LinearSegmentedColormap.from_list("white_to_darkorange", ["white", "darkorange"])
cmap_u3 = LinearSegmentedColormap.from_list("white_to_purple",     ["white", "purple"])
cmap_c  = LinearSegmentedColormap.from_list("white_to_blue",       ["white", "royalblue"])
SPECIES_LABEL = {"u1": "$u_1$", "u2": "$u_2$", "u3": "$u_3$"}
SPECIES_CMAP = {"u1": cmap_u1, "u2": cmap_u2, "u3": cmap_u3}
SUB_LABEL = {"c1": "$c_1$", "c2": "$c_2$", "c3": "$c_3$"}
SUB_CMAP = {"c1": cmap_c, "c2": cmap_c, "c3": cmap_c}

GRID_N = int(np.load(RESULTS_DIR / "grid_shape.npy")[0])
SHAPE = (GRID_N + 1, GRID_N + 1)
x = np.linspace(0, 1, GRID_N + 1)
X, Y = np.meshgrid(x, x, indexing="ij")


def render_grid(field_getter, rows, row_label, row_cmap, out_name, suptitle, per_row_scale=False):
    data = {r: {t: field_getter(r, t) for t in STEPS} for r in rows}
    global_vmax = max(float(data[r][t].max()) for r in rows for t in STEPS)
    fig, axes = plt.subplots(len(rows), len(STEPS), figsize=(2.0 * len(STEPS), 2.1 * len(rows)))
    for i, r in enumerate(rows):
        row_vmax = max(float(data[r][t].max()) for t in STEPS) if per_row_scale else global_vmax
        row_vmax = row_vmax if row_vmax > 0 else 1.0
        levels = np.linspace(0, row_vmax, 11)
        for j, t in enumerate(STEPS):
            ax = axes[i, j]
            field = data[r][t]
            cf = ax.contourf(X, Y, field, levels=levels, cmap=row_cmap[r], extend="max")
            ax.contour(X, Y, field, levels=levels, colors="black", linewidths=0.2, alpha=0.4)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{row_label[r]}\n(peak={row_vmax:.3g})", fontsize=9)
        fig.colorbar(cf, ax=axes[i, :].tolist(), fraction=0.01, pad=0.01)
    fig.suptitle(f"{suptitle} -- {REGIME} regime" + ("" if per_row_scale else f", peak={global_vmax:.3g}"), fontsize=13)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def get_u(sp, t):
    return np.load(RESULTS_DIR / f"t{t}_u_{sp}.npy").reshape(SHAPE)


def get_c(sub, t):
    return np.load(RESULTS_DIR / f"t{t}_c_{sub}.npy").reshape(SHAPE)


render_grid(get_u, SPECIES, SPECIES_LABEL, SPECIES_CMAP,
            f"commensalism_{REGIME}_bacteria.png", "Bacterial densities (commensalism, rough IC)")
render_grid(get_c, SUBSTRATES, SUB_LABEL, SUB_CMAP,
            f"commensalism_{REGIME}_substrates.png", "Substrate concentrations", per_row_scale=True)

bmh = np.load(RESULTS_DIR / "boundary_mass_history.npy")
first_touch = bmh[bmh[:, 1] > 1e-8]
if len(first_touch):
    print(f"{REGIME}: biomass first reaches the domain boundary at slow step {int(first_touch[0,0])}")
else:
    print(f"{REGIME}: biomass never reached the domain boundary within {int(bmh[-1,0])} steps")
