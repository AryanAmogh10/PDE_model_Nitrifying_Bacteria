"""Numerical against exact concentration profile and the error along the profile (Chapter 7, Dirichlet, kappa = 2, N = 25)."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODES))
OUT = Path(r"C:\Users\amogh\OneDrive\Desktop\Thesis\Overleaf_export\figures")
spec = importlib.util.spec_from_file_location("tcf", CODES / "tests/test_elliptic_closed_form.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from nitrifiers.one_dimensional.elliptic import Grid, solve_newton
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "figure.dpi": 200, "savefig.dpi": 220})
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
for (geometry, p, lab), col in zip([("slab", 0, "slab"), ("radial", 1, "cylinder"), ("radial", 2, "sphere")], ("#1f77b4", "#ff7f0e", "#2ca02c")):
    grid = Grid(N=25, geometry=geometry, p=p)
    kappa, K, u0, cinf = 2.0, 1.0e7, 1.0, 1.0
    coeffs = T._single_species_coeffs(kappa, K, u0, cinf)
    U = {"AOB": np.full(grid.N + 1, u0), "NOB": np.zeros(grid.N + 1), "CMX": np.zeros(grid.N + 1)}
    C, _, _ = solve_newton(coeffs, U, grid, bc_type="dirichlet", maxiter=100, tol=1e-9)
    ex = T._closed_form(grid.r, kappa, cinf, p)
    fine = np.linspace(0, 1, 300)
    axes[0].plot(fine, T._closed_form(fine, kappa, cinf, p), "-", color=col, lw=1, label=f"exact, {lab}")
    axes[0].plot(grid.r, C["NH4"], "o", color=col, ms=2.5)
    axes[1].plot(grid.r, C["NH4"] - ex, "-o", color=col, ms=2.5, lw=0.8, label=lab)
axes[0].set_xlabel("position $s$"); axes[0].set_ylabel("concentration $c/c^\\infty$"); axes[0].set_title("(a) numerical (dots) and exact (lines)")
axes[0].legend(frameon=False, fontsize=6.5)
axes[1].set_xlabel("position $s$"); axes[1].set_ylabel("numerical $-$ exact"); axes[1].set_title("(b) error along the profile"); axes[1].axhline(0, color="k", lw=0.5)
fig.tight_layout(); fig.savefig(OUT / "t_profile_error.png", bbox_inches="tight")
