"""
ITEM 2: quantitative reproduction of the paper's Section 4.3 travelling-wave
benchmark (arXiv:2512.13156).

System (3.12)-(3.13), the parameter-symmetric competition model reduced to
TOTAL biomass rho and a single substrate c:

    d(rho)/dt = d/dx[(d + a*rho) d(rho)/dx] + rho*(r*c/(K+c) - b*rho)
    d(c)/dt   = D * d2c/dx2 - rho*(r/Y)*c/(K+c)

on x in [0, L=200], t in [0, T=200], homogeneous NEUMANN (zero-flux) boundary
conditions for BOTH fields at both ends (paper Section 4.3: "we choose
initial conditions rho_0(x)=exp(-x) and c_0=5 ... with homogeneous Neumann
boundary conditions" -- distinct from the infinite-domain asymptotic
conditions (3.17) used only for the FORMAL travelling-wave analysis).
All other parameters: Table 1, Case (A): d=1e-6, a=1e-5, r=1, K=1, b=0.1,
Y=0.2, D=1e-4.

This is intentionally a SEPARATE, self-contained script rather than a new
Stage in the main package: this reduced system requires a genuinely
TRANSIENT two-field solve (c does NOT reach quasi-steady state on the
timescale of the wave -- the whole point of the wave is that consumption and
diffusion of c compete on comparable timescales), which nothing in this
codebase's QSSA-based pipeline (Stage 3 solves c via quasi-steady elliptic
reduction) provides. The paper's OWN numerical method is explicit method-of-
lines with scipy.integrate.solve_ivp's RK45 -- reproduced here directly
(rather than adapted to this project's usual finite-volume + Newton style),
because using a DIFFERENT transient scheme for this specific benchmark would
confound "does this project's numerics reproduce the paper" with "does some
other transient scheme also reproduce the paper", which is not the question
being asked.

Spatial discretisation still reuses this project's own validated
conservative finite-volume operators (elliptic.build_laplacian,
parabolic.build_advection_matrix) via a change of variables x_hat = x/L,
x_hat in [0,1] (matching elliptic.Grid's [0,1] convention), which rescales
the transport coefficients by 1/L^2 and leaves the (no-derivative) reaction
terms f, g unchanged -- an exact linear rescaling, not an approximation.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.integrate import solve_ivp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.elliptic import Grid, build_laplacian
from nitrifiers.parabolic import build_advection_matrix


# ---- Table 1, Case (A) parameters -----------------------------------------
d_phys, a_phys, r, K, b, Y, D_phys = 1e-6, 1e-5, 1.0, 1.0, 0.1, 0.2, 1e-4
L = 200.0
T = 200.0

d_eff = d_phys / L ** 2
a_eff = a_phys / L ** 2
D_eff = D_phys / L ** 2


def f_reaction(rho, c):
    return rho * (r * c / (K + c) - b * rho)


def g_reaction(rho, c):
    return rho * (r / Y) * (c / (K + c))


def run(Nx: int, t_eval: np.ndarray, rtol=1e-8, atol=1e-10, method="RK45"):
    grid = Grid(N=Nx, geometry="slab", p=0)
    Lap = build_laplacian(grid)          # dimensionless, zero-flux at BOTH ends
    Npts = Nx + 1
    x_phys = grid.r * L   # Grid.r is x_hat in [0,1] regardless of geometry name

    rho0 = np.exp(-x_phys)
    c0 = np.full(Npts, 5.0)
    y0 = np.concatenate([rho0, c0])

    def rhs(t, y):
        rho = y[:Npts]
        c = y[Npts:]
        Adv = build_advection_matrix(grid, rho)   # dimensionless x_hat operator
        drho = d_eff * (Lap @ rho) + a_eff * (Adv @ rho) + f_reaction(rho, c)
        dc = D_eff * (Lap @ c) - g_reaction(rho, c)
        return np.concatenate([drho, dc])

    t0 = time.time()
    sol = solve_ivp(rhs, (0.0, T), y0, method=method, t_eval=t_eval,
                     rtol=rtol, atol=atol)
    dt = time.time() - t0
    return grid, x_phys, sol, dt


if __name__ == "__main__":
    print("=" * 78)
    print("ITEM 2: travelling-wave speed reproduction, paper Section 4.3")
    print("=" * 78)

    Nx = 2000
    t_eval = np.array([0.0, 160.0, 200.0])
    print(f"\nNx={Nx} (dx_hat={1.0/Nx:.2e}, dx_phys={L/Nx:.4f}), "
          f"t_eval={list(t_eval)}, method=RK45 (matching the paper)")

    grid, x_phys, sol, dt = run(Nx, t_eval)
    print(f"solve_ivp finished in {dt:.1f}s, success={sol.success}, "
          f"nfev={sol.nfev}, status={sol.status}")
    if not sol.success:
        print(f"  message: {sol.message}")

    Npts = Nx + 1
    rho_160 = sol.y[:Npts, 1]
    rho_200 = sol.y[:Npts, 2]
    c_160 = sol.y[Npts:, 1]
    c_200 = sol.y[Npts:, 2]

    neg_rho = min(rho_160.min(), rho_200.min())
    neg_c = min(c_160.min(), c_200.min())
    print(f"\nNon-negativity check (explicit RK45, no clipping applied -- "
          f"matching the paper's own method):")
    print(f"  min(rho) over t=160,200: {neg_rho:.3e}")
    print(f"  min(c)   over t=160,200: {neg_c:.3e}")

    x_bar_160 = x_phys[np.argmax(rho_160)]
    x_bar_200 = x_phys[np.argmax(rho_200)]
    v_bar = (x_bar_200 - x_bar_160) / (200.0 - 160.0)

    print(f"\nargmax(rho) at t=160: x={x_bar_160:.4f}  (rho_max={rho_160.max():.4f})")
    print(f"argmax(rho) at t=200: x={x_bar_200:.4f}  (rho_max={rho_200.max():.4f})")
    print(f"\nMEASURED v_bar = (x_bar(200) - x_bar(160)) / 40 = {v_bar:.6f}")
    print(f"PAPER reports  v_bar ~= 0.8396")
    print(f"relative difference: {abs(v_bar - 0.8396) / 0.8396:.2%}")

    v_min = 2 * np.sqrt(d_phys * r * 5.0 / (K + 5.0))
    print(f"\nClosed-form v_min = {v_min:.6f}  (paper reports 0.0018)")
    print(f"measured v_bar / v_min = {v_bar / v_min:.1f}x "
          f"(paper's own ratio: 0.8396/0.0018 = {0.8396/0.0018:.1f}x)")
