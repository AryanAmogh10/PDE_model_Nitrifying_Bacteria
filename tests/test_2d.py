"""
Validation suite for the 2D Cartesian extension (grid2d / elliptic2d /
parabolic2d / slowfast2d).

The 2D solvers reuse the 1D reaction physics verbatim, so what needs
independent proof here is the SPATIAL machinery: the conservation structure of
the 2D operators, and agreement with the already-validated 1D solvers in the
two limits where 2D must reduce to 1D exactly or convergently.

Two ground-truth reductions are used, in increasing order of weakness:

  1. SLAB (exact). With zero-flux on both y faces and a y-independent
     bacterial field, the y-Laplacian contributes identically zero and the 2D
     problem collapses onto the 1D p=0 stencil. The two solutions must agree to
     SOLVER tolerance, not merely to discretisation error -- this pins down the
     Laplacian, the boundary bookkeeping, the reaction assembly and the Jacobian
     assembly all at once, and would catch any 2D analogue of the 1D row-0 bug.

  2. RADIAL (convergent). 2D Cartesian radial symmetry gives
     Lap = c_rr + (1/r) c_r, i.e. the CYLINDRICAL p=1 operator -- NOT the
     spherical p=2 one; comparing against p=2 would be a setup error, not a
     solver failure. Because a circle cannot be represented exactly on a
     Cartesian grid, the embedded Dirichlet boundary is staircased and
     contributes O(h) geometric error near it. The check is therefore split by
     region: the deep interior must converge at SECOND order to the 1D
     reference, while the near-boundary band is expected to be first order.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.nondim import elliptic_coefficients, SUBSTRATES
from nitrifiers.elliptic import Grid, solve_newton
from nitrifiers.grid2d import Grid2D, build_laplacian_2d
from nitrifiers.elliptic2d import solve_newton_2d
from nitrifiers.parabolic import SPECIES, growth_rate_field
from nitrifiers.parabolic2d import (build_advection_matrix_2d,
                                     build_advection_rho_jacobian_2d,
                                     solve_parabolic_2d, total_mass_2d)


# ---------------------------------------------------------------- grid2d ---

def test_control_volumes_are_exact():
    """sum of control volumes must equal the domain area exactly (the 2D
    analogue of the 1D exact-cell-volume fix; an approximate volume here would
    reintroduce that class of bug)."""
    for Nx, Ny, Lx, Ly in [(20, 20, 1.0, 1.0), (33, 17, 2.0, 0.5), (40, 25, 1.0, 1.0)]:
        g = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        assert abs(float(np.sum(g.V)) - Lx * Ly) < 1e-12


def test_laplacian_2d_is_exactly_mass_conservative():
    """Telescoping-flux identity: sum_ij V_ij*(Lap@c)_ij = 0 for ANY field, at
    ANY resolution. Holds exactly (not asymptotically), so random noise is a
    legitimate and strong test field."""
    rng = np.random.default_rng(0)
    for Nx, Ny, Lx, Ly in [(20, 20, 1.0, 1.0), (33, 17, 2.0, 0.5), (40, 25, 1.0, 1.0)]:
        g = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        Lap = build_laplacian_2d(g)
        for c in (np.sin(np.pi * g.X / Lx) * np.cos(2 * np.pi * g.Y / Ly),
                  rng.random(g.shape)):
            assert abs(float(np.sum(g.Vflat * (Lap @ c.ravel())))) < 1e-11


def test_laplacian_2d_is_second_order_accurate_and_isotropic():
    """Against an exactly radially symmetric analytic function, away from the
    edge rows. Total error must be O(h^2); the ANGULAR (anisotropic) part of
    the 5-point stencil's truncation error must not be worse than that."""
    errs, angs = [], []
    for Nc in (40, 80, 160):
        g = Grid2D(Nx=Nc, Ny=Nc, Lx=2.0, Ly=2.0)
        d = g.radius()
        num = (build_laplacian_2d(g) @ np.exp(-d ** 2).ravel()).reshape(g.shape)
        exact = (4 * d ** 2 - 4) * np.exp(-d ** 2)
        m = np.zeros(g.shape, bool)
        m[2:-2, 2:-2] = True
        errs.append(float(np.abs(num - exact)[m].max()))
        dd, ee, bw = d[m], (num - exact)[m], 3 * g.hx
        a = 0.0
        for lo in np.arange(0.2, 0.8, bw):
            sel = (dd >= lo) & (dd < lo + bw)
            if sel.sum() > 8:
                a = max(a, float(ee[sel].max() - ee[sel].min()))
        angs.append(a)
    for k in range(len(errs) - 1):
        assert errs[k] / errs[k + 1] > 3.7, errs      # ~4x => 2nd order
        assert angs[k] / angs[k + 1] > 3.7, angs      # anisotropy no worse


# ------------------------------------------------------------ elliptic2d ---

def _slab_pair(N, preset="eloi"):
    coeffs = elliptic_coefficients(preset)
    g1 = Grid(N=N, geometry="slab", p=0)
    u1 = 0.05 * np.exp(-((g1.r - 0.5) / 0.2) ** 2)
    C1, h1, _ = solve_newton(coeffs, {s: u1.copy() for s in SPECIES}, g1,
                              bc_type="dirichlet", maxiter=200, tol=1e-11)
    g2 = Grid2D(Nx=N, Ny=max(4, N // 4))
    U2 = {s: np.repeat(u1[:, None], g2.Ny + 1, axis=1).ravel() for s in SPECIES}
    bn = np.flatnonzero((np.arange(g2.Npts) // (g2.Ny + 1)) == g2.Nx)
    C2, h2, _ = solve_newton_2d(coeffs, U2, g2, bc_type="dirichlet",
                                 boundary_nodes=bn, maxiter=200, tol=1e-11)
    return g2, C1, C2


def test_2d_reduces_exactly_to_1d_slab():
    """The strongest available check: an EXACT reduction, so agreement is to
    solver tolerance rather than discretisation error."""
    for N in (20, 40):
        g2, C1, C2 = _slab_pair(N)
        for sub in SUBSTRATES:
            f2 = C2[sub].reshape(g2.shape)
            assert np.max(np.abs(f2 - C1[sub][:, None])) < 1e-10, (N, sub)
            assert np.max(np.abs(f2 - f2[:, [0]])) < 1e-10, (N, sub)  # y-independent


def test_2d_reduces_to_1d_cylindrical_second_order_in_interior():
    """Radial-symmetry reduction against the validated 1D p=1 solver. Split by
    region: the deep interior is free of the staircase geometry error and must
    converge at 2nd order; the near-boundary band is O(h) by construction and
    is checked only for monotone improvement."""
    coeffs = elliptic_coefficients("eloi")
    R = 1.0
    g1 = Grid(N=400, geometry="radial", p=1)
    r1 = g1.r
    u1 = 0.05 * np.exp(-((r1 - 0.5) / 0.25) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C1, _, _ = solve_newton(coeffs, {s: u1.copy() for s in SPECIES}, g1,
                                 bc_type="dirichlet", maxiter=200, tol=1e-11)

    deep, near = [], []
    for Nc in (40, 80):
        g2 = Grid2D(Nx=Nc, Ny=Nc, Lx=2.0, Ly=2.0)
        d = g2.radius()
        out = d >= R
        u2 = np.where(out, 0.0, np.interp(np.clip(d, 0, R), r1, u1))
        C2, _, _ = solve_newton_2d(coeffs, {s: u2.ravel().copy() for s in SPECIES},
                                    g2, bc_type="dirichlet",
                                    boundary_nodes=np.flatnonzero(out.ravel()),
                                    maxiter=200, tol=1e-10)
        ins = ~out
        dd = d[ins]
        e = C2["NH4"].reshape(g2.shape)[ins] - np.interp(dd, r1, C1["NH4"])
        deep.append(float(np.max(np.abs(e[dd < 0.7 * R]))))
        near.append(float(np.max(np.abs(e[dd >= 0.9 * R]))))
    assert deep[0] / deep[1] > 3.5, deep     # 2nd order in the clean interior
    assert deep[-1] < 1e-6, deep
    assert near[0] / near[1] > 1.5, near     # 1st order near the staircase


# ----------------------------------------------------------- parabolic2d ---

def test_advection_2d_is_exactly_mass_conservative():
    """Stronger than the 1D analogue, which only converges toward conservation
    because it normalises by the approximate volume r^p*h. In 2D Cartesian the
    control volume is exact, so this holds to machine precision."""
    for Nc in (20, 40):
        g = Grid2D(Nx=Nc, Ny=Nc)
        rho = (np.cos(np.pi * g.X) * np.cos(np.pi * g.Y) + 1.5).ravel()
        u = (0.3 + 0.2 * np.sin(2 * np.pi * g.X) * np.sin(np.pi * g.Y)).ravel()
        A = build_advection_matrix_2d(g, rho)
        assert abs(float(np.sum(g.Vflat * (A @ u)))) < 1e-12


def test_advection_rho_jacobian_2d_matches_finite_difference():
    """d/drho of Adv(rho)@u_i -- the term whose omission left the 1D Newton
    only 'modified' and stalling on sharp fronts."""
    g = Grid2D(Nx=12, Ny=10)
    rng = np.random.default_rng(1)
    u_i = rng.random(g.Npts) + 0.1
    rho = np.sort(rng.random(g.Npts)) * 2.0
    M = build_advection_rho_jacobian_2d(g, u_i, rho).toarray()
    T0 = build_advection_matrix_2d(g, rho) @ u_i
    eps = 1e-7
    Mfd = np.zeros_like(M)
    for n in range(g.Npts):
        rp = rho.copy()
        rp[n] += eps
        Mfd[:, n] = (build_advection_matrix_2d(g, rp) @ u_i - T0) / eps
    assert np.max(np.abs(M - Mfd)) / np.max(np.abs(Mfd)) < 1e-6


def test_parabolic_2d_reaches_analytic_equilibrium():
    """Uniform ample substrate + uniform seed reduces every node to the
    logistic ODE, fixed point rho* = g_i/bhat_i. Also checks no spurious
    spatial symmetry breaking."""
    coeffs = elliptic_coefficients("toy")
    g = Grid2D(Nx=16, Ny=16)
    C = {s: np.full(g.Npts, 10.0) for s in SUBSTRATES}
    U0 = {s: np.full(g.Npts, 0.05) for s in SPECIES}
    rho_star = growth_rate_field(coeffs, C, "AOB")[0] / coeffs["bhat"]["AOB"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, mh = solve_parabolic_2d(coeffs, C, U0, g, dt=0.5, n_steps=200)
    rho = U["AOB"] + U["NOB"] + U["CMX"]
    assert np.max(np.abs(rho - rho_star)) < 1e-4 * rho_star
    assert np.ptp(rho) < 1e-10
    assert all(b >= a - 1e-12 for a, b in zip(mh, mh[1:]))


def test_2d_per_substrate_bc_specs_mixed_dirichlet_neumann():
    """ITEM 1 extension: solve_newton_2d now accepts bc_specs, the same
    per-substrate {sub: (bc_type, value)} interface as the 1D solver, for
    consistency (see elliptic2d.normalize_bc_specs_2d). Genuinely mixed case:
    NH4 Dirichlet, O2 zero-flux Neumann (sealed oxygen supply, fixed NH4 feed
    -- physically meaningful, e.g. an anaerobic-leaning reactor), NO2/NO3
    Dirichlet(0) matching the Stage 6 convention. NO2/NO3 zero-flux was also
    tried and correctly reported "newton_stalled" (2D has no PTC fallback,
    documented in elliptic2d.py) rather than a silent wrong answer -- a
    genuine Fredholm-type well-posedness issue (net production under sealed
    boundaries), the same mechanism as the 1D Task 1(b) finding, not a defect
    in the bc_specs machinery; this test uses the well-posed configuration."""
    coeffs = elliptic_coefficients("toy")
    g = Grid2D(Nx=30, Ny=30)
    U = {sp: 0.02 * np.exp(-((g.X - 0.5) ** 2 + (g.Y - 0.5) ** 2) / 0.15 ** 2).ravel()
         for sp in SPECIES}
    bc_specs = {"NH4": ("dirichlet", 1.0), "O2": ("neumann", 0.0),
                "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0)}
    C, hist, method = solve_newton_2d(coeffs, U, g, bc_specs=bc_specs, tol=1e-9, maxiter=60)
    assert hist[-1] < 1e-9, (method, hist[-1])
    assert method == "newton"


def test_2d_neumann_nonzero_flux_raises_not_implemented():
    """The 2D solver's scope limitation (stated in normalize_bc_specs_2d) must
    fail loudly, not silently: there is no 2D equivalent of the 1D nonzero-
    flux Neumann row yet."""
    coeffs = elliptic_coefficients("toy")
    g = Grid2D(Nx=10, Ny=10)
    U = {sp: np.full(g.Npts, 0.02) for sp in SPECIES}
    bc_specs = {"NH4": ("neumann", -0.3), "O2": ("dirichlet", 0.1876),
                "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0)}
    try:
        solve_newton_2d(coeffs, U, g, bc_specs=bc_specs)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    test_control_volumes_are_exact()
    test_laplacian_2d_is_exactly_mass_conservative()
    test_laplacian_2d_is_second_order_accurate_and_isotropic()
    test_2d_reduces_exactly_to_1d_slab()
    test_2d_reduces_to_1d_cylindrical_second_order_in_interior()
    test_advection_2d_is_exactly_mass_conservative()
    test_advection_rho_jacobian_2d_matches_finite_difference()
    test_parabolic_2d_reaches_analytic_equilibrium()
    test_2d_per_substrate_bc_specs_mixed_dirichlet_neumann()
    test_2d_neumann_nonzero_flux_raises_not_implemented()
    print("All 2D extension validation checks passed.")
