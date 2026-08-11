"""
Closed-form ground-truth test for the FULL elliptic PDE solve (solve_newton
and solve_picard), not just the reaction-term assembly (see
test_reaction_terms.py for that). This closes the coverage gap flagged in
review: every other test validates internal consistency (Newton vs Picard vs
relaxation), which would not have caught a bug present identically in all
three solvers.

Construction: a single-species, single-substrate, no-production benchmark is
carved out of the general 4-substrate/3-species machinery by:
  - setting u_NOB = u_CMX = 0 (their entire reaction contribution vanishes,
    since every R_j term is proportional to u_i)
  - setting AOB's K_O2 tiny, so M(c_O2;K_O2) ~= 1 everywhere regardless of the
    O2 profile (decouples the O2 dependence)
  - setting AOB's K_NH4 >> c_inf, so M(c_NH4;K_NH4) ~= c_NH4/K_NH4 (the
    Monod term is deep in its linear/first-order regime across the whole
    domain, since c_NH4 <= c_inf everywhere)

This reduces the NH4 equation to the classical linear reaction-diffusion
"effectiveness factor" problem:

    0 = Lap(c) - kappa^2 * c,      c(1) = c_inf,   dc/dr(0) = 0

with kappa^2 = Lambda_AOB,NH4 * u_AOB / K_NH4, which has standard closed-form
solutions (see e.g. Bird/Stewart/Lightfoot or any reaction-diffusion text):

    slab (p=0):        c(r) = c_inf * cosh(kappa*r) / cosh(kappa)
    cylindrical (p=1):  c(r) = c_inf * I0(kappa*r) / I0(kappa)      (I0 = modified Bessel, scipy.special.iv)
    spherical (p=2):    c(r) = c_inf * sinh(kappa*r) / (r*sinh(kappa)),   c(0) = c_inf * kappa/sinh(kappa)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import iv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nitrifiers.elliptic import Grid, solve_newton, solve_picard


def _closed_form(r, kappa, c_inf, p):
    if p == 0:
        return c_inf * np.cosh(kappa * r) / np.cosh(kappa)
    elif p == 1:
        return c_inf * iv(0, kappa * r) / iv(0, kappa)
    elif p == 2:
        out = np.empty_like(r)
        mask = r > 1e-12
        out[mask] = c_inf * np.sinh(kappa * r[mask]) / (r[mask] * np.sinh(kappa))
        out[~mask] = c_inf * kappa / np.sinh(kappa)
        return out
    else:
        raise ValueError(p)


def _single_species_coeffs(kappa, K_NH4, u0, c_inf=1.0):
    Lambda_NH4 = kappa ** 2 * K_NH4 / u0
    return {
        "Lambda": {
            "AOB": {"NH4": Lambda_NH4, "O2": 1.0},
            "NOB": {"NO2": 1.0, "O2": 1.0},
            "CMX": {"NH4": 1.0, "O2": 1.0},
        },
        "LambdaProd": {"AOB": Lambda_NH4, "NOB": 1.0, "CMX": 1.0},
        "Khat": {
            "AOB": {"NH4": K_NH4, "O2": 1e-6},  # K_O2 tiny -> M_O2 ~= 1 (decoupled)
            "NOB": {"NO2": 1.0, "O2": 1.0},
            "CMX": {"NH4": 1.0, "O2": 1.0},
        },
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "production": {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"),
                        "CMX": ("NO3", "CMX_to_NO3")},
        "rhat": {"AOB": 1.0, "NOB": 1.0, "CMX": 1.0},
        "c_inf_hat": {"NH4": c_inf, "NO2": 0.0, "NO3": 0.0, "O2": 1.0},
    }


def _run_case(geometry, p, kappa, N=800):
    grid = Grid(N=N, geometry=geometry, p=p)
    Npts = grid.N + 1
    # K_NH4 >> c_inf puts the Monod term deep in its linear regime. Needs to be
    # large enough (1e4 is NOT enough) that its own linearization error stays
    # below the genuine (now correctly 2nd-order) discretization error even at
    # N~800; 1e7 was confirmed to do so by checking that increasing K further
    # does not change the result at a fixed, moderately fine N.
    K_NH4, u0, c_inf = 1.0e7, 1.0, 1.0
    coeffs = _single_species_coeffs(kappa, K_NH4, u0, c_inf)
    U = {"AOB": np.full(Npts, u0), "NOB": np.zeros(Npts), "CMX": np.zeros(Npts)}

    # tol=1e-9 rather than the (tighter) default: at kappa~2 the residual
    # plateaus in the 1e-10 range from floating-point noise once genuinely
    # converged, which otherwise trips the (correct, but here spurious)
    # backtracking-stalled warning for no practical benefit.
    C_newton, hist_n, _ = solve_newton(coeffs, U, grid, bc_type="dirichlet", maxiter=100, tol=1e-9)
    # plain (relax=1.0) Picard does not converge at this Thiele modulus (consistent
    # with the Stage 3/4 finding that Picard needs under-relaxation for stiff
    # problems); relax=0.1 here mirrors that finding rather than working around it.
    C_picard, hist_p = solve_picard(coeffs, U, grid, bc_type="dirichlet",
                                     maxiter=20000, tol=1e-10, relax=0.1)

    c_exact = _closed_form(grid.r, kappa, c_inf, p)
    err_newton = np.max(np.abs(C_newton["NH4"] - c_exact))
    err_picard = np.max(np.abs(C_picard["NH4"] - c_exact))
    return err_newton, err_picard, hist_n, hist_p


def test_slab_matches_closed_form():
    err_n, err_p, hist_n, hist_p = _run_case("slab", 0, kappa=2.0)
    assert hist_n[-1] < 1e-9
    # Newton: tightened from 1e-3 after fixing the row-0 reaction-zeroing bug
    # (true error now ~1e-6-1e-6.5, not ~1e-4); Picard's own convergence
    # tolerance/relaxation makes it noticeably less tight in practice, so it
    # keeps a looser bound.
    assert err_n < 1e-4, err_n
    assert err_p < 1e-3, err_p


def test_cylindrical_matches_closed_form():
    err_n, err_p, hist_n, hist_p = _run_case("radial", 1, kappa=2.0)
    assert hist_n[-1] < 1e-9
    # Newton: tightened from 1e-3 after fixing the row-0 reaction-zeroing bug
    # (true error now ~1e-6-1e-6.5, not ~1e-4); Picard's own convergence
    # tolerance/relaxation makes it noticeably less tight in practice, so it
    # keeps a looser bound.
    assert err_n < 1e-4, err_n
    assert err_p < 1e-3, err_p


def test_spherical_matches_closed_form():
    err_n, err_p, hist_n, hist_p = _run_case("radial", 2, kappa=2.0)
    assert hist_n[-1] < 1e-9
    # Newton: tightened from 1e-3 after fixing the row-0 reaction-zeroing bug
    # (true error now ~1e-6-1e-6.5, not ~1e-4); Picard's own convergence
    # tolerance/relaxation makes it noticeably less tight in practice, so it
    # keeps a looser bound.
    assert err_n < 1e-4, err_n
    assert err_p < 1e-3, err_p


def test_spherical_error_converges_with_grid_refinement():
    """This scheme is genuinely second-order accurate (error shrinks ~4x per
    doubling of N), matching what a "conservative finite-volume Laplacian"
    should give. An earlier version of this test found and accepted
    first-order convergence instead, attributing it to an unexplained
    property of the scheme -- that was wrong. Direct comparison of the
    assembled Newton Jacobian against a hand-built reference matrix (for the
    trivial case Lap*c - kappa^2*c = 0) found the actual cause: the reaction
    term was being incorrectly zeroed at grid index 0 (the r=0 centre point),
    in both the residual (_residual) and the Jacobian (_assemble_global) in
    elliptic.py, and in the analogous mass/reaction handling in
    relaxation.py. Index 0 is NOT a boundary-condition row (it uses a
    symmetric stencil for the true PDE at r=0, not an algebraic BC
    substitution like index N, which apply_bc fully replaces) -- so it must
    keep its reaction contribution. Dropping it left a fixed, non-vanishing
    defect at a single grid point, which is exactly the kind of error that
    degrades a 2nd-order scheme to apparent 1st-order global convergence.
    Fixed by no longer zeroing R/dR at index 0 in either module."""
    errs = []
    for N in (50, 100, 200, 400):
        err_n, _, _, _ = _run_case("radial", 2, kappa=1.5, N=N)
        errs.append(err_n)
    # second-order convergence: refining N by 8x (50->400) should shrink the
    # error by ~64x; require at least 20x to allow some slack (finite K_NH4
    # linearization and floating-point effects blunt the ideal ratio somewhat)
    assert errs[-1] < errs[0] / 20, errs


def test_weaker_reaction_regime_also_matches():
    """A second Thiele modulus, to make sure the benchmark isn't tuned to one
    lucky kappa."""
    err_n, err_p, hist_n, hist_p = _run_case("radial", 2, kappa=0.5, N=200)
    assert err_n < 1e-4, err_n
    assert err_p < 1e-4, err_p


def _single_species_neumann_coeffs(kappa, K_NH4, u0, flux):
    return {
        "Lambda": {"AOB": {"NH4": kappa ** 2 * K_NH4 / u0, "O2": 1.0},
                   "NOB": {"NO2": 1.0, "O2": 1.0}, "CMX": {"NH4": 1.0, "O2": 1.0}},
        "LambdaProd": {"AOB": kappa ** 2 * K_NH4 / u0, "NOB": 1.0, "CMX": 1.0},
        "Khat": {"AOB": {"NH4": K_NH4, "O2": 1e-6}, "NOB": {"NO2": 1.0, "O2": 1.0}, "CMX": {"NH4": 1.0, "O2": 1.0}},
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "production": {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")},
        "rhat": {"AOB": 1.0, "NOB": 1.0, "CMX": 1.0},
        "c_inf_hat": {"NH4": flux, "NO2": 0.0, "NO3": 0.0, "O2": 1.0},  # only used if bc_specs is omitted
    }


def test_nonzero_flux_neumann_matches_closed_form():
    """ITEM 1 re-verification: this test used to run against a test-only
    wrapper (_solve_newton_mixed_bc) that duplicated solve_newton's logic,
    because the production solve_newton only accepted one global bc_type for
    all 4 substrates. It now calls the REAL, refactored production
    solve_newton(bc_specs=...) directly -- the per-substrate interface added
    for ITEM 1 -- for all three geometries, not just spherical.

    Regression test for Fix 1: elliptic.py::_residual (and, by the same
    hardcoded-0.0 pattern, solve_picard and relaxation.py::solve_relaxation)
    used to force the Neumann row's residual TARGET to 0.0 regardless of the
    actual flux `value` passed to apply_bc -- so any bc_type='neumann' solve
    silently converged to a zero-flux solution no matter what nonzero flux was
    requested. Checked two ways: (1) the closed-form Neumann+linear-reaction
    solution from the Task 2 catalogue per geometry, A fixed by the flux --
    errors should shrink with N, dominated by the documented first-order
    one-sided Neumann truncation, not stuck at O(1); and (2) directly, by
    finite-differencing the converged profile at the boundary and checking it
    equals the requested flux, not zero."""
    kappa, K_NH4, u0, flux = 1.5, 1.0e7, 1.0, -0.3  # negative = influx -> physically positive profile

    def closed_form(r, p):
        if p == 0:
            A = -flux / (kappa * np.sinh(kappa))
            return A * np.cosh(kappa * r)
        if p == 1:
            A = -flux / (kappa * iv(1, kappa))
            return A * iv(0, kappa * r)
        A = -flux / (kappa * np.cosh(kappa) - np.sinh(kappa))
        out = np.empty_like(r)
        mask = r > 1e-12
        out[mask] = A * np.sinh(kappa * r[mask]) / r[mask]
        out[~mask] = A * kappa
        return out

    for geometry, p in [("slab", 0), ("radial", 1), ("radial", 2)]:
        coeffs = _single_species_neumann_coeffs(kappa, K_NH4, u0, flux)
        bc_specs = {"NH4": ("neumann", flux), "NO2": ("dirichlet", 0.0),
                    "NO3": ("dirichlet", 0.0), "O2": ("dirichlet", 1.0)}
        errs = {}
        for N in (100, 400):
            grid = Grid(N=N, geometry=geometry, p=p)
            r = grid.r
            Npts = grid.N + 1
            U = {"AOB": np.full(Npts, u0), "NOB": np.zeros(Npts), "CMX": np.zeros(Npts)}
            C, hist, method = solve_newton(coeffs, U, grid, bc_specs=bc_specs,
                                            maxiter=100, tol=1e-9)
            assert hist[-1] < 1e-9, (geometry, p, N, hist[-1])

            c_exact = closed_form(r, p)
            errs[N] = np.max(np.abs(C["NH4"] - c_exact))

            actual_flux = -(C["NH4"][-1] - C["NH4"][-2]) / grid.h
            assert abs(actual_flux - flux) < 1e-6, (geometry, p, N, actual_flux, flux)
        assert errs[400] < errs[100], (geometry, p, errs)  # (1): error shrinks with N


def test_coupled_multi_substrate_neumann_converges():
    """ITEM 1 core deliverable: a genuinely-coupled multi-substrate Neumann
    solve -- nonzero flux on NH4 AND O2 SIMULTANEOUSLY, full 3-species
    reaction network active (not a single-species reduction) -- using the
    per-substrate bc_specs interface on the real production solve_newton.
    This is the exact class of configuration that previously diverged
    (residual ~2e9) or hung indefinitely before the refactor.

    Uses a MODERATE reaction-rate scale (Lambda~O(1-10), Khat~O(1)), not the
    deep-linear-Monod-limit scaling (K>>c, Lambda~1e7) used for the
    closed-form tests above, and not the full eloi preset's realistic
    stiffness. This is a deliberate, documented scope: direct investigation
    found that BOTH the deep-linear-limit construction AND the actual eloi
    preset's coefficients produce a genuinely near-singular Jacobian at the
    natural initial guess for this specific case (condition number ~1e16,
    confirmed via direct SVD -- at the edge of double-precision
    representability), for reasons that are a property of the coupled
    nonlinear BVP's conditioning at that state, not of the bc_specs
    machinery itself (individually, e.g. via the closed-form test above,
    each substrate's own Neumann handling is exact to ~1e-15). Whether a
    better initial guess, a homotopy/continuation strategy, or a
    preconditioner resolves the realistic-stiffness case is left open; this
    test demonstrates the refactored per-substrate machinery, the physical-
    plausibility guard (elliptic.py), and the degeneracy fallback
    (relaxation.py) all work correctly TOGETHER for a coupled multi-substrate
    Neumann problem at a reaction-rate scale where the underlying Jacobian is
    not itself pathological.
    """
    coeffs = {
        "Lambda": {"AOB": {"NH4": 5.0, "O2": 5.0}, "NOB": {"NO2": 1.0, "O2": 1.0}, "CMX": {"NH4": 1.0, "O2": 1.0}},
        "LambdaProd": {"AOB": 5.0, "NOB": 1.0, "CMX": 1.0},
        "Khat": {"AOB": {"NH4": 1.0, "O2": 1.0}, "NOB": {"NO2": 1.0, "O2": 1.0}, "CMX": {"NH4": 1.0, "O2": 1.0}},
        "beta": {"AOB_to_NO2": 1.0, "NOB_to_NO3": 1.0, "CMX_to_NO3": 1.0},
        "production": {"AOB": ("NO2", "AOB_to_NO2"), "NOB": ("NO3", "NOB_to_NO3"), "CMX": ("NO3", "CMX_to_NO3")},
        "rhat": {"AOB": 1.0, "NOB": 1.0, "CMX": 1.0},
        "c_inf_hat": {"NH4": -0.3, "NO2": 0.0, "NO3": 0.0, "O2": -0.3},
    }
    bc_specs = {"NH4": ("neumann", -0.3), "O2": ("neumann", -0.3),
                "NO2": ("dirichlet", 0.0), "NO3": ("dirichlet", 0.0)}
    for N in (40, 80, 150):
        grid = Grid(N=N, geometry="radial", p=2)
        Npts = grid.N + 1
        U = {"AOB": np.full(Npts, 1.0), "NOB": np.zeros(Npts), "CMX": np.zeros(Npts)}
        C, hist, method = solve_newton(coeffs, U, grid, bc_specs=bc_specs,
                                        maxiter=300, tol=1e-9)
        assert hist[-1] < 1e-9, (N, method, hist[-1])
        assert method in ("newton", "newton_inner_relax_fallback"), (N, method)

        h = grid.h
        for sub, target in (("NH4", -0.3), ("O2", -0.3)):
            actual_flux = -(C[sub][-1] - C[sub][-2]) / h
            assert abs(actual_flux - target) < 1e-8, (N, sub, actual_flux, target)


if __name__ == "__main__":
    test_slab_matches_closed_form()
    test_cylindrical_matches_closed_form()
    test_spherical_matches_closed_form()
    test_spherical_error_converges_with_grid_refinement()
    test_weaker_reaction_regime_also_matches()
    test_nonzero_flux_neumann_matches_closed_form()
    test_coupled_multi_substrate_neumann_converges()
    print("All closed-form ground-truth tests passed.")
