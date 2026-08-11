"""
ITEM 5: adversarial, third-party-style re-verification of 3 load-bearing
claims, each re-derived FROM SCRATCH in this file -- no import of the
`nitrifiers` package, no reuse of any existing test code. The point is to
check whether these claims survive an independent derivation/implementation,
not to re-run the project's own code against itself.

CLAIM 1. The minimal travelling-wave speed formula used in ITEM 2,
    v_min = 2*sqrt(d*r*c0/(K+c0)),
re-derived here by symbolic linearisation of System (3.12) around the
unstable state (rho=0, c=c0), independently of the value typed into
report/item2_wave_speed.py.

CLAIM 2. ITEM 2's headline number, v_bar ~ 0.83 (vs. the paper's 0.8396),
re-obtained here with a SECOND, independently-coded PDE solver: centred (not
upwind) flux-form finite differences on the raw physical domain [0,200] (no
domain rescale), ghost-node ("mirror") ports for zero-flux BCs instead of
elliptic.py's row-replacement approach, ODE time integration still via
scipy.integrate.solve_ivp (a trusted external library, not project code) but
with an entirely separate right-hand-side implementation.

CLAIM 3. The exact discrete conservation identity claimed for the 2D
finite-volume Laplacian (report/audit_checklist.md D8: "sum_ij V_ij*(Lap@c)_ij
= 0 for ANY field c, to machine precision, at ANY resolution"), re-derived
analytically here via the telescoping-flux argument, then re-checked
numerically with a hand-rolled 5-point stencil built independently of
nitrifiers/grid2d.py.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# CLAIM 1: minimal wave speed, symbolic re-derivation
# =============================================================================
def claim1_v_min_symbolic():
    import sympy as sp

    print("=" * 78)
    print("CLAIM 1: v_min = 2*sqrt(d*r*c0/(K+c0)) -- symbolic re-derivation")
    print("=" * 78)

    z, v, d, a, r, K, b, c0 = sp.symbols("z v d a r K b c0", positive=True)
    rho = sp.Function("rho")(z)

    # System (3.12): d(rho)/dt = d/dx[(d+a*rho) rho_x] + rho*(r*c/(K+c) - b*rho)
    # The self-limitation term is QUADRATIC in rho (b*rho^2, a crowding term,
    # NOT a constant per-capita mortality b*rho) -- confirmed directly from
    # the ODE that f_reaction(rho, c) = rho*(r*c/(K+c) - b*rho) implements in
    # item2_wave_speed.py (re-read here independently, not assumed from memory).
    #
    # Travelling-wave ansatz z = x - v*t. Linearise around the UNSTABLE state
    # invaded by the front: rho=0 (exactly), c=c0 (its boundary/initial value,
    # valid ahead of the front where consumption has not yet acted). At rho=0:
    #   - the cross-diffusion term a*d/dx[rho*rho_x] is O(rho^2) -> drops at
    #     linear order (it's already a product of two factors that vanish).
    #   - the crowding term b*rho^2 is O(rho^2) -> ALSO drops at linear order.
    #   - only the plain diffusion d*rho_xx and the LINEAR growth
    #     rho*r*c0/(K+c0) survive.
    # So the linearised ODE in the moving frame is:
    #     d*rho'' + v*rho' + (r*c0/(K+c0))*rho = 0
    growth_rate = r * c0 / (K + c0)
    lam = sp.symbols("lambda")
    char_eq = d * lam**2 + v * lam + growth_rate
    print(f"\nLinearised ODE (moving frame, rho=0/c=c0 base state):")
    print(f"  d*rho'' + v*rho' + (r*c0/(K+c0))*rho = 0")
    print(f"Characteristic equation: {sp.Eq(char_eq, 0)}")

    disc = sp.discriminant(char_eq, lam)
    disc = sp.simplify(disc)
    print(f"\nDiscriminant (in lambda): {disc}")
    v_min_expr = sp.solve(sp.Eq(disc, 0), v)
    v_min_expr = [sp.simplify(s) for s in v_min_expr if s.is_positive is not False]
    print(f"Roots of discriminant=0 (real-root/oscillation boundary): {v_min_expr}")

    v_min_formula = 2 * sp.sqrt(d * r * c0 / (K + c0))
    match = sp.simplify(v_min_expr[-1] - v_min_formula) == 0 if v_min_expr else False
    print(f"\nCandidate closed form: v_min = 2*sqrt(d*r*c0/(K+c0))")
    print(f"Matches the boundary root of the discriminant condition: {match}")

    # numeric check at the paper's own Table 1 Case (A) values
    subs = {d: 1e-6, r: 1.0, K: 1.0, c0: 5.0}
    v_min_num = float(v_min_formula.subs(subs))
    print(f"\nNumeric value at paper's Case (A) (d=1e-6, r=1, K=1, c0=5): "
          f"v_min = {v_min_num:.6f}")
    print(f"Paper reports: 0.0018")
    print(f"Relative difference: {abs(v_min_num - 0.0018) / 0.0018:.2%}")
    return match, v_min_num


# =============================================================================
# CLAIM 2: independent PDE solve for the travelling-wave speed v_bar
# =============================================================================
def _rhs_independent(t, y, Nx, h, d, a, r, K, b, Y, D):
    """Centred (non-upwind) flux-form finite differences, hand-rolled here --
    intentionally NOT the same discretisation as elliptic.py/parabolic.py
    (which use upwind advection and row-replacement BCs). Ghost points
    enforce zero-flux (Neumann) BCs by mirroring: rho[-1]=rho[0],
    rho[Nx+1]=rho[Nx] (and same for c)."""
    rho = y[:Nx + 1]
    c = y[Nx + 1:]

    rho_g = np.empty(Nx + 3)
    rho_g[1:-1] = rho
    rho_g[0] = rho[1]      # mirror: enforces zero-flux at x=0
    rho_g[-1] = rho[-2]    # mirror: enforces zero-flux at x=L
    c_g = np.empty(Nx + 3)
    c_g[1:-1] = c
    c_g[0] = c[1]
    c_g[-1] = c[-2]

    # nonlinear diffusion flux F_m = (d + a*rho_avg)*(rho_g[m+1]-rho_g[m])/h,
    # rho_avg = 0.5*(rho_g[m]+rho_g[m+1]) -- CENTRED average, not upwind.
    # m runs over all Nx+2 faces of the ghost array (indices 0..Nx+1).
    rho_avg = 0.5 * (rho_g[:-1] + rho_g[1:])          # length Nx+2
    F = (d + a * rho_avg) * (rho_g[1:] - rho_g[:-1]) / h   # length Nx+2
    drho_diff = (F[1:] - F[:-1]) / h                  # length Nx+1, aligned to real pts

    c_xx = (c_g[2:] - 2 * c_g[1:-1] + c_g[:-2]) / h ** 2

    monod = c / (K + c)
    f = rho * (r * monod - b * rho)
    g = rho * (r / Y) * monod

    drho = drho_diff + f
    dc = D * c_xx - g
    return np.concatenate([drho, dc])


def claim2_independent_wave_speed():
    from scipy.integrate import solve_ivp

    print("\n" + "=" * 78)
    print("CLAIM 2: v_bar ~ 0.84 -- independent second solver (physical domain,")
    print("centred FD, ghost-node BCs, no rescale, no project code)")
    print("=" * 78)

    d, a, r, K, b, Y, D = 1e-6, 1e-5, 1.0, 1.0, 0.1, 0.2, 1e-4
    L, T = 200.0, 200.0
    Nx = 800   # coarser than item2_wave_speed.py's Nx=2000: this is a spot
               # check for independent agreement, not a resolution study
    h = L / Nx
    x = np.linspace(0.0, L, Nx + 1)

    rho0 = np.exp(-x)
    c0 = np.full(Nx + 1, 5.0)
    y0 = np.concatenate([rho0, c0])

    t_eval = np.array([0.0, 160.0, 200.0])
    sol = solve_ivp(_rhs_independent, (0.0, T), y0, method="RK45", t_eval=t_eval,
                     rtol=1e-7, atol=1e-9, args=(Nx, h, d, a, r, K, b, Y, D))
    print(f"\nsolve_ivp success={sol.success}, nfev={sol.nfev}")
    if not sol.success:
        print(f"  message: {sol.message}")

    Npts = Nx + 1
    rho_160 = sol.y[:Npts, 1]
    rho_200 = sol.y[:Npts, 2]
    x_bar_160 = x[np.argmax(rho_160)]
    x_bar_200 = x[np.argmax(rho_200)]
    v_bar = (x_bar_200 - x_bar_160) / 40.0

    print(f"argmax(rho) at t=160: x={x_bar_160:.4f}")
    print(f"argmax(rho) at t=200: x={x_bar_200:.4f}")
    print(f"\nINDEPENDENT v_bar = {v_bar:.6f}")
    print(f"ITEM 2's original (upwind FV, rescaled domain) result: 0.8325")
    print(f"Paper reports: 0.8396")
    print(f"Independent-vs-ITEM-2 relative difference: "
          f"{abs(v_bar - 0.8325) / 0.8325:.2%}")
    print(f"Independent-vs-paper relative difference: "
          f"{abs(v_bar - 0.8396) / 0.8396:.2%}")
    return v_bar


# =============================================================================
# CLAIM 3: 2D FV Laplacian exact conservation identity
# =============================================================================
def claim3_2d_laplacian_conservation():
    print("\n" + "=" * 78)
    print("CLAIM 3: sum_ij V_ij*(Lap@c)_ij = 0 for ANY c -- analytic + fresh")
    print("independent numerical re-check (own stencil, not grid2d.py)")
    print("=" * 78)

    print("""
Analytic re-derivation (telescoping flux argument), from scratch:
  A conservative finite-volume Laplacian on any grid can be written per node k
  as (Lap@c)_k = (1/V_k) * sum_{faces f of k} (c_neighbor - c_k) * A_f / dist_f,
  i.e. V_k*(Lap@c)_k = sum_f Flux_f(k), where Flux_f(k) for the SHARED interior
  face between nodes k and m is antisymmetric: Flux_f(k) = -Flux_f(m) (the flux
  leaving k across a face is exactly the flux entering m across that same
  face, by construction of a two-point flux with a common face area/distance).
  Summing V_k*(Lap@c)_k over ALL k: every interior face's contribution appears
  exactly twice, once as +Flux and once as -Flux (from its two adjacent
  nodes), and cancels exactly. Boundary (domain-edge) nodes in the pure-
  Neumann (zero-flux) construction have no face there at all (by construction
  -- there IS no neighbour outside the domain), so they contribute nothing
  extra. Hence sum_k V_k*(Lap@c)_k = 0 identically, for ANY c, independent of
  resolution -- a telescoping sum, not an asymptotic (h->0) statement.
""")

    rng = np.random.default_rng(0)

    def independent_fv_laplacian_2d(Nx, Ny, Lx=1.0, Ly=1.0):
        """Hand-rolled 5-point conservative FV Laplacian + control volumes,
        written independently of nitrifiers/grid2d.py (same underlying
        finite-volume idea -- there is only one standard way to do this
        correctly -- but re-derived and re-coded here, not imported)."""
        hx, hy = Lx / Nx, Ly / Ny
        dx = np.full(Nx + 1, hx); dx[0] = dx[-1] = hx / 2
        dy = np.full(Ny + 1, hy); dy[0] = dy[-1] = hy / 2
        V = np.outer(dx, dy)
        n = (Nx + 1) * (Ny + 1)

        def k(i, j):
            return i * (Ny + 1) + j

        rows, cols, vals = [], [], []
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                kc = k(i, j)
                diag = 0.0
                if i < Nx:
                    w = hy / (hx * V[i, j])
                    rows.append(kc); cols.append(k(i + 1, j)); vals.append(w); diag -= w
                if i > 0:
                    w = hy / (hx * V[i, j])
                    rows.append(kc); cols.append(k(i - 1, j)); vals.append(w); diag -= w
                if j < Ny:
                    w = hx / (hy * V[i, j])
                    rows.append(kc); cols.append(k(i, j + 1)); vals.append(w); diag -= w
                if j > 0:
                    w = hx / (hy * V[i, j])
                    rows.append(kc); cols.append(k(i, j - 1)); vals.append(w); diag -= w
                rows.append(kc); cols.append(kc); vals.append(diag)
        import scipy.sparse as sp
        Lap = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return Lap, V.ravel()

    all_ok = True
    for Nx, Ny in [(10, 10), (23, 17), (40, 40)]:
        Lap, Vflat = independent_fv_laplacian_2d(Nx, Ny)
        for trial in range(3):
            c = rng.standard_normal(Lap.shape[0])
            total = float(np.sum(Vflat * (Lap @ c)))
            ok = abs(total) < 1e-10
            all_ok &= ok
            print(f"  grid {Nx}x{Ny}, random field trial {trial}: "
                  f"sum(V*Lap@c) = {total:.3e}  {'OK' if ok else 'FAIL'}")

    print(f"\nCLAIM 3 {'CONFIRMED' if all_ok else 'FAILED'} by an independently "
          f"coded 5-point FV stencil across 3 grid sizes x 3 random fields each.")
    return all_ok


if __name__ == "__main__":
    match1, v_min_num = claim1_v_min_symbolic()
    v_bar_indep = claim2_independent_wave_speed()
    claim3_ok = claim3_2d_laplacian_conservation()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"CLAIM 1 (v_min closed form): symbolic derivation matches the formula "
          f"used = {match1}; numeric value {v_min_num:.6f} vs paper's 0.0018 "
          f"({abs(v_min_num-0.0018)/0.0018:.1%} relative diff)")
    print(f"CLAIM 2 (v_bar via independent solver): {v_bar_indep:.6f} vs ITEM 2's "
          f"0.8325 vs paper's 0.8396")
    print(f"CLAIM 3 (2D FV Laplacian exact conservation): "
          f"{'CONFIRMED' if claim3_ok else 'FAILED'} via independent stencil")
