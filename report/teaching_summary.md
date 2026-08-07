# Context dump: Slow-fast PDE model for nitrifying bacteria biofilms — everything done, for teaching purposes

I've been working on a numerical PDE solver for my thesis and want you to teach me the underlying methods/concepts in depth. Below is a complete account of the model, the six-stage numerical pipeline, every bug we found, every test we ran, and the numerical-analysis techniques used to validate it. Please use this as context to explain things to me — the math behind each method, why each technique was the right tool, and anything I should understand more deeply. I'll ask follow-up questions on specific parts.

Repository: https://github.com/AryanAmogh10/PDE_model_Nitrifying_Bacteria (Python, NumPy/SciPy)

---

## 1. The physical/mathematical model

Based on arXiv:2512.13156 (Freingruber, Gonzalez-Cabaleiro, Yoldas). Three competing nitrifying bacterial species growing in a biofilm granule:
- **AOB** (ammonia-oxidizing bacteria): NH4 + O2 → NO2
- **NOB** (nitrite-oxidizing bacteria): NO2 + O2 → NO3
- **CMX** (comammox): NH4 + O2 → NO3 directly (complete ammonia oxidation, competes with AOB for the same substrates)

Two families of coupled PDEs on a 1D domain r ∈ [0,1] (radial coordinate — see the "geometry" section below for what this actually means):

**Slow equation (bacterial density, one per species i ∈ {AOB, NOB, CMX}):**
```
∂u_i/∂t = div(d_i ∇u_i + a_i u_i ∇ρ) + u_i f_i(u_i, c)
ρ = u_AOB + u_NOB + u_CMX   (total local biomass density)
f_i(u_i, c) = r_i · M(c_p; K_ip) · M(c_s; K_is) - b_i · ρ
```
This is a **parabolic** (diffusion + advection + reaction) equation. `M(c;K) = c/(K+c)` is **Monod kinetics** (a saturating growth-rate function — think Michaelis-Menten enzyme kinetics, same functional form). `d_i ∇u_i` is ordinary diffusion of bacteria. `a_i u_i ∇ρ` is a **cross-diffusion / advection** term: bacteria move down the gradient of TOTAL biomass density (crowding-driven motion, not driven by their own gradient). The death term `-b_i·ρ` is **density-dependent** (crowding mortality) — this matters a lot, see bug #1 below.

**Fast equation (substrate concentration, one per substrate j ∈ {NH4, NO2, NO3, O2}):**
```
0 = D_j ∇²c_j + c_j g_j(u, c)
```
This is **elliptic** (no time derivative) because substrate diffusion is assumed to equilibrate much faster than bacterial growth — a **quasi-steady-state / slow-fast** assumption. The scale-separation parameter is `ε = r_max·L²/D_j`; we verified `ε << 1` for every parameter set used, justifying treating the substrate system as instantaneously equilibrated at each "slow" time step.

**Geometry:** the domain is 1D in the code (`r ∈ [0,1]`) but represents a real 3D (sphere) or 2D (infinite cylinder) object under an assumed **radial symmetry**: concentration/density depend only on distance from the center, not direction. Under that assumption the full 3D Laplacian `∇²c = ∂²c/∂x²+∂²c/∂y²+∂²c/∂z²` collapses EXACTLY (not approximately) to `(1/r^p) d/dr(r^p dc/dr)`, where p=0 (slab/Cartesian), p=1 (cylindrical), p=2 (spherical). This is the general form used throughout; `p` is just an exponent parameter in the code.

## 2. The six-stage software pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | `params.py` | Raw dimensional parameters (growth rates, K values, diffusivities) for 3 presets: `toy` (round numbers for testing), `rebeca` (email-derived estimate), `eloi` (literature/thermodynamics-derived, the "real" one) |
| 2 | `nondim.py` | Non-dimensionalization — rescales everything into dimensionless groups (Λ, K̂, etc.) and checks the ε<<1 slow-fast separation |
| 3 | `elliptic.py` | Solves the fast substrate system at a fixed bacterial profile — **Newton's method** (primary) and **Picard/fixed-point iteration** (secondary, less robust) |
| 4 | `relaxation.py` | An independent alternative solver for Stage 3 using **pseudo-transient continuation (PTC)** — used to cross-check Newton |
| 5 | `parabolic.py` | Solves the slow bacterial-density evolution for one time step — **fully-implicit backward Euler + Newton** |
| 6 | `slowfast.py` | The outer loop: alternates Stage 3 (solve substrates given current bacteria) and Stage 5 (advance bacteria given current substrates) — this IS the slow-fast splitting scheme |

## 3. Numerical methods used (this is probably what you want explained most)

### 3.1 Finite-volume discretization
Instead of finite differences, the spatial discretization is **conservative finite-volume**: each grid point owns a "control volume" (a shell of the domain), and the discrete Laplacian is built by exactly integrating the flux balance over that volume:
```
Lap_i = [flux_east - flux_west] / V_i
flux_face = r_face^p · (c_neighbor - c_i)/h     (centered difference at the face)
V_i = ∫ r^p dr over the cell                     (the cell's true volume)
```
This guarantees **exact mass conservation** (a telescoping-sum identity: interior fluxes cancel in pairs when you sum weighted by V_i, leaving only boundary flux) — this is a *much* stronger property than just "converges to the right answer"; it holds at ANY grid resolution, not just in the limit.

**Key subtlety we debugged:** the domain center r=0 needs special treatment (no true "west face" — it's the coordinate singularity, not a geometric boundary in p>0 geometries since the inner face area r^p → 0). And the interior rows next to the center have small enough r_i that the *approximate* cell volume `V_i ≈ r_i^p · h` (valid for r_i >> h) breaks down. We fixed this by using the *exact* integrated cell volume everywhere.

### 3.2 Newton's method (primary elliptic + parabolic solver)
Standard Newton-Raphson on the nonlinear system F(C)=0, with:
- **Backtracking line search**: instead of accepting the full Newton step, halve the step size until the residual actually decreases — needed because raw Newton steps can overshoot into unphysical (negative) concentrations for stiff problems
- **Non-negativity projection**: clip concentrations to ≥0 after each trial step
- We ADDED (mid-project) a **physical-plausibility upper bound** too — reject a trial step unless it BOTH decreases the residual AND keeps every concentration under a sane ceiling (5× the largest boundary/feed concentration). This was needed because plain "did the residual decrease" backtracking can accept a step that lands on a mathematically-real but physically absurd root (we found Newton converging to NH4≈100 and NO3≈5416 in one broken configuration — see bug #4 below)

### 3.3 Picard (fixed-point) iteration
A simpler alternative: freeze the nonlinear reaction term at its current value, solve the resulting LINEAR system for the substrate profile, repeat. Linear (not quadratic) convergence, much cheaper per iteration, but only converges under-relaxed (damped) for stiff parameter regimes, and — we found — genuinely fails to converge at all for the realistic `eloi` preset's reaction stiffness without a specific center-row workaround. We diagnosed this via eigenvalue/conditioning analysis of the discretized Laplacian: the fixed-point iteration matrix's spectral radius exceeds 1 for this problem, a genuine mathematical property of the iteration, not a coding bug.

### 3.4 Pseudo-transient continuation (PTC) — the `relaxation.py` solver
Instead of solving the steady-state (elliptic) equation directly, march a fictitious "pseudo-time" backward-Euler scheme:
```
(M/dt - J) ΔC = F(C)
```
starting with small `dt` (very stable, slow) and geometrically growing `dt` each step (approaching a plain Newton step as dt→∞). This is a classic trick for solving nonlinear systems that are hard for direct Newton: it "smooths" the path to the solution. We used it two ways: (1) as an independent cross-check of the Newton solver (do two totally different algorithms agree?), and (2) as an automatic fallback when Newton's Jacobian becomes singular.

### 3.5 Closed-form benchmarking (the core validation technique)
For the FULL nonlinear model (Monod kinetics), there is no closed-form solution — no known way to write down the exact answer analytically. So we constructed a REDUCED problem that DOES have one: set two of three species' densities to zero, and take the third species' half-saturation constant K for the limiting substrate to be enormous (K >> c), which pushes the Monod term `c/(K+c)` deep into its LINEAR regime (`≈ c/K`). This reduces the PDE to the classical linear reaction-diffusion "effectiveness factor" problem:
```
0 = ∇²c - κ²c,   c(1)=c_inf,   dc/dr(0)=0
```
which has textbook closed-form solutions depending on geometry:
- slab: `c(r) = c_inf · cosh(κr)/cosh(κ)`
- cylindrical: `c(r) = c_inf · I₀(κr)/I₀(κ)` (I₀ = modified Bessel function)
- spherical: `c(r) = c_inf · sinh(κr)/(r·sinh(κ))`

We ran the REAL numerical solver (Newton, Picard) on this reduced problem and compared against the exact formula — errors of 1e-4 to 1e-6, and crucially we checked the ERROR SHRINKS at the theoretically-expected rate as the grid is refined (see convergence order below). This is how you validate a solver when the real (nonlinear) problem has no ground truth to check against.

### 3.6 Convergence order studies
The core numerical-analysis idea: if a discretization is genuinely O(h²) accurate (h = grid spacing), then halving h should quarter the error (ratio 4, "order 2" since 4=2²). We ran a grid-refinement study (N=50,100,200,400,800 grid points) and computed `order = log2(error_ratio)` between successive refinements. This is THE standard way to empirically confirm a numerical scheme's theoretical accuracy order.

**A subtlety we hit and had to disentangle:** the measured order was stuck around 1.7-1.9 (not the expected 2.0) even after fixing a real discretization bug. We had to figure out whether this was (a) a genuine remaining discretization defect, or (b) an artifact of finite floating-point precision. The test: does INCREASING the numerical solver's tolerance / iteration budget change anything? No — errors were IDENTICAL to 6 significant figures whether we asked for tol=1e-9 or tol=1e-14, meaning the solver had already converged as far as double-precision arithmetic allows; something else was the floor. We traced it to CATASTROPHIC CANCELLATION: the residual computation sums two ~10⁷-magnitude terms (from the huge K constant needed to force the linear-Monod-limit) that nearly cancel to an O(1) result — double precision's ~16 significant digits leaves an absolute floor around 1e-9 to 1e-10 once you're subtracting numbers that large. To PROVE this (not just suspect it), we recomputed the identical discretization using **50-decimal-digit arbitrary-precision arithmetic** (Python's `mpmath` library) via a hand-written tridiagonal solve (Thomas algorithm) that bypasses SciPy's ordinary double-precision sparse solver entirely. Result: clean order 1.999–2.000 across the FULL range including where double precision degraded — proof the true discretization order genuinely is 2, and the apparent degradation was purely a floating-point artifact, not a real defect.

### 3.7 Mass conservation as a structural invariant check
Separately from convergence order, we checked that `Σ V_i · Lap_i(u) ≈ (boundary flux)` EXACTLY (to ~1e-13, i.e. machine precision) for an arbitrary test function, for all three geometries. This isn't a "does it converge" check — it's a "is the algebra of the discretization actually self-consistent" check, and it holds regardless of grid resolution because it's a telescoping-sum identity, not an asymptotic approximation.

### 3.8 High-precision analytic Jacobian verification
For the parabolic solver's cross-diffusion term (`div(a_i u_i ∇ρ)`), we needed a full, correct Newton Jacobian including the derivative of the advection operator with respect to ρ — a nontrivial analytic derivative. We verified our hand-derived formula by comparing it against a numerical **finite-difference Jacobian** (perturb each variable by a small ε, measure the output change, divide by ε) — relative error < 1e-6. This is the standard way to check any hand-derived Jacobian/gradient before trusting it in a solver.

## 4. Every bug we found and fixed, chronologically

1. **Death term: constant vs. density-dependent.** The model equation has death rate `-b_i·ρ` (proportional to TOTAL local biomass — crowding mortality), but an earlier code version used a plain constant `-b_i`. Found by directly reading the rendered arXiv PDF equation (a prior automated PDF-text-extraction pass had silently dropped the ρ subscript, making the bug invisible to a naive grep/read). Fixed; added a regression test that checks the per-step growth RATIO strictly decreases under ample constant substrate (a constant death term would instead give perfectly constant-ratio exponential growth — a clean mathematical signature to test for).

2. **Row-0 (domain center) reaction zeroing.** An earlier version zeroed the reaction term at the r=0 grid point, based on a mistaken belief that r=0 is a "boundary condition row" needing special algebraic treatment like the OUTER boundary. It's not — it's a genuine interior physics point that just happens to use a symmetric (L'Hôpital-limit) stencil because of the coordinate singularity. Found via comparing the assembled Jacobian against a hand-built reference matrix for a trivial linear case. This bug degraded a genuinely 2nd-order scheme down to apparent 1st order.

3. **Near-center finite-volume cell-volume approximation.** Even after fixing #2, convergence order was still stuck ~1.7-1.9. Root cause: grid points immediately next to the center (i=1,2,3...) used the standard finite-volume shortcut `V_i ≈ r_i^p · h`, which has relative error O(h²/r_i²) — fine for r_i >> h, but 7.7% wrong at the very first interior point (i=1), because r_i itself is O(h) there. Fixed by computing the EXACT integrated cell volume at every row instead of the shortcut approximation.

4. **Neumann boundary residual target hardcoded to zero (found during exploratory testing of a different boundary condition type).** The model mostly uses Dirichlet boundary conditions (fixed concentration at the domain edge, representing a well-mixed external reservoir), but Neumann conditions (fixed FLUX at the boundary — e.g. modeling a controlled feed rate) are also implemented. We found THREE separate functions across two files all had a bug: they built the boundary equation's coefficient matrix correctly (respecting whatever flux value was requested) but then, completely separately, computed the residual/right-hand-side target using a hardcoded `0.0` instead of `-flux_value` — meaning ANY nonzero-flux Neumann request silently solved a ZERO-flux problem instead, with no error or warning. This is a subtle class of bug: two pieces of code (matrix construction and residual computation) that are supposed to represent the same equation but were built independently and drifted out of sync. Fixed in all three locations; added a regression test using a genuinely nonzero flux, checked two ways — against the analytic closed-form Neumann solution, AND by directly recomputing the boundary flux via finite-difference from the converged solution array and confirming it matches the requested value to 1e-14.

5. **A spurious/unphysical root problem (discovered stress-testing the Neumann fix in a "sealed system with active reaction" scenario).** With zero-flux boundaries (a sealed reactor) and reaction switched on, Newton's backtracking line search — which ONLY checked "did the residual decrease" — accepted a step that landed on a mathematically valid root (residual really was near zero) but a physically absurd one: NH4 concentration jumped to ~100 and NO3 to ~5400, both orders of magnitude above anything the sealed system could actually contain. Fixed by adding a physical-plausibility guard: reject any backtracking step (even one that reduces the residual) if it pushes any concentration above 5× the largest boundary/feed value in the whole problem.

6. **A genuine mathematical degeneracy (NOT a bug, but needed graceful handling).** In that same sealed-system scenario, once a substrate is legitimately driven to exactly zero everywhere (correct physics — no oxygen left in a sealed anoxic system), the Newton Jacobian can become EXACTLY singular (the linear system has no unique solution direction). This isn't wrong — it's what the math genuinely does at that point — but it needs to be detected and handled instead of crashing or returning NaN. Fixed by detecting non-finite Newton steps (or exhausted backtracking) and automatically falling back to the PTC/relaxation solver (§3.4), whose pseudo-time regularization can push through the degeneracy that plain Newton cannot.

7. **A sign bug introduced by fixing bug #5.** The new "plausibility upper bound" was computed as `5 × max(boundary_values)`. This silently breaks when a boundary value can be NEGATIVE — which a Neumann FLUX legitimately can be (negative flux = influx, into the domain). `max(-0.3, 0, 0, 0) = 0.0`, collapsing the bound to zero and clipping every trial concentration to exactly zero regardless of step size — a self-inflicted bug from not considering that "boundary value" means different physical things (concentration vs. flux) depending on boundary-condition type. Fixed by using the MAGNITUDE (`max(abs(v))`), not the signed maximum.

8. **Diagnostic mislabeling / fallback tolerance mismatch.** After adding the automatic fallback from bug #6's fix, the calling code could silently get a result that actually came from the fallback solver, but its own logging/diagnostics still labeled it "newton" — a transparency bug (wrong information, not wrong numbers). Also, the fallback used the fallback solver's own DEFAULT tolerance instead of whatever tolerance the original caller had actually requested. Fixed by changing the solver's return signature to report which method actually produced the answer, and threading the caller's tolerance through to the fallback.

## 5. What we deliberately did NOT claim as fixed (important scientific honesty piece)

- The Picard solver still doesn't converge for the realistic parameter set without a specific workaround — this is a genuine property of the fixed-point iteration's mathematics (confirmed via eigenvalue analysis), not something "fixable" by more careful coding.
- The standard entry point for Neumann boundary conditions, when applied to ALL FOUR coupled substrate equations simultaneously (as opposed to a simpler single-substrate test configuration), remains genuinely unreliable — it diverges or hangs on realistic multi-substrate configurations. We tested this directly and confirmed it's still broken, and documented it openly as a known limitation rather than assuming the narrower fix generalized.
- Several parameter-preset unit-cleaning decisions (e.g., a ÷100 yield rescale for one preset) are documented as assumptions taken on faith from the source data, not independently re-derived/verified.

## 6. A worked example of the "exploration → bug → fix → re-verify" cycle (useful for understanding the methodology)

We ran a competitive-dominance parameter scan: CMX and AOB compete for the same substrates (NH4, O2); under the realistic feed concentration, AOB always wins (it has a ~14× higher maximum growth rate). We swept the NH4 feed concentration down toward and below CMX's much lower half-saturation constant (CMX has ~23× higher NH4 AFFINITY despite its lower max rate — a classic rate-vs-affinity trade-off in microbial ecology) and found CMX genuinely CAN win, but only when NH4 feed is pushed to ~1/3000th of the realistic value — and even then, both species are essentially starved/stagnant (total biomass barely changes across the whole "crossover" region), so it's not a clean competitive victory, just "who decays slower." We reported this precisely rather than rounding it up to either "CMX dominance is achievable" or "CMX dominance never happens" — both would have been misleading simplifications of a nuanced, quantitative result.

---

**What I'd like you to help me understand better:**
1. The finite-volume method in more mathematical depth (why exact conservation, the flux-balance derivation, why the r=0 coordinate singularity needs the L'Hôpital-limit treatment)
2. Newton's method for nonlinear PDEs — the full theory behind backtracking line search and when/why it's needed
3. Why Picard/fixed-point iteration can fail to converge (spectral radius / contraction mapping theory) and how PTC (pseudo-transient continuation) fixes this
4. The floating-point catastrophic cancellation issue in more depth — why does subtracting two large near-equal numbers lose precision, and how does high-precision (mpmath) arithmetic sidestep it
5. Convergence order analysis — the theory behind why grid refinement studies work, and what "order 2" formally means (Taylor expansion / truncation error arguments)
6. Monod kinetics and the biological/ecological meaning of the rate-vs-affinity trade-off we found between AOB and CMX
