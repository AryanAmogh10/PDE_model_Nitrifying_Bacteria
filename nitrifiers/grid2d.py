"""
2D Cartesian spatial core: grid, conservative finite-volume Laplacian, and
boundary-condition application. This is the 2D analogue of the (fixed,
exact-cell-volume) 1D machinery in elliptic.py, and deliberately follows the
same derivation philosophy:

    Lap_ij = [ F_east*A_east - F_west*A_west + F_north*A_north - F_south*A_south ] / V_ij

with every face area and every control volume computed EXACTLY, never via the
"interior shortcut" V ~= (coordinate)^p * h that caused the near-centre
convergence-order defect found and fixed in the 1D solver. In 2D Cartesian
there is no coordinate singularity and no r^p weighting, so the exact volume of
the control cell around node (i,j) is simply

    V_ij = dx_i * dy_j,   dx_i = hx (interior) or hx/2 (i = 0 or Nx),

i.e. the half-cell treatment at the four domain edges is the direct analogue of
the 1D outer-boundary half-cell. Because the transverse face area cancels
against the volume in each direction (A_east/V_ij = dy_j/(dx_i*dy_j) = 1/dx_i),
the resulting stencil is the standard 5-point Laplacian in the interior and
picks up the expected factor of 2 on edge rows -- exactly mirroring the 1D
row-0 / row-N factor.

NODE LAYOUT AND FLATTENING. Fields are (Nx+1, Ny+1) arrays indexed [i, j] for
the point (x_i, y_j), flattened C-order (so j varies fastest):

    k = i*(Ny+1) + j     <->     field.ravel()

Every operator here is built against that convention; use `Grid2D.ravel` /
`Grid2D.unravel` rather than hand-rolling the index arithmetic.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Grid2D:
    """Vertex-centred uniform Cartesian grid on [0, Lx] x [0, Ly].

    Exposes the EXACT per-node control-volume areas (`V`, and its flattened
    view `Vflat`), which are what the conservation identity in
    build_laplacian_2d telescopes against -- any test or diagnostic that
    integrates a field over the domain must weight by these, not by a uniform
    hx*hy, or it is silently measuring the volume-normalisation error rather
    than the quantity it claims to.
    """

    def __init__(self, Nx: int = 50, Ny: int = 50, Lx: float = 1.0, Ly: float = 1.0):
        if Nx < 2 or Ny < 2:
            raise ValueError("Nx and Ny must be >= 2")
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.hx = Lx / Nx
        self.hy = Ly / Ny
        self.x = np.linspace(0.0, Lx, Nx + 1)
        self.y = np.linspace(0.0, Ly, Ny + 1)
        self.shape = (Nx + 1, Ny + 1)
        self.Npts = (Nx + 1) * (Ny + 1)

        # exact 1D control-cell widths: full cell inside, half cell at each end
        self.dx = np.full(Nx + 1, self.hx)
        self.dx[0] = self.dx[-1] = 0.5 * self.hx
        self.dy = np.full(Ny + 1, self.hy)
        self.dy[0] = self.dy[-1] = 0.5 * self.hy

        # exact 2D control-volume (area) per node
        self.V = self.dx[:, None] * self.dy[None, :]
        self.Vflat = self.V.ravel()

        # node coordinates as 2D fields, same [i, j] convention
        self.X = self.x[:, None] * np.ones((1, Ny + 1))
        self.Y = np.ones((Nx + 1, 1)) * self.y[None, :]

        # boundary bookkeeping (flattened indices of the four edges)
        mask = np.zeros(self.shape, dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        self.boundary_mask = mask
        self.boundary_flat = np.flatnonzero(mask.ravel())
        self.interior_flat = np.flatnonzero(~mask.ravel())

    def ravel(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field).ravel()

    def unravel(self, vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec).reshape(self.shape)

    def radius(self, x0: float | None = None, y0: float | None = None) -> np.ndarray:
        """Distance field from (x0, y0), defaulting to the domain centre."""
        x0 = 0.5 * self.Lx if x0 is None else x0
        y0 = 0.5 * self.Ly if y0 is None else y0
        return np.sqrt((self.X - x0) ** 2 + (self.Y - y0) ** 2)

    def integrate(self, field: np.ndarray) -> float:
        """Exact-control-volume integral of a field over the domain."""
        return float(np.sum(self.V * np.asarray(field).reshape(self.shape)))


def build_laplacian_2d(grid: Grid2D) -> sp.csr_matrix:
    """Conservative 5-point finite-volume Laplacian with EXACT control volumes.

    All four domain edges carry a natural homogeneous-Neumann (zero-flux)
    condition, arising simply from the absence of a face there -- exactly as
    the 1D build_laplacian's row 0 and row N do before apply_bc overwrites
    them. Consequently the operator satisfies the discrete conservation
    identity

        sum_ij V_ij * (Lap @ c)_ij = 0      for ANY field c,

    to machine precision and at ANY resolution (it is a telescoping-flux
    identity, not an asymptotic statement). apply_bc_2d overwrites the edge
    rows when a Dirichlet condition is wanted.
    """
    Nx, Ny = grid.Nx, grid.Ny
    hx, hy = grid.hx, grid.hy
    dx, dy = grid.dx, grid.dy
    ny1 = Ny + 1

    rows, cols, vals = [], [], []

    def k(i, j):
        return i * ny1 + j

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            kc = k(i, j)
            diag = 0.0
            # x-direction faces: A_face/V = dy_j/(dx_i*dy_j) = 1/dx_i
            if i < Nx:
                w = 1.0 / (hx * dx[i])
                rows.append(kc); cols.append(k(i + 1, j)); vals.append(w)
                diag -= w
            if i > 0:
                w = 1.0 / (hx * dx[i])
                rows.append(kc); cols.append(k(i - 1, j)); vals.append(w)
                diag -= w
            # y-direction faces: A_face/V = dx_i/(dx_i*dy_j) = 1/dy_j
            if j < Ny:
                w = 1.0 / (hy * dy[j])
                rows.append(kc); cols.append(k(i, j + 1)); vals.append(w)
                diag -= w
            if j > 0:
                w = 1.0 / (hy * dy[j])
                rows.append(kc); cols.append(k(i, j - 1)); vals.append(w)
                diag -= w
            rows.append(kc); cols.append(kc); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(grid.Npts, grid.Npts))


def apply_bc_2d(Lap: sp.csr_matrix, grid: Grid2D, bc_type: str,
                 boundary_nodes: np.ndarray | None = None) -> sp.csr_matrix:
    """Overwrite boundary rows with the chosen outer BC.

    'dirichlet' replaces each boundary row by the algebraic identity row
    c_k = value (the value itself lives in the residual/RHS, matching the 1D
    convention). 'neumann' leaves the rows untouched: build_laplacian_2d
    already encodes zero flux there by construction, so a homogeneous-Neumann
    solve needs no row replacement at all.

    `boundary_nodes` defaults to the four domain edges; pass an explicit index
    array to impose Dirichlet data on an embedded (e.g. staircased circular)
    boundary instead.
    """
    if bc_type == "neumann":
        return Lap.tocsr()
    if bc_type != "dirichlet":
        raise ValueError("bc_type must be 'dirichlet' or 'neumann'")

    nodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)
    Lap = Lap.tolil()
    for kk in nodes:
        Lap.rows[kk] = [int(kk)]
        Lap.data[kk] = [1.0]
    return Lap.tocsr()
