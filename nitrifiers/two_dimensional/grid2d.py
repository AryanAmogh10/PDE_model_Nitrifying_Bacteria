"""2D Cartesian grid, conservative FV Laplacian, boundary-condition application.
Fields are (Nx+1, Ny+1) arrays, C-order flattened: k = i*(Ny+1) + j."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Grid2D:
    """Vertex-centred uniform Cartesian grid on [0, Lx] x [0, Ly]."""

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

        self.dx = np.full(Nx + 1, self.hx)
        self.dx[0] = self.dx[-1] = 0.5 * self.hx
        self.dy = np.full(Ny + 1, self.hy)
        self.dy[0] = self.dy[-1] = 0.5 * self.hy

        self.V = self.dx[:, None] * self.dy[None, :]
        self.Vflat = self.V.ravel()

        self.X = self.x[:, None] * np.ones((1, Ny + 1))
        self.Y = np.ones((Nx + 1, 1)) * self.y[None, :]

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
        x0 = 0.5 * self.Lx if x0 is None else x0
        y0 = 0.5 * self.Ly if y0 is None else y0
        return np.sqrt((self.X - x0) ** 2 + (self.Y - y0) ** 2)

    def integrate(self, field: np.ndarray) -> float:
        return float(np.sum(self.V * np.asarray(field).reshape(self.shape)))


def build_laplacian_2d(grid: Grid2D) -> sp.csr_matrix:
    """5-point conservative FV Laplacian, exact control volumes. Edges are
    zero-flux by construction (no face there); apply_bc_2d overwrites them
    for Dirichlet."""
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
            if i < Nx:
                w = 1.0 / (hx * dx[i])
                rows.append(kc); cols.append(k(i + 1, j)); vals.append(w)
                diag -= w
            if i > 0:
                w = 1.0 / (hx * dx[i])
                rows.append(kc); cols.append(k(i - 1, j)); vals.append(w)
                diag -= w
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
    """'dirichlet' replaces boundary rows with c_k = value. 'neumann' and
    'robin' leave rows untouched (zero-flux by construction; robin's Bi
    coupling is added to the Jacobian in elliptic2d, not here)."""
    if bc_type in ("neumann", "robin"):
        return Lap.tocsr()
    if bc_type != "dirichlet":
        raise ValueError("bc_type must be 'dirichlet', 'neumann' or 'robin'")

    nodes = grid.boundary_flat if boundary_nodes is None else np.asarray(boundary_nodes)
    Lap = Lap.tolil()
    for kk in nodes:
        Lap.rows[kk] = [int(kk)]
        Lap.data[kk] = [1.0]
    return Lap.tocsr()


def boundary_flux_coefficients(grid: Grid2D) -> np.ndarray:
    """Per-node weight so a prescribed Neumann flux `value` adds on as a
    source term. Same sign convention as 1D (-dc/dn = value, negative =
    influx). Zero except at the boundary; corners pick up two terms."""
    Nx, Ny = grid.Nx, grid.Ny
    coef = np.zeros(grid.shape)
    coef[0, :] += -1.0 / grid.dx[0]
    coef[Nx, :] += -1.0 / grid.dx[Nx]
    coef[:, 0] += -1.0 / grid.dy[0]
    coef[:, Ny] += -1.0 / grid.dy[Ny]
    return coef.ravel()
