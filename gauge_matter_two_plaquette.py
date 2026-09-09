#!/usr/bin/env python3
"""Two-plaquette compact U(1) gauge--matter benchmarks.

The sparse calculation is performed in the finite-flux Gauss-law basis
AE=rho(o).  It generates the two main-text figures and their CSV data.  The
connected plaquette observable is projected only after forming
cos(Phi_1)cos(Phi_2); this avoids the extra intermediate projection present in
the earlier exploratory script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from paper_plot_style import (
    LINESTYLES,
    MARKERS,
    add_panel_label,
    add_top_legend,
    apply_paper_style,
    setup_axis,
    sparse_staggered_markevery,
)

apply_paper_style()


@dataclass(frozen=True)
class Lattice:
    Nx: int
    Ny: int
    sites: Tuple[Tuple[int, int], ...]
    links: Tuple[Tuple[int, int, str], ...]
    link_tail: np.ndarray
    link_head: np.ndarray
    A: np.ndarray
    B: np.ndarray
    eta: np.ndarray
    bkg: np.ndarray
    jw_order: np.ndarray
    rank_to_site: np.ndarray
    tree_parent: np.ndarray
    tree_link: np.ndarray
    tree_sign_at_child: np.ndarray
    bfs_order: np.ndarray

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_plaquettes(self) -> int:
        return self.B.shape[0]


@dataclass
class PhysicalBasis:
    L: int
    flux: np.ndarray
    occ: np.ndarray
    index: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]

    @property
    def dim(self) -> int:
        return self.flux.shape[0]


def site_index(Nx: int, x: int, y: int) -> int:
    return y * (Nx + 1) + x


def spanning_tree(
    n_sites: int, tails: np.ndarray, heads: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjacency: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_sites)]
    for r, (tail, head) in enumerate(zip(tails, heads)):
        adjacency[int(tail)].append((int(head), r, +1))
        adjacency[int(head)].append((int(tail), r, -1))
    parent = -np.ones(n_sites, dtype=int)
    tree_link = -np.ones(n_sites, dtype=int)
    sign_at_child = np.zeros(n_sites, dtype=int)
    parent[0] = 0
    queue = [0]
    order: List[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for other, r, sign_at_node in adjacency[node]:
            if parent[other] < 0:
                parent[other] = node
                tree_link[other] = r
                sign_at_child[other] = -sign_at_node
                queue.append(other)
    if np.any(parent < 0):
        raise AssertionError("Disconnected lattice graph.")
    return parent, tree_link, sign_at_child, np.array(order, dtype=int)


def build_lattice(Nx: int = 2, Ny: int = 1) -> Lattice:
    sites = tuple((x, y) for y in range(Ny + 1) for x in range(Nx + 1))
    links: List[Tuple[int, int, str]] = []
    for y in range(Ny + 1):
        for x in range(Nx):
            links.append((x, y, "x"))
    for y in range(Ny):
        for x in range(Nx + 1):
            links.append((x, y, "y"))
    link_index = {link: r for r, link in enumerate(links)}
    tails: List[int] = []
    heads: List[int] = []
    for x, y, direction in links:
        tails.append(site_index(Nx, x, y))
        heads.append(site_index(Nx, x + (direction == "x"), y + (direction == "y")))
    tails_array = np.array(tails, dtype=int)
    heads_array = np.array(heads, dtype=int)
    A = np.zeros((len(sites), len(links)), dtype=int)
    for r, (tail, head) in enumerate(zip(tails_array, heads_array)):
        A[tail, r] = +1
        A[head, r] = -1
    B = np.zeros((Nx * Ny, len(links)), dtype=int)
    p = 0
    for y in range(Ny):
        for x in range(Nx):
            B[p, link_index[(x, y, "x")]] += 1
            B[p, link_index[(x + 1, y, "y")]] += 1
            B[p, link_index[(x, y + 1, "x")]] -= 1
            B[p, link_index[(x, y, "y")]] -= 1
            p += 1
    eta = np.array([(-1) ** (x + y) for x, y in sites], dtype=int)
    bkg = ((1 - eta) // 2).astype(int)
    snake: List[int] = []
    for y in range(Ny + 1):
        xs = range(Nx + 1) if y % 2 == 0 else range(Nx, -1, -1)
        snake.extend(site_index(Nx, x, y) for x in xs)
    rank_to_site = np.array(snake, dtype=int)
    jw_order = np.empty(len(sites), dtype=int)
    for rank, site in enumerate(rank_to_site):
        jw_order[site] = rank
    parent, tree_link, sign_at_child, order = spanning_tree(
        len(sites), tails_array, heads_array
    )
    lattice = Lattice(
        Nx=Nx,
        Ny=Ny,
        sites=sites,
        links=tuple(links),
        link_tail=tails_array,
        link_head=heads_array,
        A=A,
        B=B,
        eta=eta,
        bkg=bkg,
        jw_order=jw_order,
        rank_to_site=rank_to_site,
        tree_parent=parent,
        tree_link=tree_link,
        tree_sign_at_child=sign_at_child,
        bfs_order=order,
    )
    if not np.array_equal(A @ B.T, np.zeros((len(sites), Nx * Ny), dtype=int)):
        raise AssertionError("AB^T is not zero.")
    return lattice


def particular_flux(lattice: Lattice, rho: np.ndarray) -> Optional[np.ndarray]:
    if int(np.sum(rho)) != 0:
        return None
    flux = np.zeros(lattice.n_links, dtype=int)
    accumulated = np.zeros(lattice.n_sites, dtype=int)
    for node in lattice.bfs_order[::-1]:
        if node == 0:
            continue
        residual = int(rho[node] - accumulated[node])
        r = int(lattice.tree_link[node])
        sign = int(lattice.tree_sign_at_child[node])
        flux[r] = residual * sign
        parent = int(lattice.tree_parent[node])
        accumulated[parent] += -sign * flux[r]
    if not np.array_equal(lattice.A @ flux, rho):
        raise AssertionError("Tree solution failed Gauss's law.")
    return flux


def build_basis(lattice: Lattice, L: int) -> PhysicalBasis:
    fluxes: List[np.ndarray] = []
    occupations: List[np.ndarray] = []
    index: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    for occ_tuple in product((0, 1), repeat=lattice.n_sites):
        occ = np.array(occ_tuple, dtype=int)
        rho = occ - lattice.bkg
        flux0 = particular_flux(lattice, rho)
        if flux0 is None:
            continue
        # This bound is complete for the 2x1 open strip used here because each
        # plaquette has an exterior boundary link.
        h_bound = L + int(np.max(np.abs(flux0))) + lattice.n_plaquettes
        for h_tuple in product(range(-h_bound, h_bound + 1), repeat=lattice.n_plaquettes):
            flux = flux0 + lattice.B.T @ np.array(h_tuple, dtype=int)
            if np.all(np.abs(flux) <= L):
                key = (tuple(map(int, flux)), tuple(map(int, occ)))
                if key not in index:
                    index[key] = len(fluxes)
                    fluxes.append(flux.copy())
                    occupations.append(occ.copy())
    basis = PhysicalBasis(
        L=L,
        flux=np.vstack(fluxes),
        occ=np.vstack(occupations),
        index=index,
    )
    residual = basis.flux @ lattice.A.T - (basis.occ - lattice.bkg[None, :])
    if np.any(residual):
        raise AssertionError("Physical basis contains a Gauss-law violation.")
    return basis


def jw_sign(lattice: Lattice, occ: np.ndarray, a: int, b: int) -> int:
    lo, hi = sorted((int(lattice.jw_order[a]), int(lattice.jw_order[b])))
    sites = lattice.rank_to_site[lo + 1 : hi]
    return -1 if int(np.sum(occ[sites])) % 2 else +1


def build_hamiltonian(
    lattice: Lattice,
    basis: PhysicalBasis,
    g_inv2: float,
    m0: float,
    kappa: float,
    a_lattice: float = 1.0,
) -> sp.csr_matrix:
    dim = basis.dim
    g2 = 1.0 / g_inv2
    magnetic_prefactor = g_inv2 / (a_lattice * a_lattice)
    diagonal = (
        0.5 * g2 * np.sum(basis.flux.astype(float) ** 2, axis=1)
        + m0 * (basis.occ @ lattice.eta.astype(float))
        + lattice.n_plaquettes * magnetic_prefactor
    )
    rows: List[int] = list(range(dim))
    cols: List[int] = list(range(dim))
    data: List[float] = list(map(float, diagonal))
    for source, (flux, occ) in enumerate(zip(basis.flux, basis.occ)):
        occ_key = tuple(map(int, occ))
        for boundary in lattice.B:
            for direction in (-1, +1):
                key = (
                    tuple(map(int, flux + direction * boundary)),
                    occ_key,
                )
                target = basis.index.get(key)
                if target is not None:
                    rows.append(target)
                    cols.append(source)
                    data.append(-0.5 * magnetic_prefactor)
        for r, (a, b) in enumerate(zip(lattice.link_tail, lattice.link_head)):
            a = int(a)
            b = int(b)
            sign = jw_sign(lattice, occ, a, b)
            if occ[a] == 0 and occ[b] == 1:
                new_occ = occ.copy()
                new_occ[a], new_occ[b] = 1, 0
                new_flux = flux.copy()
                new_flux[r] += 1
                target = basis.index.get(
                    (tuple(map(int, new_flux)), tuple(map(int, new_occ)))
                )
                if target is not None:
                    rows.append(target)
                    cols.append(source)
                    data.append(0.5 * kappa * sign)
            if occ[a] == 1 and occ[b] == 0:
                new_occ = occ.copy()
                new_occ[a], new_occ[b] = 0, 1
                new_flux = flux.copy()
                new_flux[r] -= 1
                target = basis.index.get(
                    (tuple(map(int, new_flux)), tuple(map(int, new_occ)))
                )
                if target is not None:
                    rows.append(target)
                    cols.append(source)
                    data.append(0.5 * kappa * sign)
    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()
    if H.nnz and np.max(np.abs((H - H.T).data), initial=0.0) > 1e-13:
        raise AssertionError("Hamiltonian is not Hermitian.")
    return H


def plaquette_cosines(
    lattice: Lattice, basis: PhysicalBasis
) -> Tuple[List[sp.csr_matrix], sp.csr_matrix]:
    dim = basis.dim
    single: List[sp.csr_matrix] = []
    for boundary in lattice.B:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for source, (flux, occ) in enumerate(zip(basis.flux, basis.occ)):
            occ_key = tuple(map(int, occ))
            for direction in (-1, +1):
                target = basis.index.get(
                    (tuple(map(int, flux + direction * boundary)), occ_key)
                )
                if target is not None:
                    rows.append(target)
                    cols.append(source)
                    data.append(0.5)
        single.append(
            sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
        )
    if lattice.n_plaquettes != 2:
        raise ValueError("The composite observable is implemented for two plaquettes.")
    rows = []
    cols = []
    data = []
    for source, (flux, occ) in enumerate(zip(basis.flux, basis.occ)):
        occ_key = tuple(map(int, occ))
        for d0, d1 in product((-1, +1), repeat=2):
            new_flux = flux + d0 * lattice.B[0] + d1 * lattice.B[1]
            target = basis.index.get((tuple(map(int, new_flux)), occ_key))
            if target is not None:
                rows.append(target)
                cols.append(source)
                data.append(0.25)
    composite = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    if composite.nnz and np.max(np.abs((composite - composite.T).data), initial=0.0) > 1e-13:
        raise AssertionError("Composite plaquette observable is not Hermitian.")
    return single, composite


def diagonal_observables(lattice: Lattice, basis: PhysicalBasis) -> Dict[str, np.ndarray]:
    occ = basis.occ.astype(float)
    odd = (lattice.eta == -1).astype(float)
    even = (lattice.eta == +1).astype(float)
    return {
        "E2_link": np.mean(basis.flux.astype(float) ** 2, axis=1),
        "boundary_per_link": np.mean(np.abs(basis.flux) == basis.L, axis=1),
        "I_M": occ @ odd - occ @ even,
        "n_total": np.sum(occ, axis=1),
    }


def expect_diag(psi: np.ndarray, values: np.ndarray) -> float:
    return float(np.dot(np.abs(psi) ** 2, values).real)


def expect_op(psi: np.ndarray, operator: sp.csr_matrix) -> float:
    return float(np.vdot(psi, operator @ psi).real)


def lowest_two(H: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, float]:
    dim = H.shape[0]
    if dim <= 160:
        values, vectors = la.eigh(H.toarray())
        residual = max(
            la.norm(H @ vectors[:, j] - values[j] * vectors[:, j]) for j in range(2)
        )
        return values[:2], vectors[:, :2], float(residual)
    abs_rows = np.asarray(np.abs(H).sum(axis=1)).ravel()
    diagonal = H.diagonal()
    lower_bound = float(np.min(diagonal - (abs_rows - np.abs(diagonal))))
    sigma = lower_bound - max(1e-4, 1e-6 * (1 + abs(lower_bound)))
    rng = np.random.default_rng(20260710 + dim)
    v0 = rng.standard_normal(dim)
    v0 /= la.norm(v0)
    values, vectors = spla.eigsh(
        H,
        k=2,
        sigma=sigma,
        which="LM",
        tol=2e-11,
        maxiter=100000,
        v0=v0,
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    residual = max(
        la.norm(H @ vectors[:, j] - values[j] * vectors[:, j]) for j in range(2)
    )
    if residual > 2e-7:
        raise AssertionError(f"Eigenpair residual is too large: {residual}")
    return values, vectors, float(residual)


def static_scan(
    lattice: Lattice,
    L_values: Sequence[int],
    g_grid: np.ndarray,
    m0: float,
    kappa: float,
) -> Tuple[List[Dict[str, float]], Dict[int, int], float]:
    rows: List[Dict[str, float]] = []
    dimensions: Dict[int, int] = {}
    max_residual = 0.0
    for L in L_values:
        basis = build_basis(lattice, L)
        dimensions[L] = basis.dim
        cosines, _ = plaquette_cosines(lattice, basis)
        diagonal = diagonal_observables(lattice, basis)
        for g_inv2 in g_grid:
            H = build_hamiltonian(lattice, basis, g_inv2, m0, kappa)
            values, vectors, residual = lowest_two(H)
            max_residual = max(max_residual, residual)
            psi = vectors[:, 0]
            W = [expect_op(psi, operator) for operator in cosines]
            rows.append(
                {
                    "L": L,
                    "dim": basis.dim,
                    "g_inv2": float(g_inv2),
                    "E0": float(values[0]),
                    "gap": float(values[1] - values[0]),
                    "Wbar": float(np.mean(W)),
                    "E2_link": expect_diag(psi, diagonal["E2_link"]),
                    "I_M": expect_diag(psi, diagonal["I_M"]),
                    "boundary_per_link": expect_diag(
                        psi, diagonal["boundary_per_link"]
                    ),
                }
            )
    return rows, dimensions, max_residual


def strong_coupling_index(lattice: Lattice, basis: PhysicalBasis) -> int:
    key = (
        tuple([0] * lattice.n_links),
        tuple(map(int, lattice.bkg)),
    )
    if key not in basis.index:
        raise AssertionError("Strong-coupling state is absent from the basis.")
    return basis.index[key]


def dynamics(
    lattice: Lattice,
    L_values: Sequence[int],
    times: np.ndarray,
    g_inv2: float,
    m0: float,
    kappa: float,
) -> Tuple[List[Dict[str, float]], float, float]:
    if len(times) < 2 or not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("Dynamics times must be equally spaced.")
    rows: List[Dict[str, float]] = []
    max_norm_error = 0.0
    max_number_error = 0.0
    initial_number = float(np.sum(lattice.bkg))
    for L in L_values:
        basis = build_basis(lattice, L)
        H = build_hamiltonian(lattice, basis, g_inv2, m0, kappa)
        cosines, composite = plaquette_cosines(lattice, basis)
        diagonal = diagonal_observables(lattice, basis)
        psi0 = np.zeros(basis.dim, dtype=complex)
        psi0[strong_coupling_index(lattice, basis)] = 1.0
        states = spla.expm_multiply(
            -1j * H,
            psi0,
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )
        for time, psi in zip(times, states):
            W = [expect_op(psi, operator) for operator in cosines]
            norm = float(np.vdot(psi, psi).real)
            n_total = expect_diag(psi, diagonal["n_total"])
            max_norm_error = max(max_norm_error, abs(norm - 1.0))
            max_number_error = max(max_number_error, abs(n_total - initial_number))
            rows.append(
                {
                    "L": L,
                    "dim": basis.dim,
                    "time": float(time),
                    "g_inv2": g_inv2,
                    "Wbar": float(np.mean(W)),
                    "C12": expect_op(psi, composite) - W[0] * W[1],
                    "I_M": expect_diag(psi, diagonal["I_M"]),
                    "boundary_per_link": expect_diag(
                        psi, diagonal["boundary_per_link"]
                    ),
                    "E2_link": expect_diag(psi, diagonal["E2_link"]),
                    "n_total": n_total,
                    "norm": norm,
                }
            )
    if max_norm_error > 2e-10 or max_number_error > 2e-10:
        raise AssertionError(
            f"Conservation check failed: norm={max_norm_error}, number={max_number_error}"
        )
    return rows, max_norm_error, max_number_error


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_static(rows: Sequence[Dict[str, float]], outbase: Path) -> None:
    fig = plt.figure(figsize=(13.9, 8.7))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.10, 1.0],
        hspace=0.42,
        wspace=0.32,
    )
    axes = (
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )
    L_values = sorted({int(r["L"]) for r in rows})
    handles = []
    legend_labels = []
    for curve_idx, L in enumerate(L_values):
        selected = sorted((r for r in rows if r["L"] == L), key=lambda r: r["g_inv2"])
        x = np.array([r["g_inv2"] for r in selected])
        style = dict(
            marker=MARKERS[curve_idx % len(MARKERS)],
            linestyle=LINESTYLES[curve_idx % len(LINESTYLES)],
            markevery=sparse_staggered_markevery(
                len(x), curve_idx, len(L_values), target_markers=5
            ),
            markersize=4.3,
            linewidth=1.45,
        )
        handle, = axes[0].plot(
            x, [r["Wbar"] for r in selected], label=rf"$L={L}$", **style
        )
        axes[1].plot(x, [r["gap"] for r in selected], **style)
        axes[2].plot(x, [r["E2_link"] for r in selected], **style)
        handles.append(handle)
        legend_labels.append(rf"$L={L}$")

    axes[0].set_ylabel(r"$\overline{W}$")
    axes[1].set_ylabel(r"$\Delta=\mathcal{E}_1-\mathcal{E}_0$")
    axes[2].set_ylabel(r"$\langle E^2\rangle_{\rm link}$")
    setup_axis(axes[0], xlog=True)
    setup_axis(axes[1], xlog=True, ylog=True)
    setup_axis(axes[2], xlog=True, ylog=True)
    for ax in axes:
        ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0, zorder=1.5)
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.set_xlabel(r"$g^{-2}$")
        add_panel_label(ax, label)
    fig.subplots_adjust(top=0.88)
    add_top_legend(
        fig, handles, legend_labels, ncol=min(5, len(legend_labels)), y=0.985
    )
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"))
    plt.close(fig)


def plot_dynamics(rows: Sequence[Dict[str, float]], outbase: Path) -> None:
    fig = plt.figure(figsize=(14.5, 9.0))
    grid = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )
    keys = ("Wbar", "C12", "I_M", "boundary_per_link")
    labels = (
        r"$\overline{W}(t)$",
        r"$C_{12}(t)$",
        r"$I_M(t)$",
        r"$\overline{p}_{\partial L}(t)$",
    )
    L_values = sorted({int(r["L"]) for r in rows})
    handles = []
    legend_labels = []
    for curve_idx, L in enumerate(L_values):
        selected = sorted((r for r in rows if r["L"] == L), key=lambda r: r["time"])
        time = np.array([r["time"] for r in selected])
        style = dict(
            marker=MARKERS[curve_idx % len(MARKERS)],
            linestyle=LINESTYLES[curve_idx % len(LINESTYLES)],
            markevery=sparse_staggered_markevery(
                len(time), curve_idx, len(L_values), target_markers=5
            ),
            markersize=4.0,
            linewidth=1.30,
        )
        for panel_idx, (ax, key) in enumerate(zip(axes, keys)):
            values = [max(r[key], 1e-16) if key == "boundary_per_link" else r[key] for r in selected]
            line, = ax.plot(time, values, label=rf"$L={L}$", **style)
            if panel_idx == 0:
                handles.append(line)
                legend_labels.append(rf"$L={L}$")
    for ax in axes[:3]:
        setup_axis(ax)
    setup_axis(axes[3], ylog=True)
    for panel, ax, ylabel in zip(("(a)", "(b)", "(c)", "(d)"), axes, labels):
        ax.set_xlabel(r"time $t$")
        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel)
    fig.subplots_adjust(top=0.88)
    add_top_legend(
        fig, handles, legend_labels, ncol=min(4, len(legend_labels)), y=0.995
    )
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"))
    plt.close(fig)


def parse_int_list(text: str) -> List[int]:
    values = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values or min(values) < 1:
        raise ValueError("Cutoffs must be positive integers.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("../figures"))
    parser.add_argument("--datadir", type=Path, default=Path("../data"))
    parser.add_argument("--L-static", type=str, default="1,2,3,4,6")
    parser.add_argument("--L-dynamics", type=str, default="1,2,3,4")
    parser.add_argument("--g-min", type=float, default=1e-2)
    parser.add_argument("--g-max", type=float, default=10.0)
    parser.add_argument("--g-count", type=int, default=55)
    parser.add_argument("--g-dynamics", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--t-count", type=int, default=241)
    parser.add_argument("--m0", type=float, default=1.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    lattice = build_lattice(2, 1)
    L_static = parse_int_list(args.L_static)
    L_dynamics = parse_int_list(args.L_dynamics)
    g_count = min(args.g_count, 18) if args.fast else args.g_count
    t_count = min(args.t_count, 81) if args.fast else args.t_count
    if args.fast:
        L_static = [1, 2, 3]
        L_dynamics = [1, 2, 3]
    g_grid = np.geomspace(args.g_min, args.g_max, g_count)
    times = np.linspace(0.0, args.t_max, t_count)
    static_rows, dimensions, eigen_residual = static_scan(
        lattice, L_static, g_grid, args.m0, args.kappa
    )
    dynamic_rows, norm_error, number_error = dynamics(
        lattice,
        L_dynamics,
        times,
        args.g_dynamics,
        args.m0,
        args.kappa,
    )
    args.datadir.mkdir(parents=True, exist_ok=True)
    write_csv(args.datadir / "gauge_matter_two_plaquette_static.csv", static_rows)
    write_csv(args.datadir / "gauge_matter_two_plaquette_dynamics.csv", dynamic_rows)
    plot_static(static_rows, args.outdir / "gauge_matter_two_plaquette_static")
    plot_dynamics(dynamic_rows, args.outdir / "gauge_matter_two_plaquette_dynamics")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "model": "2x1 open compact U(1) gauge-matter model in the Gauss-law basis",
        "sites": lattice.n_sites,
        "links": lattice.n_links,
        "plaquettes": lattice.n_plaquettes,
        "link_phase": 0.0,
        "L_static": L_static,
        "L_dynamics": L_dynamics,
        "basis_dimensions": {str(k): int(v) for k, v in dimensions.items()},
        "g_inv2_range": [args.g_min, args.g_max],
        "g_count": g_count,
        "g_inv2_dynamics": args.g_dynamics,
        "time_range": [0.0, args.t_max],
        "time_count": t_count,
        "m0": args.m0,
        "kappa": args.kappa,
        "a_lattice": 1.0,
        "max_eigenpair_residual": eigen_residual,
        "max_norm_error": norm_error,
        "max_matter_number_error": number_error,
        "composite_C12_projection": "P_L cos(Phi_1) cos(Phi_2) P_L",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "script_sha256": source_hash,
    }
    with (args.datadir / "gauge_matter_two_plaquette_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
