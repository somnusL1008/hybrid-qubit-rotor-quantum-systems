from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal

from paper_plot_style import (
    LINESTYLES,
    MARKERS,
    apply_paper_style,
    setup_axis,
    sparse_staggered_markevery,
)

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
L = 32
MATHIEU_CUTOFF = 192
BISECTION_STEPS = 55

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "plots"
STATE_FILE = DATA_DIR / "qpe_probe_states.npz"
CSV_FILE = DATA_DIR / "qpe_probe_momentum.csv"
FIG_PDF = FIG_DIR / "qpe_probe_momentum.pdf"
FIG_PNG = FIG_DIR / "qpe_probe_momentum.png"


def kinetic_energy(ell: np.ndarray, amplitudes: np.ndarray) -> float:
    return float(np.sum(ell**2 * np.abs(amplitudes) ** 2))


def circular_variance(amplitudes: np.ndarray) -> float:
    resultant = abs(np.sum(amplitudes[:-1] * np.conj(amplitudes[1:])))
    return float(1.0 - resultant)


def mathieu_ground_state(
    lam: float, ell: np.ndarray
) -> tuple[np.ndarray, float]:
    """Ground state of H_lambda = ell^2 - lambda cos(theta).

    In the momentum basis, cos(theta) has 1/2 on the first off-diagonals.
    """
    diagonal = ell.astype(float) ** 2
    off_diagonal = -0.5 * lam * np.ones(len(ell) - 1)
    _, eigenvectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, 0),
        check_finite=False,
    )
    amplitudes = eigenvectors[:, 0].astype(float)

    # Fix the irrelevant global sign so the state is positive at ell = 0.
    center = np.argmin(np.abs(ell))
    if amplitudes[center] < 0:
        amplitudes *= -1.0

    amplitudes /= np.linalg.norm(amplitudes)
    return amplitudes, kinetic_energy(ell, amplitudes)


def match_mathieu_energy(
    target_energy: float, ell: np.ndarray
) -> tuple[float, np.ndarray]:
    """Choose lambda so the Mathieu ground state has the requested <ell^2>."""
    lo = 0.0
    hi = 1.0

    while True:
        _, energy_hi = mathieu_ground_state(hi, ell)
        if energy_hi >= target_energy:
            break
        hi *= 2.0
        if hi > 1.0e9:
            raise RuntimeError("Could not bracket the Mathieu energy target.")

    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        _, energy_mid = mathieu_ground_state(mid, ell)
        if energy_mid < target_energy:
            lo = mid
        else:
            hi = mid

    lam = 0.5 * (lo + hi)
    amplitudes, _ = mathieu_ground_state(lam, ell)
    return lam, amplitudes


def generate_probe_data() -> dict[str, np.ndarray | float]:
    """Generate the three probe states and save momentum-space data.

    Uniform and cosine probes use the same finite support |ell| <= L.
    The Mathieu probe is matched to the cosine probe's mean kinetic energy,
    which makes the cosine--Mathieu comparison a fixed-energy comparison.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ell = np.arange(-MATHIEU_CUTOFF, MATHIEU_CUTOFF + 1, dtype=int)
    support = np.abs(ell) <= L

    a_uniform = np.zeros(len(ell), dtype=float)
    a_uniform[support] = 1.0 / np.sqrt(2 * L + 1)

    a_cosine = np.zeros(len(ell), dtype=float)
    a_cosine[support] = (
        np.cos(np.pi * ell[support] / (2.0 * (L + 1))) / np.sqrt(L + 1)
    )

    raw_norm_uniform = float(np.sum(np.abs(a_uniform) ** 2))
    raw_norm_cosine = float(np.sum(np.abs(a_cosine) ** 2))
    if not np.isclose(raw_norm_uniform, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError(
            f"Uniform probe is not normalized: {raw_norm_uniform:.16e}"
        )
    if not np.isclose(raw_norm_cosine, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError(
            f"Cosine-window probe is not normalized: {raw_norm_cosine:.16e}"
        )

    # Both formulas are exactly normalized; remove floating-point roundoff.
    a_uniform /= np.linalg.norm(a_uniform)
    a_cosine /= np.linalg.norm(a_cosine)

    energy_uniform = kinetic_energy(ell, a_uniform)
    energy_cosine = kinetic_energy(ell, a_cosine)

    lam_mathieu, a_mathieu = match_mathieu_energy(energy_cosine, ell)
    energy_mathieu = kinetic_energy(ell, a_mathieu)
    energy_match_error = abs(energy_mathieu - energy_cosine)
    if energy_match_error > 1.0e-10:
        raise RuntimeError(
            "Mathieu probe does not match the cosine-window kinetic energy: "
            f"|Delta E| = {energy_match_error:.6e}"
        )

    norm_uniform = float(np.sum(np.abs(a_uniform) ** 2))
    norm_cosine = float(np.sum(np.abs(a_cosine) ** 2))
    norm_mathieu = float(np.sum(np.abs(a_mathieu) ** 2))
    for name, normalization in (
        ("uniform", norm_uniform),
        ("cosine-window", norm_cosine),
        ("Mathieu", norm_mathieu),
    ):
        if not np.isclose(normalization, 1.0, atol=2.0e-14, rtol=0.0):
            raise RuntimeError(
                f"{name} momentum amplitudes are not normalized: "
                f"{normalization:.16e}"
            )

    mathieu_edge_probability = float(
        np.abs(a_mathieu[0]) ** 2 + np.abs(a_mathieu[-1]) ** 2
    )

    variance_uniform = circular_variance(a_uniform)
    variance_cosine = circular_variance(a_cosine)
    variance_mathieu = circular_variance(a_mathieu)

    np.savez_compressed(
        STATE_FILE,
        ell=ell,
        a_uniform=a_uniform,
        a_cosine=a_cosine,
        a_mathieu=a_mathieu,
        L=np.array(L),
        lambda_mathieu=np.array(lam_mathieu),
        energy_uniform=np.array(energy_uniform),
        energy_cosine=np.array(energy_cosine),
        energy_mathieu=np.array(energy_mathieu),
        energy_match_error=np.array(energy_match_error),
        variance_uniform=np.array(variance_uniform),
        variance_cosine=np.array(variance_cosine),
        variance_mathieu=np.array(variance_mathieu),
        norm_uniform=np.array(norm_uniform),
        norm_cosine=np.array(norm_cosine),
        norm_mathieu=np.array(norm_mathieu),
        mathieu_cutoff=np.array(MATHIEU_CUTOFF),
        mathieu_bisection_steps=np.array(BISECTION_STEPS),
        mathieu_edge_probability=np.array(mathieu_edge_probability),
    )

    table = np.column_stack(
        [
            ell,
            np.abs(a_uniform) ** 2,
            np.abs(a_cosine) ** 2,
            np.abs(a_mathieu) ** 2,
        ]
    )
    np.savetxt(
        CSV_FILE,
        table,
        delimiter=",",
        header="ell,p_uniform,p_cosine,p_mathieu",
        comments="",
    )

    print(f"Saved probe-state data to {STATE_FILE}")
    print(f"Saved momentum probabilities to {CSV_FILE}")
    print(f"L = {L}")
    print(f"norm_uniform = {norm_uniform:.16e}")
    print(f"norm_cosine  = {norm_cosine:.16e}")
    print(f"norm_mathieu = {norm_mathieu:.16e}")
    print(f"E_uniform = {energy_uniform:.10f}")
    print(f"E_cosine  = {energy_cosine:.10f}")
    print(f"E_mathieu = {energy_mathieu:.10f}")
    print(f"|E_mathieu - E_cosine| = {energy_match_error:.6e}")
    print(f"lambda_mathieu = {lam_mathieu:.10f}")
    print(f"V_uniform = {variance_uniform:.10e}")
    print(f"V_cosine  = {variance_cosine:.10e}")
    print(f"V_mathieu = {variance_mathieu:.10e}")
    print(f"Mathieu endpoint probability = {mathieu_edge_probability:.6e}")

    return {
        "ell": ell,
        "a_uniform": a_uniform,
        "a_cosine": a_cosine,
        "a_mathieu": a_mathieu,
        "energy_uniform": energy_uniform,
        "energy_cosine": energy_cosine,
        "energy_mathieu": energy_mathieu,
        "lambda_mathieu": lam_mathieu,
    }


def plot_from_saved_data() -> None:
    """Load the saved data and draw the paper-style momentum figure."""
    if not STATE_FILE.exists():
        generate_probe_data()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(STATE_FILE)
    ell = data["ell"]

    curves = [
        ("Uniform", np.abs(data["a_uniform"]) ** 2),
        ("Cosine-window", np.abs(data["a_cosine"]) ** 2),
        ("Mathieu", np.abs(data["a_mathieu"]) ** 2),
    ]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    # Plot only the range needed to show the finite-support probes and the
    # visible Mathieu tails; the full data remain in the CSV/NPZ files.
    plot_mask = np.abs(ell) <= 2 * L
    ell_plot = ell[plot_mask]

    for i, (label, probability) in enumerate(curves):
        y = probability[plot_mask]
        markevery = sparse_staggered_markevery(
            len(ell_plot), i, len(curves), target_markers=6, minimum_step=12
        )
        ax.plot(
            ell_plot,
            y,
            label=label,
            linestyle=LINESTYLES[i],
            marker=MARKERS[i],
            markevery=markevery,
        )

    setup_axis(ax)
    ax.set_xlabel(r"Momentum $\ell$")
    ax.set_ylabel(r"Probability $|a_\ell|^2$")
    ax.set_xlim(-2 * L, 2 * L)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG)
    plt.close(fig)

    print(f"Saved figure to {FIG_PDF}")
    print(f"Saved figure to {FIG_PNG}")


if __name__ == "__main__":
    generate_probe_data()
    plot_from_saved_data()
