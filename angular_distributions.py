from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from momentum_distributions import STATE_FILE, generate_probe_data
from paper_plot_style import (
    LINESTYLES,
    MARKERS,
    apply_paper_style,
    setup_axis,
    sparse_staggered_markevery,
)

# -----------------------------------------------------------------------------
# Parameters and output files
# -----------------------------------------------------------------------------
N_THETA = 4096
PLOT_LIMIT = np.pi / 4

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "plots"
CSV_FILE = DATA_DIR / "qpe_probe_angle.csv"
CHECK_FILE = DATA_DIR / "qpe_probe_angle_checks.npz"
FIG_PDF = FIG_DIR / "qpe_probe_angle.pdf"
FIG_PNG = FIG_DIR / "qpe_probe_angle.png"


def angular_density(
    ell: np.ndarray, amplitudes: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """p_eta(theta|0) = |sum_l a_l exp(i l theta)|^2 / (2 pi)."""
    phase_matrix = np.exp(1j * np.outer(theta, ell))
    wavefunction = phase_matrix @ amplitudes / np.sqrt(2.0 * np.pi)
    return np.abs(wavefunction) ** 2


def generate_angle_data() -> None:
    """Generate and save angle-space probability densities."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Regenerate the shared state file so this script cannot silently use probe
    # amplitudes produced by an older parameter choice or implementation.
    generate_probe_data()

    data = np.load(STATE_FILE)
    ell = data["ell"]
    theta = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)

    p_uniform = angular_density(ell, data["a_uniform"], theta)
    p_cosine = angular_density(ell, data["a_cosine"], theta)
    p_mathieu = angular_density(ell, data["a_mathieu"], theta)

    dtheta = 2.0 * np.pi / N_THETA
    normalizations: dict[str, float] = {}
    for name, p in [
        ("uniform", p_uniform),
        ("cosine", p_cosine),
        ("mathieu", p_mathieu),
    ]:
        normalization = float(np.sum(p) * dtheta)
        normalizations[name] = normalization
        if not np.isclose(normalization, 1.0, atol=2.0e-10):
            raise RuntimeError(f"{name} angular density is not normalized: {normalization}")

    table = np.column_stack([theta, p_uniform, p_cosine, p_mathieu])
    np.savetxt(
        CSV_FILE,
        table,
        delimiter=",",
        header="theta,p_uniform,p_cosine,p_mathieu",
        comments="",
    )
    np.savez_compressed(
        CHECK_FILE,
        n_theta=np.array(N_THETA),
        dtheta=np.array(dtheta),
        norm_uniform=np.array(normalizations["uniform"]),
        norm_cosine=np.array(normalizations["cosine"]),
        norm_mathieu=np.array(normalizations["mathieu"]),
    )
    print(f"Saved angular-density data to {CSV_FILE}")
    print(f"Saved angular normalization checks to {CHECK_FILE}")
    for name, normalization in normalizations.items():
        print(f"angle_norm_{name} = {normalization:.16e}")


def plot_from_saved_data() -> None:
    """Load the saved angle data and draw the paper-style figure."""
    if not CSV_FILE.exists():
        generate_angle_data()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(CSV_FILE, delimiter=",", skiprows=1)
    theta = data[:, 0]

    curves = [
        ("Uniform", data[:, 1]),
        ("Cosine-window", data[:, 2]),
        ("Mathieu", data[:, 3]),
    ]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    plot_mask = np.abs(theta) <= PLOT_LIMIT
    theta_plot = theta[plot_mask]

    for i, (label, probability) in enumerate(curves):
        y = probability[plot_mask]
        markevery = sparse_staggered_markevery(
            len(theta_plot), i, len(curves), target_markers=5, minimum_step=80
        )
        ax.plot(
            theta_plot,
            y,
            label=label,
            linestyle=LINESTYLES[i],
            marker=MARKERS[i],
            markevery=markevery,
        )

    setup_axis(ax)
    ax.set_xlabel(r"Angle $\theta$")
    ax.set_ylabel(r"Probability density $p_\eta(\theta\mid 0)$")
    ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(
        [-np.pi / 4, -np.pi / 8, 0.0, np.pi / 8, np.pi / 4],
        [r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$"],
    )
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG)
    plt.close(fig)

    print(f"Saved figure to {FIG_PDF}")
    print(f"Saved figure to {FIG_PNG}")


if __name__ == "__main__":
    generate_angle_data()
    plot_from_saved_data()
