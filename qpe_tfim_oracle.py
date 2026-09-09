from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from paper_plot_style import apply_paper_style, setup_axis

# -----------------------------------------------------------------------------
# TFIM and rotor-QPE parameters
# -----------------------------------------------------------------------------
N = 5
J = 1.0
H_FIELD = 0.5

# Cosine-window rotor probe supported on ell = -L, ..., L.
L = 16

# First-order Trotterization: total time = N_STEPS * DELTA_T = 1.
DELTA_T = 1.0e-3
N_STEPS = 1000
SAVE_EVERY = 5
TOTAL_TIME = DELTA_T * N_STEPS

# A common time grid is used to verify first-order convergence independently
# of the denser time sampling used in the final heat map.
CONVERGENCE_DELTA_T = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
CONVERGENCE_SAVE_INTERVAL = 2.0e-2

N_THETA = 512

# The N=5 periodic TFIM has nondegenerate eigenvalues at indices 0 and 2.
# They give two well-separated phase trajectories over 0 <= t <= 1.
EIGENSTATE_INDICES = (0, 2)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "plots"
DATA_FILE = DATA_DIR / "qpe_tfim_oracle.npz"
SUMMARY_FILE = DATA_DIR / "qpe_tfim_summary.csv"
CONVERGENCE_FILE = DATA_DIR / "qpe_tfim_convergence.csv"
FIG_PDF = FIG_DIR / "qpe_tfim_oracle.pdf"
FIG_PNG = FIG_DIR / "qpe_tfim_oracle.png"


# -----------------------------------------------------------------------------
# Spin operators and TFIM Hamiltonian
# -----------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def tensor_product(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def operator_on_sites(site_operators: dict[int, np.ndarray]) -> np.ndarray:
    operators = [I2 for _ in range(N)]
    for site, operator in site_operators.items():
        operators[site] = operator
    return tensor_product(operators)


def build_tfim() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H = H_Z + H_X for the periodic N-site TFIM."""
    dimension = 2**N
    H_Z = np.zeros((dimension, dimension), dtype=complex)
    H_X = np.zeros((dimension, dimension), dtype=complex)

    for j in range(N):
        H_Z += -J * operator_on_sites({j: Z, (j + 1) % N: Z})
        H_X += -H_FIELD * operator_on_sites({j: X})

    return H_Z + H_X, H_Z, H_X


# -----------------------------------------------------------------------------
# Rotor probe and Trotterized momentum-sector dynamics
# -----------------------------------------------------------------------------
def cosine_probe(ell: np.ndarray) -> np.ndarray:
    amplitudes = np.cos(np.pi * ell / (2.0 * (L + 1))) / np.sqrt(L + 1)
    amplitudes = amplitudes.astype(complex)
    amplitudes /= np.linalg.norm(amplitudes)
    return amplitudes


def trotter_steps(
    H_Z: np.ndarray,
    H_X: np.ndarray,
    ell: np.ndarray,
    delta_t: float = DELTA_T,
) -> list[np.ndarray]:
    """One first-order Trotter step in each integer momentum sector.

    The rotor-QPE oracle is exp(+i t H tensor ell_hat), hence sector ell uses
    exp(+i ell dt H_Z) exp(+i ell dt H_X).
    """
    return [
        expm(1j * m * delta_t * H_Z) @ expm(1j * m * delta_t * H_X)
        for m in ell
    ]


def propagate_sectors(
    initial_spin_state: np.ndarray,
    sector_steps: list[np.ndarray],
    delta_t: float = DELTA_T,
    n_steps: int = N_STEPS,
    save_every: int = SAVE_EVERY,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the spin state independently in every rotor momentum sector."""
    n_sector = len(sector_steps)
    states = np.tile(initial_spin_state, (n_sector, 1)).astype(complex)

    saved_states: list[np.ndarray] = []
    saved_times: list[float] = []

    for step_number in range(n_steps + 1):
        if step_number % save_every == 0:
            saved_states.append(states.copy())
            saved_times.append(step_number * delta_t)

        if step_number == n_steps:
            break

        for sector, U_step in enumerate(sector_steps):
            states[sector] = U_step @ states[sector]

    return np.asarray(saved_times), np.asarray(saved_states)


def angle_density_from_sector_states(
    sector_states: np.ndarray,
    amplitudes: np.ndarray,
    ell: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Trace out the spins and evaluate the rotor angle POVM density.

    rho_R[ell,ell'] = a_ell a_ell'^* <psi_ell'|psi_ell>.
    Then p(theta) = (2pi)^(-1) sum rho_R exp[i(ell-ell')theta].
    """
    fourier = np.exp(1j * np.outer(theta, ell))
    densities = np.empty((len(sector_states), len(theta)), dtype=float)

    amplitude_outer = np.outer(amplitudes, np.conj(amplitudes))

    for time_index, states in enumerate(sector_states):
        gram = states @ states.conj().T
        rho_rotor = amplitude_outer * gram
        p_theta = np.einsum(
            "tl,lm,tm->t", fourier, rho_rotor, fourier.conj(), optimize=True
        ).real
        densities[time_index] = p_theta / (2.0 * np.pi)

    return densities


def initial_probe_density(
    theta: np.ndarray, amplitudes: np.ndarray, ell: np.ndarray
) -> np.ndarray:
    phase = np.exp(1j * np.outer(theta, ell))
    wavefunction = phase @ amplitudes / np.sqrt(2.0 * np.pi)
    return np.abs(wavefunction) ** 2


def shifted_probe_density(
    theta: np.ndarray,
    shifts: np.ndarray,
    amplitudes: np.ndarray,
    ell: np.ndarray,
) -> np.ndarray:
    """Evaluate p_eta(theta + shift | 0) for one shift per saved time."""
    result = np.empty((len(shifts), len(theta)), dtype=float)
    for i, shift in enumerate(shifts):
        result[i] = initial_probe_density(theta + shift, amplitudes, ell)
    return result


def exact_two_eigenstate_density(
    theta: np.ndarray,
    times: np.ndarray,
    energies: tuple[float, float],
    amplitudes: np.ndarray,
    ell: np.ndarray,
) -> np.ndarray:
    """Exact rotor density for an equal superposition of two eigenstates."""
    energy_a, energy_b = energies
    return 0.5 * shifted_probe_density(
        theta, energy_a * times, amplitudes, ell
    ) + 0.5 * shifted_probe_density(theta, energy_b * times, amplitudes, ell)


def common_grid_convergence_row(
    delta_t: float,
    H_Z: np.ndarray,
    H_X: np.ndarray,
    initial_spin_state: np.ndarray,
    ell: np.ndarray,
    amplitudes: np.ndarray,
    theta: np.ndarray,
    energies: tuple[float, float],
) -> tuple[float, int, float, float, float]:
    """Return TV and normalization checks on a common time grid."""
    n_steps = int(round(TOTAL_TIME / delta_t))
    if not np.isclose(n_steps * delta_t, TOTAL_TIME, atol=1.0e-14, rtol=0.0):
        raise RuntimeError(f"Delta t={delta_t} does not divide the total time.")

    save_every = int(round(CONVERGENCE_SAVE_INTERVAL / delta_t))
    if save_every < 1 or not np.isclose(
        save_every * delta_t,
        CONVERGENCE_SAVE_INTERVAL,
        atol=1.0e-14,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Delta t={delta_t} does not resolve the common convergence grid."
        )

    sector_steps = trotter_steps(H_Z, H_X, ell, delta_t=delta_t)
    times, sector_states = propagate_sectors(
        initial_spin_state,
        sector_steps,
        delta_t=delta_t,
        n_steps=n_steps,
        save_every=save_every,
    )
    p_trotter = angle_density_from_sector_states(
        sector_states, amplitudes, ell, theta
    )
    p_exact = exact_two_eigenstate_density(
        theta, times, energies, amplitudes, ell
    )

    dtheta = 2.0 * np.pi / len(theta)
    normalization_error = float(
        np.max(np.abs(np.sum(p_trotter, axis=1) * dtheta - 1.0))
    )
    tv_distance = 0.5 * np.sum(np.abs(p_trotter - p_exact), axis=1) * dtheta
    return (
        delta_t,
        n_steps,
        float(np.max(tv_distance)),
        float(tv_distance[-1]),
        normalization_error,
    )


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def generate_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    H, H_Z, H_X = build_tfim()
    if N != 5:
        raise RuntimeError(f"The requested benchmark has N=5, not N={N}.")
    hamiltonian_hermiticity_error = float(np.max(np.abs(H - H.conj().T)))
    if hamiltonian_hermiticity_error > 1.0e-13:
        raise RuntimeError("The TFIM Hamiltonian is not Hermitian.")

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    ia, ib = EIGENSTATE_INDICES
    E_a = float(eigenvalues[ia])
    E_b = float(eigenvalues[ib])

    psi_a = eigenvectors[:, ia]
    psi_b = eigenvectors[:, ib]
    norm_psi_a = float(np.vdot(psi_a, psi_a).real)
    norm_psi_b = float(np.vdot(psi_b, psi_b).real)
    overlap_ab = float(abs(np.vdot(psi_a, psi_b)))
    gap_a = float(np.min(np.abs(np.delete(eigenvalues, ia) - E_a)))
    gap_b = float(np.min(np.abs(np.delete(eigenvalues, ib) - E_b)))
    if not np.isclose(norm_psi_a, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Selected eigenstate a is not normalized.")
    if not np.isclose(norm_psi_b, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Selected eigenstate b is not normalized.")
    if overlap_ab > 2.0e-14:
        raise RuntimeError("Selected eigenstates are not orthogonal.")
    if min(gap_a, gap_b) < 1.0e-10:
        raise RuntimeError("At least one selected energy is numerically degenerate.")

    initial_spin_state = (psi_a + psi_b) / np.sqrt(2.0)
    initial_spin_state /= np.linalg.norm(initial_spin_state)
    initial_spin_norm = float(np.vdot(initial_spin_state, initial_spin_state).real)

    ell = np.arange(-L, L + 1, dtype=int)
    integer_momentum_support = bool(
        ell[0] == -L
        and ell[-1] == L
        and np.array_equal(np.diff(ell), np.ones(len(ell) - 1, dtype=int))
    )
    if not integer_momentum_support:
        raise RuntimeError("Rotor momentum labels are not the integers -L,...,L.")

    amplitudes = cosine_probe(ell)
    probe_norm = float(np.sum(np.abs(amplitudes) ** 2))
    if not np.isclose(probe_norm, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("The cosine-window rotor probe is not normalized.")

    sector_steps = trotter_steps(H_Z, H_X, ell)

    times, sector_states = propagate_sectors(initial_spin_state, sector_steps)

    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    p_trotter = angle_density_from_sector_states(
        sector_states, amplitudes, ell, theta
    )

    # U(t) = exp(-itH) has eigenphase phi_a(t) = -E_a t mod 2pi.
    trajectory_a = np.mod(-E_a * times, 2.0 * np.pi)
    trajectory_b = np.mod(-E_b * times, 2.0 * np.pi)

    # For an equal superposition of two orthogonal energy eigenstates, tracing
    # out the target gives an equal mixture of the two translated probe profiles.
    p_exact = exact_two_eigenstate_density(
        theta, times, (E_a, E_b), amplitudes, ell
    )

    dtheta = 2.0 * np.pi / N_THETA
    norm_trotter = np.sum(p_trotter, axis=1) * dtheta
    norm_exact = np.sum(p_exact, axis=1) * dtheta
    if not np.allclose(norm_trotter, 1.0, atol=2.0e-10):
        raise RuntimeError("Trotterized angle density is not normalized.")
    if not np.allclose(norm_exact, 1.0, atol=2.0e-10):
        raise RuntimeError("Exact angle density is not normalized.")

    tv_distance = 0.5 * np.sum(np.abs(p_trotter - p_exact), axis=1) * dtheta
    max_trotter_normalization_error = float(np.max(np.abs(norm_trotter - 1.0)))
    max_exact_normalization_error = float(np.max(np.abs(norm_exact - 1.0)))
    max_sector_state_norm_error = float(
        np.max(np.abs(np.sum(np.abs(sector_states) ** 2, axis=2) - 1.0))
    )

    convergence_rows = np.asarray(
        [
            common_grid_convergence_row(
                delta_t,
                H_Z,
                H_X,
                initial_spin_state,
                ell,
                amplitudes,
                theta,
                (E_a, E_b),
            )
            for delta_t in CONVERGENCE_DELTA_T
        ],
        dtype=float,
    )
    if not np.all(np.diff(convergence_rows[:, 2]) < 0.0):
        raise RuntimeError("The Trotter error did not decrease with Delta t.")

    np.savez_compressed(
        DATA_FILE,
        times=times,
        theta=theta,
        p_trotter=p_trotter,
        p_exact=p_exact,
        tv_distance=tv_distance,
        trajectory_a=trajectory_a,
        trajectory_b=trajectory_b,
        selected_energies=np.array([E_a, E_b]),
        selected_indices=np.array(EIGENSTATE_INDICES),
        ell=ell,
        probe_amplitudes=amplitudes,
        N=np.array(N),
        J=np.array(J),
        h=np.array(H_FIELD),
        L=np.array(L),
        delta_t=np.array(DELTA_T),
        n_steps=np.array(N_STEPS),
        total_time=np.array(TOTAL_TIME),
        norm_eigenstate_a=np.array(norm_psi_a),
        norm_eigenstate_b=np.array(norm_psi_b),
        eigenstate_overlap=np.array(overlap_ab),
        selected_gap_a=np.array(gap_a),
        selected_gap_b=np.array(gap_b),
        norm_initial_spin_state=np.array(initial_spin_norm),
        norm_probe=np.array(probe_norm),
        max_trotter_normalization_error=np.array(
            max_trotter_normalization_error
        ),
        max_exact_normalization_error=np.array(max_exact_normalization_error),
        max_sector_state_norm_error=np.array(max_sector_state_norm_error),
        hamiltonian_hermiticity_error=np.array(hamiltonian_hermiticity_error),
        integer_momentum_support=np.array(integer_momentum_support),
        physical_rotor_is_cyclic=np.array(False),
        convergence_delta_t=convergence_rows[:, 0],
        convergence_max_tv=convergence_rows[:, 2],
    )

    summary = np.column_stack(
        [times, trajectory_a, trajectory_b, tv_distance, norm_trotter, norm_exact]
    )
    np.savetxt(
        SUMMARY_FILE,
        summary,
        delimiter=",",
        header=(
            "t,theta_a_exact,theta_b_exact,total_variation_distance,"
            "trotter_normalization,exact_normalization"
        ),
        comments="",
    )
    np.savetxt(
        CONVERGENCE_FILE,
        convergence_rows,
        delimiter=",",
        header=(
            "delta_t,n_steps,max_total_variation_distance,"
            "final_total_variation_distance,max_normalization_error"
        ),
        comments="",
    )

    print(f"Saved simulation data to {DATA_FILE}")
    print(f"Saved summary data to {SUMMARY_FILE}")
    print(f"Saved convergence data to {CONVERGENCE_FILE}")
    print(f"Selected energies: E_a = {E_a:.10f}, E_b = {E_b:.10f}")
    print(f"Selected indices: {ia}, {ib}")
    print(f"Selected spectral gaps: {gap_a:.6e}, {gap_b:.6e}")
    print(f"Initial-state norm = {initial_spin_norm:.16e}")
    print(f"Probe norm = {probe_norm:.16e}")
    print(
        "max_t normalization error = "
        f"{max_trotter_normalization_error:.6e}"
    )
    print(f"max_t D_TV = {np.max(tv_distance):.6e}")
    print(f"D_TV(t_final) = {tv_distance[-1]:.6e}")
    for row in convergence_rows:
        print(
            "convergence: "
            f"Delta t={row[0]:.4e}, steps={int(row[1])}, "
            f"max D_TV={row[2]:.6e}, final D_TV={row[3]:.6e}"
        )


# -----------------------------------------------------------------------------
# Plotting from saved data
# -----------------------------------------------------------------------------
def plot_from_saved_data() -> None:
    if not DATA_FILE.exists():
        generate_data()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_FILE)

    times = data["times"]
    theta = data["theta"]
    p_trotter = data["p_trotter"]
    trajectory_a = data["trajectory_a"]
    trajectory_b = data["trajectory_b"]
    E_a, E_b = data["selected_energies"]

    def break_periodic_wraps(trajectory: np.ndarray) -> np.ndarray:
        plotted = trajectory.copy()
        jump_indices = np.flatnonzero(np.abs(np.diff(trajectory)) > np.pi) + 1
        plotted[jump_indices] = np.nan
        return plotted

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    mesh = ax.pcolormesh(
        times,
        theta,
        p_trotter.T,
        shading="auto",
        rasterized=True,
    )

    ax.plot(
        times,
        break_periodic_wraps(trajectory_a),
        linestyle="--",
        linewidth=1.5,
        label=rf"$-E_a t$, $E_a={E_a:.3f}$",
    )
    ax.plot(
        times,
        break_periodic_wraps(trajectory_b),
        linestyle="-.",
        linewidth=1.5,
        label=rf"$-E_b t$, $E_b={E_b:.3f}$",
    )

    setup_axis(ax)
    ax.grid(False)
    ax.set_xlabel(r"Evolution time $t$")
    ax.set_ylabel(r"Rotor angle $\theta$")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_yticks(
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    ax.legend(loc="upper left", frameon=True)

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(r"$p(\theta,t)$")

    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG)
    plt.close(fig)

    print(f"Saved figure to {FIG_PDF}")
    print(f"Saved figure to {FIG_PNG}")


if __name__ == "__main__":
    generate_data()
    plot_from_saved_data()
