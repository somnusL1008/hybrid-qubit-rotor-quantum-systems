# Hybrid Qubit–Rotor Quantum Systems: Numerical Reproducibility

This repository contains the Python scripts used for the numerical results in the paper

**Hybrid Qubit–Rotor Quantum Systems: Clifford Structure, Universal Control, and Applications**.

The scripts cover three numerical components of the manuscript:

1. the two-plaquette compact \(U(1)\) gauge–matter calculations in Sec. 5;
2. the rotor probe-state calculations in Sec. 6.2;
3. the transverse-field Ising model (TFIM) rotor-QPE benchmark in Sec. 6.3.

All numerical data used by these scripts are generated from the model parameters defined in the source files; no external experimental data are required.

---

## Repository contents

Use the following filenames in the public repository:

```text
README.md
gauge_matter_two_plaquette.py
momentum_distributions.py
angular_distributions.py
qpe_tfim_oracle.py
paper_plot_style.py
```

The current local file `circuit.py` corresponds to `qpe_tfim_oracle.py`.

`paper_plot_style.py` is required because all four numerical scripts import plotting utilities from it. The probe-angle script also imports `momentum_distributions.py`, so the public filenames should not contain suffixes such as `(1)` or `(2)`.

A `requirements.txt` containing at least

```text
numpy
scipy
matplotlib
```

is recommended, although the scripts themselves do not require pinned package versions.

---

## Python environment

The code requires Python 3 and the following packages:

- NumPy
- SciPy
- Matplotlib

A minimal installation is

```bash
python -m pip install numpy scipy matplotlib
```

If your system uses `python3` instead of `python`, replace `python` by `python3` in the commands below.

---

# 1. Sec. 5: two-plaquette compact \(U(1)\) gauge–matter model

Run

```bash
python gauge_matter_two_plaquette.py --outdir plots --datadir data
```

For a faster diagnostic run, use

```bash
python gauge_matter_two_plaquette.py --fast --outdir plots --datadir data
```

The production calculation uses the default parameters encoded in the script, including

- a \(2\times1\) open two-plaquette lattice;
- electric-flux cutoffs \(L=1,2,3,4,6\) for the static scan;
- electric-flux cutoffs \(L=1,2,3,4\) for the real-time calculation;
- \(g^{-2}\in[10^{-2},10]\) with 55 static-scan points;
- \(g^{-2}=1\) for the dynamics;
- \(0\le t\le8\) with 241 time points;
- \(m_0=1.5\);
- \(\kappa=1\).

The code constructs the finite-flux Gauss-law basis directly, builds the sparse Hamiltonian, performs the static and real-time calculations, and checks Gauss's law, Hermiticity, eigenpair residuals, norm conservation, and matter-number conservation.

### Generated data

```text
data/gauge_matter_two_plaquette_static.csv
data/gauge_matter_two_plaquette_dynamics.csv
data/gauge_matter_two_plaquette_metadata.json
```

The metadata file records the numerical parameters, basis dimensions, validation errors, Python/NumPy/SciPy versions used for that run, and the SHA-256 hash of the script.

### Generated figures

```text
plots/gauge_matter_two_plaquette_static.pdf
plots/gauge_matter_two_plaquette_static.png
plots/gauge_matter_two_plaquette_dynamics.pdf
plots/gauge_matter_two_plaquette_dynamics.png
```

These calculations support the static finite-flux and real-time two-plaquette results reported in Sec. 5.

---

# 2. Sec. 6.2: rotor probe states

The probe-state calculation is split into momentum-space and angle-space scripts.

## 2.1 Momentum distributions

Run

```bash
python momentum_distributions.py
```

The script constructs three rotor probes:

- a uniform finite-support probe;
- a cosine-window finite-support probe;
- an energy-matched Mathieu probe.

The uniform and cosine-window probes have support \(|\ell|\le L\) with \(L=32\). The Mathieu calculation uses a larger numerical cutoff and matches the cosine-window probe's mean kinetic energy.

The script checks normalization and the energy matching between the cosine-window and Mathieu probes.

### Generated data

```text
data/qpe_probe_states.npz
data/qpe_probe_momentum.csv
```

### Generated figures

```text
plots/qpe_probe_momentum.pdf
plots/qpe_probe_momentum.png
```

---

## 2.2 Angular distributions

Run

```bash
python angular_distributions.py
```

This script regenerates the probe-state data by calling `generate_probe_data()` from `momentum_distributions.py`, computes the corresponding angle-space probability densities, checks their normalization, and produces the angular-distribution panel used in the paper.

### Generated data

```text
data/qpe_probe_angle.csv
data/qpe_probe_angle_checks.npz
```

### Generated figures

```text
plots/qpe_probe_angle.pdf
plots/qpe_probe_angle.png
```

The momentum- and angle-space outputs together reproduce the numerical probe comparison in Sec. 6.2.

---

# 3. Sec. 6.3: TFIM rotor-QPE benchmark

Run

```bash
python qpe_tfim_oracle.py
```

The benchmark uses a periodic five-site transverse-field Ising model with

- \(N=5\);
- \(J=1\);
- transverse field \(h=0.5\);
- cosine-window rotor support \(L=16\);
- first-order Trotter step \(\Delta t=10^{-3}\);
- 1000 Trotter steps, giving total time \(t=1\);
- 512 angle-grid points.

The script constructs the TFIM Hamiltonian, selects two nondegenerate eigenstates, propagates the spin state independently in each rotor momentum sector, forms the reduced rotor angle distribution, and compares the Trotterized result with the analytical eigenphase trajectories.

It also performs a convergence check over

```text
Delta t = 4e-3, 2e-3, 1e-3, 5e-4
```

and records the total-variation distance between the Trotterized and exact rotor distributions.

### Generated data

```text
data/qpe_tfim_oracle.npz
data/qpe_tfim_summary.csv
data/qpe_tfim_convergence.csv
```

### Generated figures

```text
plots/qpe_tfim_oracle.pdf
plots/qpe_tfim_oracle.png
```

These outputs support the numerical TFIM rotor-QPE benchmark in Sec. 6.3.

---

# Reproducing all numerical results

Starting from the repository root, run

```bash
python momentum_distributions.py
python angular_distributions.py
python qpe_tfim_oracle.py
python gauge_matter_two_plaquette.py --outdir plots --datadir data
```

The scripts create `data/` and `plots/` directories automatically when needed.

For an initial quick check of the gauge calculation, replace the final command by

```bash
python gauge_matter_two_plaquette.py --fast --outdir plots --datadir data
```

The `--fast` option is intended only as a diagnostic run and does not reproduce the full production parameter grid.

---

# Numerical validation

The scripts contain explicit internal validation checks rather than relying only on visual agreement of the plots.

The gauge–matter calculation checks, among other conditions,

- the lattice incidence/plaquette consistency relation;
- exact satisfaction of Gauss's law by the constructed physical basis;
- Hermiticity of the Hamiltonian and plaquette observables;
- eigenpair residuals;
- real-time norm conservation;
- total matter-number conservation.

The probe-state scripts check

- normalization of all momentum-space probes;
- matching of the Mathieu and cosine-window kinetic energies;
- normalization of the angle-space densities.

The TFIM benchmark checks

- Hamiltonian Hermiticity;
- normalization and orthogonality of the selected eigenstates;
- normalization of the rotor probe;
- normalization of the Trotterized and exact angle distributions;
- monotonic reduction of the Trotter error under step-size refinement.

---

# Notes on generated data

The repository can be distributed either with or without the generated `data/` and `plots/` directories because the scripts regenerate these files from the encoded model parameters.

For a permanent archival release, it is useful to include the small CSV/NPZ/JSON files from the final author-validated run in addition to the source code. This allows readers to compare their regenerated outputs directly with the archived numerical results.

The generated PDF/PNG figures are optional for reproducibility because they can be recreated from the scripts.

---

# Code provenance

The manuscript contains the corresponding AI-use disclosures for the numerical code. The scientific models, parameters, observables, validation criteria, and interpretation of the numerical results are the responsibility of the authors.

---

# Citation

If you use this code, please cite the accompanying paper:

**D. Luo, A. Kushwaha, M. Tirfe, and B. N. Bakalov,  
“Hybrid Qubit–Rotor Quantum Systems: Clifford Structure, Universal Control, and Applications.”**

A persistent archive identifier can be added here after the public release is deposited in Zenodo or another archival repository.
