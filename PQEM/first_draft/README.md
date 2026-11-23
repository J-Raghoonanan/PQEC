# PQEC — Purification-based Quantum Error Correction

Density-matrix simulation of the rho2 purification protocol,
with exact and approximate Clifford twirling for dephasing noise channels.

---

## Repository structure

```
project/
├── src/
│   └── simulation/
│       ├── rho2_sims/                   # rho2 simulation — exact full Clifford twirl
│       │   ├── configs.py               # RunSpec, TargetSpec, NoiseSpec, TwirlingSpec dataclasses
│       │   ├── main_grid_run.py         # CLI entry point for grid sweeps
│       │   ├── streaming_runner.py      # Iterative rho2 runner; produces steps/finals DataFrames
│       │   ├── rho2_purification.py     # Core rho2 operation: ρ → ρ²/Tr(ρ²)
│       │   ├── noise_engine.py          # Noise channels + Clifford-twirl averaging
│       │   └── state_factory.py         # Target state preparation (Haar, GHZ, Hadamard, …)
│       │
│       └── rho2_approx_twirl_sim/       # rho2 simulation — approximate (subset) Clifford twirl
│           ├── configs.py               # Extends rho2_sims configs with subset_fraction, subset_mode
│           ├── main_grid_run.py         # CLI entry point; adds --subset-fraction, --subset-mode flags
│           ├── streaming_runner.py      # Identical protocol; delegates twirl to noise_engine
│           ├── rho2_purification.py     # Same rho2 core
│           ├── noise_engine.py          # Subset selection + averaging loop
│           └── state_factory.py         # Same state preparation utilities
│
├── figures/
│   ├── rho2_plots.py                    # Reads data/rho2_sim CSVs; produces figures/rho2_results
│   ├── rho2_results/                    # Output PDFs from rho2_plots.py
│   └── theory_ana_plots/
│   └── theory_plotter.py                # Analytic/closed-form theory figures (no CSV dependency)
│
├── data/
│   └── rho2_sim/                        # CSV output from both simulation packages
│       ├── steps_rho2_<noise>.csv         # Per-iteration metrics (fidelity, purity, ε_L, …)
│       └── finals_rho2_<noise>.csv        # Per-run summary metrics
│
├── requirements.txt
└── README.md
```

---

## Simulation packages

Both packages share the same protocol and module structure; they differ only in how
Clifford twirling is handled.

### `rho2_sims` — exact twirl

Computes the exact average over all 3^M local Clifford combinations {I, H, HS}^⊗M:

```
E_twirled(ρ) = (1/3^M) Σ_{C ∈ {I,H,HS}^⊗M}  C†  E(C ρ C†)  C
```

For dephasing noise this converts the Z-dephasing channel into an effective
depolarizing channel exactly.

### `rho2_approx_twirl_sim` — approximate (subset) twirl

Approximates the same average using a subset of K < 3^M combinations:

```
E_twirled(ρ) ≈ (1/K) Σ_{k=1}^{K}  C_k†  E(C_k ρ C_k†)  C_k
```

K is controlled by `--subset-fraction` (0 < f ≤ 1.0, where 1.0 recovers the exact twirl).
The subset is drawn once per run with a fixed seed and applied consistently across
all iterations, making the approximate twirl a deterministic, reproducible channel.

---

## Protocol

Both packages implement the same iterative rho2 protocol. Per cycle:

1. Apply noise once to the current state (with or without Clifford twirling).
2. Clone 2^ℓ identical copies of the noisy state.
3. Run ℓ clean rho2 rounds via a binary merge tree:
   `(ρ_L, ρ_R) → ρ_L² / Tr(ρ_L²)`
4. The purified output becomes the input for the next cycle.

Number of cycles: `floor(log₂(N))`, consistent with the SWAP-test convention.

---

## Running a simulation

```bash
# Exact twirl, Z-dephasing, M=1..5, purification level ℓ=1
python -m src.simulation.rho2_sims.main_grid_run \
    --out data/rho2_sim \
    --noise z \
    --m-values 1 2 3 4 5 \
    --iterative \

# Approximate twirl, 30% subset, random sampling
python -m src.simulation.rho2_approx_twirl_sim.main_grid_run \
    --out data/rho2_sim \
    --noise z \
    --m-values 1 2 3 4 5 \
    --iterative \
    --subset-fraction 0.2 \
    --subset-mode random \
    --subset-seed 42
```

Key CLI flags (shared by both packages):

| Flag                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `--noise`           | `all`, `depol`, `z`, `x`                                   |
| `--m-values`        | Qubit counts to simulate (cap at 6 — memory scales as 4^M) |
| `--iterative`       | Apply fresh noise each cycle (always recommended)          |
| `--no-twirl`        | Disable Clifford twirling even for dephasing               |
| `--subset-fraction` | _(approx only)_ Fraction of 3^M combinations to use        |
| `--subset-mode`     | _(approx only)_ `random` or `first_k`                      |
| `--subset-seed`     | _(approx only)_ RNG seed for subset selection              |

---

## Output format

Both packages append rows to two CSVs per noise type under `--out`:

**`steps_rho2_<noise>.csv`** — one row per iteration per run:

| Column                  | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `run_id`                | Unique identifier encoding all run parameters           |
| `iteration`             | Cycle index (1-based)                                   |
| `M`                     | Number of qubits                                        |
| `purification_level`    | ℓ                                                       |
| `p`                     | Physical error rate                                     |
| `fidelity`              | F = ⟨ψ\|ρ\|ψ⟩ after rho2 purification                   |
| `eps_L`                 | Trace distance ½‖ρ − \|ψ⟩⟨ψ\|‖₁ after rho2 purification |
| `purity`                | Tr(ρ²) after rho2 purification                          |
| `fidelity_before_noise` | F before noise is applied this cycle                    |
| `fidelity_after_noise`  | F immediately after noise, before rho2 purification     |
| `subset_fraction`       | Subset fraction used (1.0 for exact twirl)              |

**`finals_rho2_<noise>.csv`** — one row per run (summary):

| Column                       | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| `fidelity_init`              | Baseline F after one noise application from perfect state |
| `fidelity_final`             | F after all cycles                                        |
| `eps_L_init` / `eps_L_final` | Trace distance before/after all cycles                    |
| `error_reduction_ratio`      | ε_L_final / ε_L_init                                      |
| `iterations`                 | Total cycles run                                          |
| `twirling_enabled`           | Whether twirling was active for this noise type           |

---

## Generating figures

```bash
python figures/rho2_plots.py --data data/rho2_sim --out figures/rho2_results

# Analytic theory figures (no simulation data required)
python figures/theory_ana_plots/theory_plotter.py --figures-dir figures/theory_ana_plots
```

---

## Noise types

| Key            | Channel               | Twirling effect                                 |
| -------------- | --------------------- | ----------------------------------------------- |
| `depolarizing` | E(ρ) = (1−p)ρ + p I/D | No twirling applied (channel already isotropic) |
| `dephase_z`    | E(ρ) = (1−p)ρ + p ZρZ | Twirling converts to effective depolarizing     |
| `dephase_x`    | E(ρ) = (1−p)ρ + p XρX | Twirling converts to effective depolarizing     |

---

## Supported target states

`hadamard` · `ghz` · `haar` · `random_circuit` · `single_qubit_product` · `manual`

---

## Dependencies

See `requirements.txt`. Core dependencies:

- `numpy`, `pandas` — numerics and data handling
- `qiskit` — `DensityMatrix`, `Statevector`, `Kraus`, `partial_trace`, state preparation
- `matplotlib`, `seaborn` — figure generation
