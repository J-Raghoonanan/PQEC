# PQES: Noisy-implementation numerical suite

This is a **new, self-contained numerical project** for the revised manuscript *Hybrid Quantum Error Correction and Mitigation by Purification*. Its purpose is to compute the architecture-specific evidence missing from the original paper, rather than regenerate the old ideal-density-power plots.

The original uploaded files were read as references for organization, state preparation, noise notation, CSV conventions and plotting style. **None of them is imported or modified by this project.** You can place `pqes_new_numerics/` next to `main_grid_run.py`, `rho2_plots.py`, etc.

## Quick start

Requires Python 3.10+ and NumPy, pandas, Matplotlib and pytest. **Qiskit is not required.**

From the directory containing your existing Python files:

```bash
python -m pip install -r pqes_new_numerics/requirements.txt
python pqes_new_numerics/run_simulations.py --profile quick
python pqes_new_numerics/plot_results.py
python -m pytest -q pqes_new_numerics/tests/
```

For a more useful manuscript-level sweep (the supplied sample data were generated with this profile):

```bash
python pqes_new_numerics/run_simulations.py --profile standard
python pqes_new_numerics/plot_results.py
```

The `full` profile is a denser sensitivity sweep:

```bash
python pqes_new_numerics/run_simulations.py --profile full --seed 31 --rmse 0.12 --width-cap 7
python pqes_new_numerics/plot_results.py
```

Run just one investigation, e.g.:

```bash
python pqes_new_numerics/run_simulations.py --task placement --profile standard
python pqes_new_numerics/plot_results.py --figure placement
```

All simulation CSVs, irrespective of the user's current working directory, are written by default to **`data/pqes_new_numerics/`** under the same old-code parent directory. Plotting **only reads the saved CSVs** and creates PDF/PNG files in **`figures/pqes_new_numerics/`**. The simulator never reads old data or imports old code. The plotting script never runs simulations.

## Files and the paper's logic

| File | Purpose |
|---|---|
| `run_simulations.py` | **Main executable scientific narrative**: detailed assumptions, equations, all grid schedules and saved observables |
| `core.py` | Native controlled-Fredkin circuit instruments, noisy control/data channels, recursive tree and coherent VD, physical statistics |
| `circuits.py` | Splittable entangling algorithm, noisy algorithmic gates, ideal Pauli-channel weighting reference |
| `vcp_physical.py` | **Physical order-two layerwise VCP** ancilla/swap/channel/swap/readout instrument and honest state-VD comparator resource accounting |
| `plot_results.py` | CSV-based publication figures; user-provided Times New Roman / seaborn paper style with a font fallback |
| `tests/test_validation.py` | 43 deterministic checks including entangled states, parity consistency and noise-channel positivity |
| `FIGURE_CAPTIONS.md` | Ready-to-adapt captions with precise simulation settings and comparison limitations |
| `data/pqes_new_numerics/*.csv` | Numerical data saved by the actual runs, included in the download archive |
| `figures/pqes_new_numerics/*.pdf` | Vector figures generated only from CSVs, included in the download archive |

The long opening docstring of `run_simulations.py` is an executable Methods section. I recommend reading it before changing the parameter grids, followed by the `core.py` docstring and then `circuits.py`.

## Physical observable rather than a fictitious deterministic purification gate

Each SWAP instrument physically measures a control ancilla after a controlled exchange of two registers, discards one register and retains the other. An experiment yields a physical branch state with a parity record. For a tree of independent copies, combine all records with the same total parity and denote the **unnormalized** surviving-register states by `tau_plus` and `tau_minus`.

The ordinary and signed operators are

\[
R=\tau_++\tau_-, \quad W=\tau_+-\tau_-.
\]

The ideal identities are

\[
R=\rho,\qquad W=\rho^N.
\]

The physically estimated observable is a normalized *ratio of signed moments*,

\[
\mu=\frac{\operatorname{Tr}(OW)}{\operatorname{Tr} W}.
\]

The code **never** substitutes the normalized mathematical matrix power as the physical output of the SWAP circuit. `spectral_power` exists solely to evaluate the ideal mathematical reference. Non-postselection does not imply zero sampling cost.

The generic noisy apparatus is completely characterized, for its final binary record, by `R_A`, `W_A`. We save

\[
 t_A=\mathrm{Tr}W_A,\quad \mu_A=\mathrm{Tr}(O W_A)/t_A,\quad
 b_A=\mu_A-\mu_{\mathrm{ideal}}.
\]

The large-shot ratio-estimator variance is

\[
\mathrm{Var}(\widehat\mu_A)=\frac{v_A}{S}+o(S^{-1}),\qquad
v_A=\frac{\mathrm{Tr}(O^2R_A)-2\mu_A\mathrm{Tr}(OR_A)+\mu_A^2}{t_A^2}.
\]

For finite-shot estimates, the denominator can be zero. The separate `shots.csv` dataset draws actual parity and observable outcomes and saves undefined ratios as `NaN`, rather than concealing such failures. The `phase.csv` shot estimates use a **leading asymptotic cost surrogate**, not a rigorous finite-shot confidence bound.

## Four implementable architectures

- **Parallel measured SWAP tree:** all register pairs needed for one balanced level are simultaneously available; the ancilla is measured at every internal node. Peak coherent qubits (including concurrent ancillas) = `MN + N/2` for N≥2.
- **Recycled balanced tree:** depth-first binary tree reusing registers; its earlier subtree is exposed to memory noise during preparation of a later subtree. Peak count = `M(log2(N)+1)+1` (for N a power of two).
- **Measured sequential accumulator:** one retained register and one fresh copy are compared repeatedly. One control ancilla is measured/reset after *every* comparison; peak `2M+1`.
- **Coherent-control sequential VD:** keep one ancilla quantum-coherent through all N−1 controlled register swaps and measure it once at the end; discard/recycle spent data registers. Also `2M+1` peak. It is a **specific REQUEST-like implementation** and a matched nondestructive coherent-control baseline, **not a claim to simulate every published variant of REQUEST or cyclic VD**.

All perform `(N-1)M` **native three-qubit Fredkins** and use `N` independently prepared copies. Each native Fredkin is assigned a three-qubit depolarizing channel after the gate; each control experiences a configurable Z dephasing channel; each ancilla readout has an independent classical assignment error. `tree_recycled` and `tree_sequential` include specified storage memory-time errors. The coherent-control arm includes optional additional control idle dephasing **between** register swaps; the measured arms do not preserve their ancillary coherences over that interval.

This is deliberately an explicit, falsifiable **native-Fredkin model**, not a hardware-calibrated two-qubit-gate decomposition. Replace the gate model for a real hardware claim. Do **not** compare a noisy compiled Fredkin against an ideal logical Fredkin and interpret it as a protocol advantage.

## How the simulation answers the referees

**Experiment A: ideal and readout sanity checks.** `validation.csv` and `readout.csv` verify that all ideal implementations access the same state power. If readout errors are independent, measured trees have visibility `(1−2q)^(N−1)` while coherent control has `(1−2q)`. The sampling-variance ratio grows as `(1−2q)^[−2(N−2)]` for N>2. This is a **negative** result about the tree and must be retained if supported by physical measurements.

**Experiment B: physical apparatus noise.** `gate_noise.csv` sweeps independent mechanisms and their combination. The actual physical signed operators are computed for all four implementations. Some models can favor measured instruments; others favor coherent control. The simulator never forces either conclusion.

**Experiment C: architecture-selection phase diagram.** `phase.csv` evaluates a two-parameter slice of readout error and additional coherent-control idle noise while retaining native Fredkin, control, reset and data-memory imperfections. To prevent misleading rankings, it rejects an implementation whose bias already exceeds the target RMSE or whose peak width exceeds `--width-cap`. Only feasible implementations are ranked, first by total Fredkin count and then, for ties, by coherent width and swap critical-path depth. The width cap defaults to 7 for M=2,N=4; a fully parallel 10-qubit circuit is excluded from that phase diagram but included in all other comparisons.

**Experiment D: mid-circuit algorithm placement.** `placement.csv` runs a known entangling algorithm with independent algorithmic gate noise. It allows **both** the measured-accumulator PQES and coherent-control VD baselines to act at the same intermediate placement, then compares their circuit calls, signed normalization, bias and variance. One experiment estimates `Tr(O Lambda_B(rho_A^N))/Tr(rho_A^N)` and the other estimates `Tr(O Lambda_B(rho_A)^N)/Tr(Lambda_B(rho_A)^N)`. These need not coincide for a noisy downstream channel.

**Experiment E: twirling.** `twirling.csv` distinguishes independent frame choices for each consumed copy, a shared frame across a multi-copy experiment, and a fixed empirical K-frame twirling channel. Only independently randomized copies reproduce `(E_f rho_f)^N` in expectation. There is no required exhaustive `3^M` enumeration.

**Experiment F: repeated ideal cycles.** `cycles.csv` retains the mathematically correct normalized recurrence, *but* also tracks the exponentially small signed trace `Z_t` and direct nested trajectory cost `N^t`. This is a resource warning, **not** simulation of a physical QEC memory.

**Physical layerwise VCP2.** `channel_physical.csv` uses the **actual controlled-SWAP--Pauli-channel--controlled-SWAP ancilla circuit** with one maximally mixed auxiliary data register, independent noise channels on both data registers, ancilla X measurement and full signed branch tracking. It is applied after each intended algorithmic gate. It compares physical order-two VCP with two physical output-only order-two state-VD implementations using the same elementary Pauli noise model. `vcp_physical.py` derives the channel-instrument identity and documents the distinct cost vectors: VCP2 uses one propagated target-state preparation, two noise-channel calls per location and **2M Fredkins per noise location**; state VD2 uses two completed circuit preparations and **M Fredkins total**. `channel_proxy.csv` separately verifies ideal Pauli probability-powering with **no apparatus cost** and is never ranked as a physical circuit. The physical VCP comparison is specifically order two, stochastic Pauli noise, ideal auxiliary I/d and target observable, not a general implementation of every layerwise VCP method.

## Important assumptions and limitations

1. **State copies:** each consumed input copy is independently prepared according to the specified noisy preparation channel. A quantum register is not cloned. Recycled reset errors are modeled phenomenologically as a mixture with the all-zero state before nominal state re-preparation.
2. **Quantum operations:** small dense density matrices allow exact circuit noise under the defined Markovian channels. No noisy trajectories are approximated by a direct `rho**N` operation. Because a control + two M-qubit registers is stored, costs grow exponentially with M (but not with N); M=1,2 are the main intended benchmarks.
3. **All register swaps:** implemented as M native Fredkin gates rather than a physical compilation into 1- and 2-qubit gates. Only hypothetical native three-body noise, not actual calibration fidelity, is represented.
4. **Timing:** a register-swap step is a normalized time unit; memory noise is stochastic Z dephasing per waiting unit. The code does not calculate a detailed instruction-level wall-clock schedule, reset timing or interconnect routing.
5. **Measurement:** the final benchmark observable is often the projector onto a known ideal target state, measured ideally. This is straightforward in simulation and conceptually feasible by ideal uncomputation, but the physical cost and noise of implementing that final measurement are not included.
6. **Estimator:** normalized signed ratios are not exactly unbiased at finite shots. The corrected bias coefficient is recorded. The phase-map RMSE cost is a conservative variance-based planning estimate and is not an experimental CI.
7. **Algorithmic noise:** is applied after every ideal one- or two-qubit gate in a fixed deterministic entangling circuit. The same downstream channel is used for all placement schedules. It is not the same operator as the native Fredkin noise model.
8. **VCP:** a physical order-two Pauli-channel superinstrument is included, but higher-order VCP, general coherent channels, realistic noisy maximally mixed preparation, and optimal VCP circuit decompositions are not modeled. The separate ideal proxy should not be hardware-ranked.
9. **Scope:** no postselected physical state distillation, fault-tolerant QEC memory, error-syndrome decoding, routing optimization, leakage, non-Markovian correlated errors, or surface-code comparison is claimed.

## CSV and figure reproducibility

Each task overwrites **only its own** CSV. `manifest.json` records the most recently invoked selection, profile, seed, RMSE and noise conventions. Therefore, after sequential single-task runs, it is useful to rerun `--task all` for a coherent manifest. Existing CSV files from other tasks remain untouched. All generated plots are regenerated from saved data; changing line width, text or axis limits does not require resimulation.

Most comparison tables include: `architecture`, `M`, `N`, the primitive probabilities, `normalization`, `estimate`, `bias`, `variance_coeff`, `ratio_bias_coeff`, `ordinary_expect`, `peak_qubits`, `register_swaps`, `native_fredkins`, `fresh_preparations`, `controls_measured` and `critical_swap_stages`.

For manuscript figures, `FIGURE_CAPTIONS.md` provides full parameter definitions and statements that still require a stronger real-hardware or compiled-two-qubit-gate validation before publication. All proposed figures are generated as **PDF** and **PNG (300 dpi)**. Matplotlib uses Times New Roman when the font is installed on the machine (as in your original plotting script); otherwise it uses an available serif alternative. No font files are distributed.

## Unit tests

```bash
python -m pytest -q pqes_new_numerics/tests/
```

These check the ideal identity for all architectures (including two-qubit entangled inputs), physical positivity and trace preservation under apparatus noise, readout-visibility attenuation and variance penalty, arbitrary integer powers, channel depolarization, unitary placement equivalence, noisy noncommutativity, and randomized-frame averaging. A failing test should block use of the corresponding plot in the manuscript.

## Research decisions before final claims

Before using these results to claim an architecture-specific advantage, (i) replace the native three-qubit Fredkin noise model with a shared compiled native-gate basis and calibrated noise, (ii) assign measured preparation, memory, measurement and reset times, (iii) optimize all competitor schedules including coherent low-width VD, (iv) present both favorable and unfavorable parameter regions and (v) extend the included physical order-two Pauli VCP comparator to a realistic two-qubit compiled gate set, non-Pauli errors, and multiple channel powers. The present suite gives **complete reproducible evidence for its explicitly specified toy hardware model**, not a device-independent superiority theorem.
