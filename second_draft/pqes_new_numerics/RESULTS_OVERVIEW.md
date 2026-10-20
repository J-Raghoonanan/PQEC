# Full-profile numerical run: technical summary and interpretation

**Data source:** `data/pqes_new_numerics/`, produced by `run_simulations.py --profile full --seed 31 --rmse 0.12 --width-cap 7`. Figures regenerated independently by `plot_results.py`. The `manifest.json` is the authoritative record of the run conventions. The results are for an **explicitly defined hypothetical native-Fredkin, Markovian noise model**, not a calibrated quantum computer. These values may be quoted only together with their noise-model qualifications.

## Validation

- **43 independent unit tests passed.**
- The maximum Frobenius-norm residual for `R=rho` across the saved ideal validation cases was below `3e-15`; the maximum for `W=rho**N` was below `2e-15`.
- Physical outcomes for the noisy measured SWAP circuit and noisy order-two VCP superinstrument remain positive, and the unconditional maps preserve trace.
- Independent parity readout errors obey their exact analytic attenuation formulas; the ideal unitary covariance/placement relation was checked directly.

## Independent readout errors provide a counterexample to any universal SWAP-tree advantage

For the simulated two-qubit entangled target, power `N=8`, assignment error `q=0.08`, and otherwise ideal apparatus:

| Physical architecture | Signed normalization | Leading variance coefficient |
|---|---:|---:|
| Parallel measured SWAP tree | 0.018734 | 831.066879 |
| Recycled balanced SWAP tree | 0.018734 | 831.066879 |
| Sequential measured accumulator | 0.018734 | 831.066879 |
| Coherent-control streaming VD | 0.053329 | 102.562219 |

The ratio `831.066879/102.562219` matches the predicted `(1−2q)^(−2(N−2))`. **Measured SWAP parity is more costly in this model** because the product of N−1 noisy reported outcomes has lower visibility. This is a methodological validation and a scientifically important unfavorable case for PQES.

## Width- and coherence-constrained tradeoff

The `phase.csv` table contains a grid across assignment error, additional coherent-control idle Z noise, native-Fredkin depolarization and data idle noise. It includes the systematic estimator bias, variance coefficient, peak width and a leading-asymptotic planning estimate for shots and Fredkin count needed at RMSE 0.12.

A maximum width of seven coherent qubits applies only in this specific phase-map study; the 10-qubit parallel balanced tree is excluded from its cost ranking while retained in other studies. The chosen native-Fredkin model, wait-time schedule, and width limit determine the winning region. A real hardware advantage cannot be claimed without a matched native two-qubit gate compilation, calibrated reset and measurement timings, and physical-control constraints.

## Physical order-two VCP versus physical state VD

An additional experiment models the **actual order-two VCP quantum instrument** (two controlled swaps surrounding two Pauli-channel instances, one maximally mixed auxiliary input and measured ancilla) at every circuit noise location. These trajectories are compared with measured and coherent output-only state VD2 circuits under the same per-location Pauli noise model.

For `M=2`, 2 repetitions of the downstream algorithmic block, Pauli noise probability `p_channel=0.06`, and physical apparatus `gate_p=0.008`, `control_z_p=0.005`, `readout_q=0.012`:

| Protocol | Estimated target expectation | Absolute bias | Variance coefficient | Native Fredkins/trajectory | Noisy channel calls | Target-state preparations |
|---|---:|---:|---:|---:|---:|---:|
| Measured state VD2 | 0.641873 | 0.358127 | 3.619423 | 2 | 34 | 2 |
| Coherent state VD2 | 0.641873 | 0.358127 | 3.619423 | 2 | 34 | 2 |
| Physical layerwise VCP2 | 0.977379 | 0.022621 | 3500.757083 | 68 | 34 | 1 |

This shows that **lower systematic bias can coexist with much higher sampling variance** and very different resource requirements. At the listed parameters, the VCP2 normalization is so small that its physical estimation may be statistically more expensive despite having the more accurate asymptotic mean. The comparison does **not** imply broad superiority of state VD or VCP. It is limited to order two and a stochastic Pauli channel, with ideal auxiliary preparation and final target measurement.

## Ideal repeated noise–power cycles are not deterministic physical memory correction

The repeated-cycle CSV records both virtual-state fidelity and `log10 Z_t`, where `Z_t` is the normalization of a direct nested signed-ensemble realization. For example, the simulated `M=2` nonmaximally entangled input, local Pauli `p=0.25`, purification power `N=4`, at `t=9` normalized cycles yields a mathematically defined virtual fidelity of approximately `0.7667`, but the nested trajectory consumes `4^9=262144` original copies, and `log10(Z_t)≈−71931`. Thus its signed estimator is not a practically accessible deterministic corrected memory. Do not interpret the virtual normalized trajectory alone as physical error-correction performance.

## Current claim boundaries

The provided full-profile data are reliable **within the explicitly documented simulation** and support detailed apparatus comparisons. They **do not yet prove** a device-independent advantage for PQES over VD or VCP. For publication, the remaining improvements are native 1q/2q compilation, realistic ancilla readout and reset scheduling, independently calibrated noise channels, sampling confidence intervals for selected optimized points, and an order-higher VCP comparison if that becomes necessary for the target journal.
