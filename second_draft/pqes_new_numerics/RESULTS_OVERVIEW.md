# Full-profile results: what the data show

**Data:** `data/pqes_new_numerics/`, produced by `run_simulations.py --profile full --seed 31 --rmse 0.03 --width-cap 7`.
- Tasks first run by hand were rerun individually and are recorded in `manifest.json`.
- The full profile takes about 590 s of compute (about 10 min wall), most of it in the twirl Monte Carlo (383 s) and the K-frame study (110 s).

**Model:**
- compiled CNOT–Toffoli–CNOT SWAP gadgets with the draft's two-qubit depolarizing channel;
- one shared clock with idle noise on every waiting qubit;
- resets charged per actual reuse;
- the reference hardware in `README.md`.

These are **generic toy-hardware results**, not a device calibration. Each statement below holds within that model and the stated regime.

## Validation

- **Tests and oracle:**
  - 86 tests pass, including the independent oracle and the regression test for every audit defect.
  - The oracle is a brute-force simulator with explicit Kraus sums, every register kept, every outcome string enumerated, and hand-derived clock waits.
  - Agreement is 10⁻¹³ under all noise mechanisms at once.
  - The perturbations tested (a one-unit wait error, a missing reset error or a missing ε₂ term) each give disagreements of about 10⁻³.
- **Ideal identities:** all six arms give R = ρ and W = ρᴺ to ≤ 2.7×10⁻¹⁵ (70 cases; M = 1, 2; N = 2, 4, 8).
- **Shot planning at RMSE 0.03:**
  - The Monte Carlo RMSE at the planned shot count is 0.961–1.075 times the target over all 1104 feasible phase-map cells.
  - The 95% CI covers the target in 91% of cells.
- **Estimator Monte Carlo** (`shots.csv`, 400 trials per point):
  - sd/√(v/S) ranges from 0.93 to 1.11 over 64 points, with no zero denominators;
  - the ratio bias agrees with c_A/S.
- **Breakdown regime:** the delta method fails for S·t² ≲ 10. At such points, sampled MSEs exceed the leading formula by orders of magnitude, and leading MSEs are reported as undefined.

## 1. Readout errors favour one-control arms (unfavourable for PQES)

With independent readout error q on every recorded outcome:
- the measured trees keep visibility (1−2q)^{N−1};
- the coherent, wide cyclic and derangement arms keep (1−2q).

At M = 2, N = 8, q = 0.08:

| Arm | v_A |
|---|---|
| Every measured tree | 831.07 |
| Coherent and wide cyclic | 102.56 |

The ratio is 8.103, which equals (1−2q)^{−12}.

Ancilla reset error acts the same way. It is a pure visibility factor charged N−2 times on the sequential and recycled trees, N/2−1 times on the parallel tree, and never on one-control arms.

## 2. Which mechanism hurts whom (M = 2, N = 4, one mechanism at a time)

**Two-qubit gate error on the data pair (ε₂)** is the only gate mechanism that biases the estimator:

| Arm | Bias change at ε₂ = 0.02 |
|---|---|
| Parallel and recycled trees | −0.038 |
| Sequential, coherent and wide | −0.041 |
| Derangement | −0.027 |

Balanced trees put fewer merges on the surviving path.

**Visibility-only mechanisms.** ε₁ (on the ancilla H), readout and ancilla reset change v only; the bias change is exactly 0.

**Idle dephasing** (5×10⁻³ per unit):

| Arm | Bias change |
|---|---|
| Wide cyclic | −0.008 |
| Parallel tree | −0.010 |
| Recycled tree | −0.014 |
| Derangement | −0.028 (spent registers wait for the final measurement) |

**Measurement latency** (t_meas = 20):
- trees: bias −0.004 to −0.008 and v × 1.18–1.32;
- one-readout arms: unaffected.

**Preparation time** (t_prep = 30), v multipliers:

| Arm | v × |
|---|---|
| Coherent streaming (its control idles during every re-preparation) | 1.80 |
| Sequential tree | 1.29 |
| Recycled tree | 1.16 |
| Parallel, wide and derangement | 1.00 |

## 3. Architecture cost at RMSE 0.03 (`phase.csv`, Figs B, B2)

The study uses M = 2, N = 4 and the entangled source with Pauli noise 0.14.
- Axes: readout q ∈ [0, 0.05] and idle rate ∈ [0, 4×10⁻³].
- Slices: ε₂ ∈ {0.002, 0.005, 0.01, 0.02} and fast/slow measurement.
- Costs: shots, two-qubit gates, duration and space-time volume, with and without a 7-qubit width cap.

**Uncapped:**
- The cyclic-derangement estimator (9 qubits) is cheapest in almost every feasible cell, needing about 2.5× fewer shots than single-register readout.
- Exceptions are at the highest idle rates, where its waiting registers decohere:
  - wide cyclic VD wins the 6 cells at idle 4×10⁻³ for ε₂ = 0.002 and 0.005 (and at ε₂ = 0.01 with slow measurement);
  - the parallel tree wins 6 cells at ε₂ = 0.01 with fast measurement.

**Width ≤ 7 qubits, ε₂ = 0.002 and 0.005:**
- Coherent streaming VD wins 28–36 of 36 cells.
- Measured trees win only at q = 0 or at the highest idle rates with fast measurement (5–8 cells). With slow measurement they win no cell.
- The coherent arm needs a median 0.72–0.86 times the shots of the best capped tree.

**Width ≤ 7 qubits, ε₂ = 0.01 (favourable for PQES):**
- The depth-first recycled tree is the cheapest capped arm in 23 of 30 feasible cells with fast measurement, and in 7 of 24 with slow measurement.
- With fast measurement, coherent streaming needs a median 1.13× (up to 1.71×) as many shots.
- The cause is the recycled tree's smaller gate-induced bias (e.g. −0.0234 vs −0.0257 at q = 0.01), which matters because the target is close to the bias floor (ε² − b² is small).
- At a larger target RMSE the margin shrinks; at a smaller one every capped arm becomes infeasible.

**ε₂ = 0.02:** no arm within 7 qubits reaches RMSE 0.03. Only the derangement arm does, in 6 cells.

**Feasibility:** infeasible cells (|bias| ≥ 0.03) occur at the highest idle rate for every ε₂, spread with ε₂, and dominate at ε₂ = 0.02.

## 4. Compression placement at fixed total depth (`placement.csv`, Fig C)

The circuit is `algorithm_blocks`, M = 2, b_repeats = 2: L = 13 layers (entangling) and L = 6 (product). Settings: ε₂ = 0.005, fast measurement, N = 4.

**Entangling family:**

| Configuration | Bias |
|---|---|
| Unmitigated | −0.0239 |
| Output-only VD, every arm | −0.0100 to −0.0108 |
| Compression at L_A/L = 0.5 | −0.0238 |
| Compression at L_A/L = 0.25 | −0.028 |

- **Mid-circuit compression is unfavourable here.** At L_A/L = 0.5 the bias is no better than unmitigated: the downstream noise is unfiltered and the SWAP gadget adds its own errors. Its two-qubit-gate saving does not compensate.
- **Total two-qubit gates to RMSE 0.03, output-only:**

| Arm | Gates |
|---|---|
| Derangement | 2368 |
| Wide cyclic | 3552 |
| Parallel tree | 4256 |
| Sequential tree | 5216 |
| Coherent streaming | 5472 |
| Recycled tree | 5472 |

- **Across ε₂ ∈ {0.002, …, 0.02},** output-only VD lowers |bias| about 2.3×, from −0.0108 to −0.0049 (ε₂ = 0.002) and from −0.0860 to −0.036 (ε₂ = 0.02, infeasible at 0.03).

**Product family (controlled benchmark; unfavourable for all VD):**
- The unmitigated bias is −0.0030.
- Every VD arm, at every cut, has bias −0.0085 to −0.0105: the distillation apparatus adds more error than this weakly noisy circuit contains.

**VCP2 at the same cut** is infeasible for every cut ≥ 0.25 in both families (§5).

## 5. Physical channel purification vs state VD (`channel_physical.csv`, Fig F)

Order-two, Liu-style channel purification: per layer, on the gate support, one reused control. The M = 2 entangling circuit has 13 layers.

**Ideal apparatus** (the algorithm gates and their idle noise are unchanged):
- VCP2 removes the gate noise, but not the algorithm's idle noise on qubits outside each layer.
- Its bias is about −0.0021 at every ε₂. Output-only state VD (N = 2) has bias −0.00004 (ε₂ = 0.002), −0.0002, −0.0008 and −0.0030 (ε₂ = 0.02).
- So VCP2 has the smaller bias only at ε₂ = 0.02.
- VCP2 costs 98 two-qubit gates, 44 Toffolis and 146 time units per shot, against 14 gates, 2 Toffolis and 18 units for state VD.
- Total two-qubit gates to RMSE 0.03 are 5.2–14.7×10³, against 0.74–1.9×10³ for state VD.

**Physical apparatus (the same compiled gates):**
- VCP2's own unpurified CSWAP errors, plus idle noise over its 8× longer schedule, give bias −0.127 (ε₂ = 0.002) to −0.458 (ε₂ = 0.02).
- That is worse than the unmitigated circuit, and infeasible at every ε₂.
- With 10× better gadget gates (readout, resets and idle unchanged) the bias is still −0.078 to −0.127.
- The only physically feasible VCP2 row is M = 1, ε₂ = 0.002, fast measurement.
- Output-only state VD is feasible for ε₂ ≤ 0.01 (bias −0.005 to −0.019).

This result is specific to order two, this compilation (one control per layer, no parallel controls), and gate-support auxiliaries prepared ideally.

## 6. Randomized Clifford frames (Fig D)

**Fixed K-frame sets** (p = 0.45, θ = π/2, N = 4; frames sampled with replacement, 100 sets per K): the mean fidelity error falls as K grows:

| M | K = 1 | K = 243 |
|---|---|---|
| 1 | 0.146 | 0.0006 |
| 5 | 0.478 | 0.0028 |

The power-state trace distance falls roughly as K^{−1/2}; for M = 5 it goes from 0.67 to 0.066.

**Fresh frames for every copy and every shot:**
- The unbiased numerator and denominator match Tr(P ρ̄ᴺ) and Tr ρ̄ᴺ (|z| ≤ 3.0 over 192 configurations, M up to 5).
- Where S·t² ≥ 50, the ratio matches the prediction with |z| ≤ 2.4, and sd/√(v/S) is 0.88–1.10.
- No enumeration of the 3ᴹ frames is needed.

**Exact fidelity** (θ = π/2, N = 4, M = 2), independent / shared / untwirled:

| p | Independent | Shared | Untwirled |
|---|---|---|---|
| 0.45 | 0.936 | 0.748 | 0.477 |
| 0.80 | 0.137 | 0.230 | 0.000 |

Past the twirled boundary (p = 0.8 > 3/4), shared frames exceed independent ones.

**Fixed budget** (C = 10⁴ copies, best N, M = 2):
- Twirling lowers the RMSE, e.g. 0.592 → 0.213 at θ = π/2, p = 0.45, and 0.213 → 0.070 at p = 0.3.
- Clifford gate error ε_C = 0.02 raises the RMSE by at most 12%.
- At these dephasing strengths every RMSE stays far above 0.03, because bias dominates.

## 7. Repeated virtual cycles (Fig E)

The normalized recurrence can keep a high fidelity while the nested signed-ensemble cost explodes. At p = 0.15, global depolarizing, D = 4:
- **N = 4:** F_5 ≈ 1.000, yet log₁₀ Z_5 = −70.7, so one estimate to RMSE 0.03 needs about 10^147 original copies.
- **N = 2:** the budget is about 10^10.6 copies at t = 5 and about 10^218 at t = 10.

Local depolarizing (M = 2) behaves the same: at N = 4, about 10^340 copies at t = 5. The repeated map is therefore not an operational memory beyond a few cycles.

## 8. Global and local depolarizing benchmarks (Figs G, H)

**Global saddle-node p_SN(D,N)** (numerical):

| D | N = 2 | N = 4 | N = 8 |
|---|---|---|---|
| 2 | 0.500 | 0.750 | 0.875 |
| 4 | 0.536 | 0.781 | 0.895 |
| 8 | 0.602 | 0.833 | 0.925 |

- For N = 2 these equal D/(D+2√(D−1)).
- For D = 2 they equal p_c(N) = 1 − 1/N.

**Global optimal N** at C = 10⁴ copies:
- D = 4: N = 2 for p ≤ 0.25, N = 4 for 0.26–0.51, N = 2 beyond.
- D = 8: N = 2 throughout.

**Local depolarizing** (|+⟩^M, C = 10⁴):
- The optimal N is 2–4.
- Sampled and leading MSE agree to a median 1.8% (90th percentile 5.6%) wherever S·t² ≥ 50.
- For M = 5 at p ≥ 0.5 the optimal choice is **N = 1**, i.e. no distillation (unfavourable).
- The entangled target cos(π/8)|00⟩ + sin(π/8)|11⟩ saturates at its eigenvector-mismatch floor (F_16 = 0.9851 at p = 0.3, equal to the floor). The small-p prediction 1 − p²/9 holds at small p.

## Claim boundaries

- **What the data support:**
  - an explicit, unfavourable readout result for measured trees;
  - a narrow favourable regime for the depth-first recycled tree (width ≤ 7, ε₂ ≈ 0.01, fast measurement, target RMSE near the bias floor);
  - no advantage elsewhere;
  - larger gains from estimators that read out every register (cyclic derangement), whenever width allows.
- **Unfavourable for compression and channel purification:** mid-circuit compression and compiled VCP2 are both unfavourable at these parameters.
- **What could move these conclusions:** a different compilation (native CSWAP, parallel controls), a different noise model (coherent or correlated errors) or a calibrated device. None is modelled here.
- **No claim** of a device-independent or hardware advantage of PQES over other virtual-distillation variants is supported.
