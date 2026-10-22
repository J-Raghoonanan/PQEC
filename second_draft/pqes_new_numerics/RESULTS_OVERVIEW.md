# Full-profile results: what the data show

**Data:** `data/pqes_new_numerics/`, produced by `run_simulations.py --profile full --seed 31 --rmse 0.03`. The run takes about 590 s of compute (10 min wall), most of it in the twirl Monte Carlo (388 s) and the K-frame study (112 s).

**Model:**
- compiled CNOT–Toffoli–CNOT SWAP gadgets with the draft's two-qubit depolarizing channel;
- one shared clock, with idle noise on every waiting qubit;
- resets charged per actual reuse;
- ideal observable readouts in every arm (readout error acts only on recorded sign and control bits);
- spent derangement registers read out mid-circuit as soon as their CSWAP ends;
- VCP2 purifying only layers that contain a two-qubit gate, each with a fresh control;
- the reference hardware in `README.md` (ε₁ = ε₂/10).

These are **generic toy-hardware results**, not a device calibration. Each statement holds within that model and the stated regime.

## Validation

- **Tests and oracle:**
  - 97 tests pass, including the independent oracle and a regression test for every audit and review defect.
  - The oracle is a brute-force simulator with explicit Kraus sums, every register kept, every outcome string enumerated, and hand-derived clock waits.
  - Agreement is 10⁻¹³ under all noise mechanisms at once.
- **Ideal identities:** all six arms give R = ρ and W = ρᴺ to ≤ 2.7×10⁻¹⁵ (70 cases; M = 1, 2; N = 2, 4, 8).
- **Shot planning at RMSE 0.03:**
  - The Monte Carlo RMSE at the planned shot count is 0.961–1.071 times the target over all 1152 feasible phase-map cells.
  - The 95% CI covers the target in 90% of cells.
- **Estimator Monte Carlo** (`shots.csv`, 400 trials per point):
  - sd/√(v/S) ranges from 0.93 to 1.11 over 64 points, with no zero denominators;
  - the ratio bias agrees with c_A/S.
- **Breakdown regime:** where S·t² ≲ 10 the delta method fails. There, sampled MSEs exceed the leading formula, and leading MSEs and copy budgets are reported as undefined.

## 1. Readout errors (unfavourable for PQES)

Measured trees keep visibility (1−2q)^{N−1}; the one-readout coherent, wide cyclic and derangement arms keep (1−2q).

At M = 2, N = 8, q = 0.08:

| Arm | v_A |
|---|---|
| Every measured tree | 831.07 |
| Coherent and wide cyclic | 102.56 |

The ratio is 8.103 = (1−2q)^{−12}.

Readout error, like ancilla reset error, is a pure visibility factor in every arm: it never biases the estimate. Ancilla reset error is charged N−2 times on the sequential and recycled trees, N/2−1 times on the parallel tree, and never on one-control arms.

## 2. Which mechanism hurts whom (M = 2, N = 4, one mechanism at a time)

**Two-qubit gate error on the data pair (ε₂ = 0.02):**

| Arm | Bias change |
|---|---|
| Parallel and recycled trees | −0.038 |
| Sequential, coherent and wide | −0.041 |
| Derangement | −0.027 |

Balanced trees put fewer merges on the surviving path.

**Visibility-only mechanisms** (they change v only; bias change is exactly 0): ε₁, readout and ancilla reset.

**Idle dephasing** (5×10⁻³ per unit), bias change:

| Arm | Bias change |
|---|---|
| Wide cyclic | −0.008 |
| Derangement | −0.008 |
| Parallel tree | −0.010 |
| Sequential | −0.012 |
| Coherent streaming | −0.012 |
| Recycled tree | −0.014 |

**Measurement latency** (t_meas = 20):
- trees: bias −0.004 to −0.008 and v × 1.18–1.32;
- one-readout arms: unaffected.

**Preparation time** (t_prep = 30), v multipliers:

| Arm | v × |
|---|---|
| Coherent streaming (its control idles through every re-preparation) | 1.80 |
| Sequential tree | 1.29 |
| Recycled tree | 1.16 |
| Parallel, wide and derangement | 1.00 |

## 3. Architecture cost at RMSE 0.03 (`phase.csv`, Figs B, B2, B3)

The study uses M = 2, N = 4 and the entangled source with Pauli noise 0.14.
- Axes: readout q ∈ [0, 0.05] and idle rate ∈ [0, 4×10⁻³].
- Slices: ε₂ ∈ {0.002, 0.005, 0.01, 0.02} and fast/slow measurement.
- Widths: sequential and coherent 5, recycled 7, wide cyclic and derangement 9, parallel 10.

**Primary result: no width constraint.**
- The cyclic-derangement estimator is the cheapest arm in **every** feasible cell of every slice, for all four cost metrics (shots, two-qubit gates, duration, space-time volume): 36 of 36 cells for ε₂ ≤ 0.01, and all 6 feasible cells at ε₂ = 0.02.
- The best single-register arm needs a median 2.5–5.1× its shots (2.4–13.7× across cells).
- No PQES tree is cheapest anywhere without a width constraint.

**Width as an explicit budget (Fig B3; Pareto reporting).**

*Width ≤ 5 (sequential or coherent):*
- coherent streaming wins most cells, e.g. 30 of 36 at ε₂ = 0.005 with fast measurement, with the measured accumulator taking the other 6;
- at ε₂ = 0.01 only 24 cells have a feasible arm of width ≤ 5.

*Width ≤ 7 (adds the recycled tree):*
- ε₂ = 0.002 and 0.005: coherent streaming still wins 29–36 cells (of the 30–36 cells with a feasible arm within the budget); its median shot ratio to the best tree is 0.72–0.86.
- ε₂ = 0.01, fast: **the depth-first recycled tree wins 23 of the 30 cells** that have a feasible arm within the budget; coherent streaming needs a median 1.13× (up to 1.71×) its shots. With slow measurement the recycled tree wins 7 of 24.

*Width ≤ 9:* the derangement arm wins every feasible cell.

Any statement that a PQES tree is preferable is therefore **conditional on a width budget below 9 qubits**, the width of the wide-cyclic and derangement arms.

**Feasibility:** infeasible cells (|bias| ≥ 0.03) grow with the idle rate and with ε₂. At ε₂ = 0.02 only the derangement arm is feasible, in 6 of 36 cells.

## 4. Compression placement at fixed total depth (`placement.csv`, Fig C)

The study circuits are:
- **entangling:** L = 12 layers, the first 12 of U_B² U_A; 4 layers contain a two-qubit gate;
- **product:** L = 8 layers, U_B³ U_A; no two-qubit gates.

Both depths are divisible by 4, so the cuts L_A/L = 0, ¼, ½, ¾, 1 are realized exactly. Settings: ε₂ = 0.005, ε₁ = ε₂/10, fast measurement, N = 4.

**Entangling family:**

| Configuration | Bias |
|---|---|
| Unmitigated | −0.0202 |
| Output-only VD: derangement | −0.0066 |
| Output-only VD: parallel and wide cyclic | −0.0098 |
| Output-only VD: sequential, coherent, recycled | −0.0100 |
| Compression at L_A/L = 0.5 (single-register arms) | −0.0202 |
| Compression at L_A/L = 0.25 (single-register arms) | −0.024 |

- **Mid-circuit compression is unfavourable.** At L_A/L = 0.5 the single-register arms are no better than unmitigated, and at 0.25 they are worse: the downstream noise is unfiltered and the gadget adds its own errors. The derangement arm at 0.5 (−0.0172) is only slightly better than unmitigated.
- **Total two-qubit gates to RMSE 0.03, output-only:**

| Arm | Gates |
|---|---|
| Derangement | 2016 |
| Wide cyclic | 2744 |
| Parallel tree | 3332 |
| Sequential tree | 4032 |
| Coherent streaming | 4200 |
| Recycled tree | 4256 |

- **Leading RMSE at C = 10⁴ copies:** derangement 0.0072, wide 0.0113, parallel 0.0116, coherent and sequential 0.0121, recycled 0.0122.
- **Across ε₂ (L_A/L = 1), unmitigated vs output-only VD (N = 4):**

| ε₂ | Unmitigated | Output-only VD |
|---|---|---|
| 0.002 | −0.0094 | −0.0033 to −0.0050 |
| 0.005 | −0.0202 | −0.0066 to −0.0100 |
| 0.01 | −0.0380 | −0.0121 to −0.0185 |
| 0.02 | −0.0725 | −0.024 to −0.036 (only the derangement arm is feasible) |

**VCP2 at the same cut** (purifying only the cut's layers that contain a two-qubit gate) has bias −0.031 at L_A/L = ¼ and −0.069 at 1, worse than unmitigated: infeasible for every cut ≥ ¼.

**Product family (controlled benchmark; unfavourable for all VD):**
- The unmitigated bias is −0.0040.
- Every VD arm, at every cut, has bias −0.0061 to −0.0115.
- VCP2 purifies no layer here, because there is no two-qubit gate, so it equals the unmitigated circuit (−0.0040) at every cut.

## 5. Physical channel purification vs state VD (`channel_physical.csv`, Fig F)

Order-two, Liu-style channel purification on the gate support of each layer that contains a two-qubit gate (4 of the 12 layers of the M = 2 entangling study circuit). The 8 one-qubit-only layers run unpurified with ε₁ = ε₂/10.
- Each purified layer has its own freshly prepared control, from the smallest pool that keeps measurement and reset off the main register's critical path: 2 pairs with fast measurement, 3 with slow.
- The M = 1 circuit and the M = 2 product circuit contain no two-qubit gate, so there VCP2 equals the unmitigated circuit.

**Per-shot cost:**

| Protocol | 2q gates | Toffolis | Duration (units) | Width (qubits) |
|---|---|---|---|---|
| VCP2 | 40 | 16 | 45 (fast) / 54 (slow) | 8 (fast) / 11 (slow) |
| State VD, N = 2 | 12 | 2 | 17 | 5 |

**Ideal apparatus** (the algorithm gates and their idle noise are unchanged):
- VCP2 removes the noise of the entangling layers only. Its bias is −0.0034 (ε₂ = 0.002) to −0.0155 (ε₂ = 0.02).
- Output-only state VD has bias −0.00003 to −0.0020. **VCP2 never has the smaller bias.**
- Total two-qubit gates to RMSE 0.03 are 2.1–5.7×10³ for VCP2, against 0.62–1.3×10³ for state VD.

**Apparatus gates 10× better** (readout, resets and idle unchanged):
- VCP2 bias is −0.015 (ε₂ = 0.002) to −0.047 (ε₂ = 0.02), feasible for ε₂ ≤ 0.01.
- It needs 2.7×10³, 5.5×10³ and 1.2×10⁵ two-qubit gates, against 0.66–0.74×10³ for state VD.

**Physical apparatus (the same compiled gates):**
- VCP2 bias is −0.035 (ε₂ = 0.002) to −0.218 (ε₂ = 0.02), worse than the unmitigated circuit (−0.009 to −0.072) and infeasible at every ε₂.
- State VD (N = 2) is feasible for ε₂ ≤ 0.01, with bias −0.0048, −0.0099 and −0.0185 for the single-register arms and −0.0050, −0.0100 and −0.0186 for the derangement arm.

This result is specific to order two, this compilation, and gate-support auxiliaries prepared ideally.

## 6. Randomized Clifford frames (Fig D)

**Fixed K-frame sets** (p = 0.45, θ = π/2, N = 4; frames sampled with replacement, 100 sets per K): the mean fidelity error falls as K grows:

| M | K = 1 | K = 243 |
|---|---|---|
| 1 | 0.146 | 0.0006 |
| 5 | 0.478 | 0.0028 |

The power-state trace distance falls roughly as K^{−1/2}; for M = 5 it goes from 0.67 to 0.066.

**Fresh frames for every copy and every shot:**
- The unbiased numerator and denominator match Tr(P ρ̄ᴺ) and Tr ρ̄ᴺ (|z| ≤ 3.0 over 192 configurations, M up to 5).
- Where S·t² ≥ 50, the ratio matches the prediction with |z| ≤ 2.5, and sd/√(v/S) is 0.88–1.10.
- No enumeration of the 3ᴹ frames is needed.

**Exact fidelity** (θ = π/2, N = 4, M = 2), independent / shared / untwirled:

| p | Independent | Shared | Untwirled |
|---|---|---|---|
| 0.45 | 0.936 | 0.748 | 0.477 |
| 0.80 | 0.137 | 0.230 | 0.000 |

Past the twirled boundary (p = 0.8 > 3/4), shared frames exceed independent ones.

**Fixed budget** (`twirl_budget.csv`, M = 2, best N, C = 10⁴ copies), untwirled → twirled RMSE:

| θ | p | Untwirled | Twirled |
|---|---|---|---|
| π/2 | 0.45 | 0.642 | 0.287 |
| π/2 | 0.3 | 0.287 | 0.070 |
| π/3 | 0.45 | 0.425 | 0.287 |

- Clifford gate error ε_C = 0.02 raises the RMSE by at most 12%.
- At these dephasing strengths every RMSE stays far above 0.03, because bias dominates.

## 7. Repeated virtual cycles (Fig E)

Here `p` is Pauli-form for local depolarizing and replacement-form for global depolarizing; the two channels are not the same at equal p.

The normalized recurrence can keep a high fidelity while the nested signed-ensemble cost explodes. Global depolarizing, D = 4, p = 0.15:
- **N = 4:** F_5 ≈ 1.000, yet log₁₀ Z_5 = −70.7, so one estimate to RMSE 0.03 needs about 10^147 original copies.
- **N = 2:** about 10^10.6 copies at t = 5, and about 10^218 at t = 10.

Local depolarizing (M = 2), p = 0.15: the virtual fidelity itself falls below 0.97 (F_5 = 0.931 at N = 4, 0.897 at N = 2). Those cycles cannot reach RMSE 0.03 at any shot count; 143 of 480 cycle rows are infeasible.

## 8. Global and local depolarizing benchmarks (Figs G, H)

**Global saddle-node p_SN(D,N)** (replacement form, numerical):

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

**Local depolarizing** (Pauli form, |+⟩^M, C = 10⁴):
- The optimal N is 2–4.
- Over all `local_resource.csv` rows at C = 10⁴ with S·t² ≥ 50, sampled and leading MSE agree to a median 1.8% (90th percentile 5.6%).
- For M = 5 at p ≥ 0.5 the delta-method optimum is **N = 1**, i.e. no distillation (unfavourable).
- The entangled target cos(π/8)|00⟩ + sin(π/8)|11⟩ saturates at its eigenvector-mismatch floor (F_16 = 0.9851 at p = 0.3, equal to the floor). The small-p prediction 1 − p²/9 holds at small p.

## Claim boundaries

**What the data support:**
- an unfavourable readout result for measured trees;
- the cyclic-derangement estimator (9 qubits at M = 2, N = 4) winning every feasible cell when width is not constrained;
- one narrow favourable regime for PQES: the depth-first recycled tree under a width budget of 7 qubits, at ε₂ ≈ 0.01, fast measurement, and a target RMSE near the bias floor.

**Unfavourable for compression, channel purification and weak noise:**
- Mid-circuit compression and compiled VCP2 are unfavourable at these parameters.
- VCP2 becomes feasible only with an ideal or 10×-better apparatus, and even then needs 3–170× the two-qubit gates of state VD.
- Every VD arm loses to no mitigation on the weakly noisy product benchmark.

**What could move these conclusions:** a different compilation (native CSWAP, parallel controls for the trees), a different noise model (coherent or correlated errors) or a calibrated device. None is modelled here.

**No claim** of a device-independent or hardware advantage of PQES over other virtual-distillation variants is supported.
