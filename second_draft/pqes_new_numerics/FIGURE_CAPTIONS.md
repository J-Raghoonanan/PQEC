# Draft figure captions

These captions describe exactly what `plot_results.py` draws from the full-profile CSVs; they are not yet in the Overleaf draft. Parameter conventions are in `README.md` and `manifest.json`.

Common settings for every figure:
- "Physical" means the compiled two-qubit-gate model with the shared clock.
- The reference hardware is: ε₁ = ε₂/10, ε₃ = ε₂, readout q = 0.01, reset errors 0.005, idle rates 5×10⁻⁴ per unit, t_prep = 3.
- **Fast** measurement: t_meas = t_reset = 1. **Slow** measurement: t_meas = 10, t_reset = 5.
- The target RMSE is 0.03.

**Do not claim** anywhere: a model-independent or hardware-independent advantage of any architecture.

---

## Figure A — Readout errors and estimator statistics
`PQES_FigA_readout_and_shots.pdf`

The SWAP instruments are ideal except for independent assignment errors q on every recorded ancilla outcome. Panels (a) and (b) use an M = 2 nonmaximally entangled target with Pauli-form source noise 0.18. The cases shown are N = 4, 8.

**(a)** Relative signed normalization t(q)/t(0).
- The measured trees (N−1 records) follow (1−2q)^{N−1}.
- The coherent, wide cyclic and derangement arms (one record) follow (1−2q).

**(b)** Variance ratio v_tree/v_coh against the exact prediction (1−2q)^{−2(N−2)}.

**(c)** Direct Monte Carlo of the ratio estimator. The plotted quantity is the sample standard deviation divided by √(v_A/S), with 95% χ² intervals.
- Setup: M = 1, ε₂ = 0.005, q ∈ {0, 0.05, 0.1}, N = 4.
- Trials: 400 per point, with independent seeds per shot count.

**(d)** S times the Monte Carlo mean bias, against the leading ratio-bias coefficient c_A, with 95% CI.

This figure is an **unfavourable** result for the measured trees: readout errors alone favour one-control implementations for N > 2.

## Figure A2 — One-at-a-time apparatus mechanisms
`PQES_FigA2_apparatus_mechanisms.pdf`

Setup: M = 2 entangled target, source noise 0.13, N = 4. All six arms. Each column sweeps one mechanism from zero with all others off:
- ε₂, ε₁, readout q;
- ancilla and data reset errors;
- idle dephasing and idle amplitude damping;
- measurement latency and preparation time (these two at idle rate 10⁻³ per unit).

**Top row:** change of the limiting bias. **Bottom row:** v_A relative to the start of the sweep.

The ε₁, readout and ancilla-reset mechanisms only rescale the signed operator; their bias change is exactly zero and they are marked "visibility only". Observable readouts are ideal in every arm. Gate errors on the data pair (ε₂) and idle noise bias every arm.

## Figure B — Lowest-cost architecture at fixed RMSE, no width constraint (primary result)
`PQES_FigB_architecture_cost_map.pdf`

Setup: M = 2, N = 4, the entangled target with source noise 0.14, ε₂ = 0.005.
- **Axes:** readout error q (horizontal) and idle rate γ = φ per unit (vertical).
- **Rows:** fast and slow measurement.
- **Columns** (four cost metrics, each to RMSE 0.03): shots, total two-qubit gates, total duration, total space-time volume.
- **Cell labels:** the cheapest arm among all six, whatever its width; `*` marks a runner-up within 5%.
- **Widths** (printed on the figure): sequential and coherent 5, recycled 7, wide cyclic and derangement 9, parallel 10.

The CSV also holds ε₂ = 0.002, 0.01 and 0.02. The boundaries follow from the noise model; a different device can move them.

## Figure B3 — Winner under explicit width budgets
`PQES_FigB3_width_budgets.pdf`

The cheapest arm by shots at RMSE 0.03 when only arms of width ≤ 5, ≤ 7, ≤ 9 or unconstrained are allowed.
- **Rows:** ε₂ = 0.005 and 0.01, each at fast and slow measurement.
- **Blank cells:** no arm within the budget is feasible.

Width is a reported budget axis, not a default. A PQES tree appears as cheapest only under budgets below the 9-qubit width of the wide-cyclic and derangement arms; with no constraint the cyclic-derangement arm wins every feasible cell.

## Figure B2 — Validation of the shot planning
`PQES_FigB2_planning_validation.pdf`

For every feasible phase-map cell (every arm and slice), the figure shows the RMSE achieved by the raw ratio estimator at the planned shot count. It comes from 2000 Monte Carlo repetitions and carries a 95% CI; the target is 0.03.

## Figure C — Compression placement at fixed total depth
`PQES_FigC_compression_placement.pdf`

The circuits are fixed `algorithm_blocks` circuits with M = 2 (`circuits.study_layers`):
- **entangling:** L = 12 layers, the first 12 of U_B² U_A;
- **product:** L = 8 layers, U_B³ U_A.

Both depths are divisible by 4, so the plotted realized cut/L ∈ {0, ¼, ½, ¾, 1} equals the requested fraction exactly. Every gate is followed by the same compiled-model noise: ε₂ = 0.005, fast measurement.

Compression follows the first L_A layers:
- **State VD arms (N = 4):** distil ρ_A, then the remaining layers act once on the surviving register. For the derangement arm they act on every register.
- **VCP2:** purifies the layers before the cut that contain a two-qubit gate and leaves the rest unpurified.
  - Entangling circuit: 4 of the 12 layers contain a two-qubit gate.
  - Product circuit: none of its 8 layers do, so VCP2 there is the unmitigated circuit at every cut.
  - ε₁ = ε₂/10 throughout.
- **Endpoints:** L_A/L = 1 is output-only VD and full layerwise VCP2. L_A/L = 0 is unmitigated for VCP2.

Panels, one row per family (entangling, product):
- **(a, e)** limiting |bias|;
- **(b, f)** v_A;
- **(c, g)** total two-qubit gates to reach RMSE 0.03, where missing points are infeasible (|bias| ≥ 0.03);
- **(d, h)** leading RMSE at copy budgets C = 10³ (dashed), 10⁴ (solid) and 10⁵ (dotted), drawn only where S·t² ≥ 50.

Slow measurement and the other ε₂ values are in `placement.csv`. Mid-circuit compression leaves downstream noise unfiltered. Its bias is reported as it is; it is not hidden behind the gate savings.

## Figure D — Randomized Clifford frames
`PQES_FigD_randomized_twirling.pdf`

The noise is local Z dephasing p on product targets of polar angle θ (φ = π/4). The frames are U ∈ {I, H, HS}, which map the Z error to Z, X, Y. Noisy Cliffords carry D_{ε_C} after each nontrivial frame gate.

**(a)** Fixed empirical K-frame sets against the independent twirl, at p = 0.45, θ = π/2, N = 4, M = 1, 2, 5. Each point averages 100 frame sets (95% CI). Two curves per M:
- the mean fidelity error;
- the power-state trace distance.

**(b)** Shot-level Monte Carlo with fresh, independent frames for every copy and every shot.
- Setup: ideal sequential SWAP tree, M = 2, θ = π/3, ε_C = 0.01.
- Plotted: mean ± sd over 200 repetitions.
- Reference lines: Tr(P ρ̄ᴺ)/Tr ρ̄ᴺ (solid) and the shared-frame value (dotted).
- Circled points have S·t² < 50.

**(c)** Exact fidelity with independent frames, shared frames, and no twirl, for θ = π/2 and N = 4.

**(d)** Best-N leading RMSE at C = 10⁴ copies, against the Clifford error ε_C, for M = 2. The dotted line is untwirled.

The CSVs also cover θ = π/3, M = 1, 2, 5, p ∈ {0.45, 0.55, 0.7, 0.8} and N = 2, 4. No fixed fraction of 3ᴹ frames is used anywhere.

## Figure E — Repeated virtual cycles and their nested cost
`PQES_FigE_repeated_cycle_cost.pdf`

Setup: local depolarizing (M = 2 entangled target) and global depolarizing (D = 4), p = 0.15, N = 2, 4, 8.

The two `p` are different channels:
- local: Pauli form (1−p)ρ + p/3 ΣPρP;
- global: replacement form (1−p)ρ + p I/D.

Curves are compared within a family, not across families at equal p. Points are missing where the virtual fidelity's bias already exceeds 0.03, so no shot count reaches RMSE 0.03.

Panels:
- **(a)** the formal normalized fidelity F_t;
- **(b)** log₁₀ Z_t;
- **(c)** log₁₀ of the original copies N^t·S needed for RMSE 0.03, with S from the corrected variance and S·t² ≥ 50;
- **(d)** log₁₀ of the SWAP tests (N^t−1)·S.

The normalized recurrence is not a physical memory. The nested signed-ensemble cost is the operational one.

## Figure F — Physical order-two channel purification vs state VD
`PQES_FigF_vcp2_vs_state_vd.pdf`

Setup: M = 2 entangling study circuit, L = 12 layers, fast measurement, N = 2 for state VD.

VCP2 purifies only the 4 of 12 layers that contain a two-qubit gate; the 8 one-qubit-only layers run unpurified with ε₁ = ε₂/10.

Each purified layer has its own freshly prepared control and auxiliary, drawn round robin from the smallest pool that keeps measurement and reset off the main register's critical path (2 pairs here). The physical controls and auxiliaries are included in the width.

Output-only state VD arms are compared with layerwise VCP2 under three apparatus cases:
- **physical:** the same hardware as the algorithm;
- **10× better:** gadget gate errors divided by 10; readout, resets and idle unchanged;
- **ideal:** a perfect apparatus.

Panels:
- **(a)** |bias| against ε₂;
- **(b)** v_A;
- **(c)** total two-qubit gates to reach RMSE 0.03, feasible points only;
- **(d)** per-shot resources at ε₂ = 0.005, physical apparatus: copies, two-qubit gates, Toffolis, duration, width.

VCP2 counts 2G_purified + G_unpurified noisy-gate executions, and one auxiliary preparation, control preparation and measurement per purified layer. Results are order two only.

## Figure G — Global depolarizing: exact results
`PQES_FigG_global_depolarizing.pdf`

**(a)** Fixed point reached by the repeated map iterated from the pure target, for D = 2, 4, 8 and N = 2, 4, 8.
- Vertical lines mark the numerical saddle-node p_SN(D,N).
- For N = 2 it matches D/(D+2√(D−1)).
- For D = 2 it equals p_c(N) = 1−1/N.

**(b)** One-shot F_N.

**(c)** T_N.

**(d)** Leading MSE (draft Eq. `global_resource_mse`) at C = 10⁴ copies for D = 4 and N = 1, 2, 4, 8, with the optimal N marked. It is drawn only where S·t² ≥ 50.

## Figure H — Local depolarizing: resource trade-off
`PQES_FigH_local_resource_tradeoff.pdf`

**(a, b)** Exact F_N and T_N for |+⟩^⊗M, M = 1, 2, 5, N = 1–16.

**(c)** MSE at C = 10⁴ copies for p = 0.3: leading formula against direct sampling of the raw ratio estimator (2000 repetitions, 95% CI). The optimal N is circled.
- The leading formula is drawn only where S·t² ≥ 50.
- Hollow sampled points have S·t² < 50; there the heavy-tailed raw ratio has an MSE far above any delta-method value.

**(d)** The entangled target cos(π/8)|00⟩ + sin(π/8)|11⟩. Dotted lines mark its dominant-eigenvector mismatch floor.

## Appendix — Ideal channel-purification proxy
`PQES_App_IdealVCP_channel_proxy_ONLY.pdf`

Order-N ideal purification of the depolarizing channel of every gate in the layers that contain a two-qubit gate (the same policy as physical VCP2), with no ancilla, sampling cost or gadget noise. This is a reference curve only and is never ranked against physical circuits.
