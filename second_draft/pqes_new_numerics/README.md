# PQES: noisy-implementation numerical suite

This folder holds the numerics for the revised manuscript *Recursive SWAP-Tree Virtual Distillation for Quantum Error Suppression*.

The suite simulates recursive SWAP-tree virtual distillation (PQES) and its baselines as explicit noisy quantum instruments:
- compiled two-qubit gates;
- one shared clock for every architecture;
- exact density matrices.

It reports estimator bias, variance, shot counts and resources at a fixed target RMSE. It never asserts an architecture advantage; every result holds only within the stated toy hardware model.

The code is self-contained and does not import the first-draft simulators.

## Quick start

Requires Python 3.10+ with NumPy, SciPy, pandas, Matplotlib and pytest. Qiskit is not needed.

```bash
python -m pip install -r pqes_new_numerics/requirements.txt
python -m pytest -q pqes_new_numerics/tests/        # ~2 s
python pqes_new_numerics/run_simulations.py --profile quick      # ~6 s, smoke test
python pqes_new_numerics/run_simulations.py --profile standard   # ~2 min
python pqes_new_numerics/run_simulations.py --profile full --seed 31 --rmse 0.03
python pqes_new_numerics/plot_results.py
```

The committed data in `../data/pqes_new_numerics/` come from the `full` profile, with the arguments shown above. `manifest.json` records:
- the profile and per-task runtimes;
- the seed;
- the reference hardware;
- every convention.

The full profile runs in about 10 minutes (≈590 s of compute) on one laptop core (per-task times are in the manifest). A single task can be rerun alone, e.g. `--task placement`; the manifest then merges that task's profile and runtime.

The committed data were generated with NumPy 2.5.4 (x86_64 wheel). Seeds are fixed per key, but NumPy does not guarantee identical `Generator` streams across versions, and the LAPACK build sets the eigenbasis in which degenerate observable outcomes are sampled. Under another NumPy version or build, every sampled quantity changes within its sampling error: the Monte Carlo RMSE and its CI, `mc_*`, `empirical`, `mse_sampled_*`, and the `fixed_K_empirical` rows of `twirling.csv`. Deterministic columns agree to round-off, about 1e-13 relative across NumPy versions; a shot count that sits exactly on the S·t² ≥ 50 guard can round up by one.

## Files

| File | Purpose |
|---|---|
| `core.py` | Hardware model, compiled CSWAP gadget, shared clock, the six architectures, estimator moments, shot planning, Monte Carlo helpers |
| `circuits.py` | Algorithm families (entangling and product), ASAP layering, noisy layers, fixed-depth cuts, ideal channel-purification proxy |
| `vcp_physical.py` | Physical order-two virtual channel purification (Liu et al.), per layer on the gate support |
| `run_simulations.py` | One function per study, one CSV per study; its docstring is the methods section |
| `plot_results.py` | Figures from the CSVs only |
| `tests/oracle.py` | Independent brute-force simulator: explicit Kraus sums, all registers kept, all outcome strings enumerated |
| `tests/test_validation.py`, `tests/test_studies.py` | Ideal identities, oracle comparisons, hand-derived clock checks, one regression test per audit defect, and study formulas against brute force |
| `RESULTS_OVERVIEW.md` | What the full-profile data show, favourable and unfavourable |
| `FIGURE_CAPTIONS.md` | Draft captions describing exactly what each figure plots |

## What is simulated

**Instrument.** N independent copies of ρ enter an architecture. The output is the unnormalized surviving-register operators grouped by *reported* parity, τ₊ and τ₋, with R = τ₊+τ₋ and W = τ₊−τ₋.
- Ideally R = ρ and W = ρᴺ for every architecture; this is checked to machine precision.
- The estimator is the ratio of signed sample means.
- The suite records:
  - its limit μ_A = Tr(OW)/Tr W;
  - its delta-method variance v_A = [m₂ − 2μm₁ + μ²]/t²;
  - its leading ratio bias c_A = (μ − m₁)/t².

  Here m₁ and m₂ are the unsigned moments of the measured outcome.

**Architectures** (`core.ARCH`):

| Arm | Registers | Controls | Notes |
|---|---|---|---|
| `tree_parallel` | N | N/2 reusable | balanced tree, levels in parallel |
| `tree_recycled` | ℓ+1 | 1 reusable | depth-first balanced tree |
| `tree_sequential` | 2 | 1 reusable | measured accumulator |
| `coherent_request` | 2 | 1 coherent | streaming controlled-cyclic VD; B is reset and reprepared |
| `wide_cyclic` | N | 1 coherent | all copies present, no resets |
| `derangement` | N | 1 coherent | wide_cyclic circuit; O measured on all N registers and averaged; each spent register read out mid-circuit as soon as its CSWAP ends |

The `derangement` arm is Koczor's cyclic-derangement estimator. Each spent register is measured right after its CSWAP, charged the mid-circuit measurement duration `t_meas`, so it never idles until the final readout. It is simulated densely and only for 1+MN ≤ 9.

The three `tree_*` arms are PQES. The coherent arm is ideal-equivalent to a controlled N-cycle; a test checks this against the explicit permutation unitary.

**Gate noise (one convention).** Every gate is followed by the replacement depolarizing channel D_ε(X) = (1−ε)X + ε I/d ⊗ Tr X on its wires (draft Eq. `noisy_entangling_channel`):
- ε₁ after one-qubit gates;
- ε₂ after two-qubit gates;
- ε₃ (default ε₂) after Toffolis.

For one qubit, the Pauli-form probability is 3ε/4.

Each register controlled SWAP is compiled as:
1. a CNOT(b→a) layer;
2. M Toffolis CCX(c,a→b) sharing the control;
3. a CNOT layer.

That is M+2 time units.

Abstract source states use Pauli-form local depolarizing `source_pauli_p` (manuscript Eq. `pauli-kraus`). Algorithm circuits use the gate noise above.

Wherever a bare `p` appears, a `noise_convention` column names its channel:
- `pauli_form`: (1−p)ρ + p/3 ΣPρP (local depolarizing);
- `replacement_form`: (1−p)ρ + p I/D (global depolarizing);
- `z_flip`: (1−p)ρ + pZρZ (dephasing).

In `channel_physical.csv`, unprefixed hardware columns record the algorithm and source hardware; `app_*` columns record the apparatus actually used.

**The clock (shared by every arm).**
- **Durations:** a gate takes 1 unit; a source preparation takes `t_prep` units (or the source circuit's depth); mid-circuit measurement takes `t_meas` and reset `t_reset`.
- **Just-in-time preparation:** preparations are scheduled just in time, so a fresh copy never idles. A merged survivor that must wait does idle.
- **No feed-forward:** a register never waits for its own ancilla's measurement. Only ancilla reuse (measure, reset, H) blocks the next gadget.
- **Idle noise:** every waiting qubit, data or control, inside or between gadgets, receives amplitude damping `idle_gamma` and a Z flip `idle_phi` per unit.
- **Resets:** first use of any register is perfect. Every actual mid-circuit reuse pays `t_reset` and the reset error:
  - `data_reset_p`: the fresh copy is replaced by |0…0⟩ (phenomenological);
  - `ancilla_reset_p`: |+⟩ becomes |−⟩.
- **Readout:** each recorded ancilla outcome flips independently with probability `readout_q`.
- **Per-run ledger:** duration, idle exposure of data and control, preparations, resets, measurements, CNOTs, Toffolis, one-qubit gates and peak width. When the source is an algorithm circuit (placement, channel purification), every arm's ledger includes that circuit's gates and idle units on every copy that runs it, and the downstream block on every register that runs it.

**Reference hardware.** These values are generic and dimensionless; they are not a device calibration:
- ε₂ ∈ {0.002, 0.005, 0.01, 0.02}, with ε₁ = ε₂/10 and ε₃ = ε₂;
- q = 0.01, reset errors 0.005;
- idle rates 5×10⁻⁴ per unit;
- `t_prep` = 3.

There are two latency profiles: **fast** (`t_meas` = `t_reset` = 1) and **slow** (`t_meas` = 10, `t_reset` = 5). Conclusions are stated per profile and per regime. The phase map puts the readout error and the idle rate on its axes.

**Shot planning.** The target RMSE is ε = 0.03.
- The shot count is S = ⌈(v+2bc)/(ε²−b²)⌉, and a point is infeasible if b² ≥ ε².
- S·t² ≥ 50 is enforced, because below it the delta method fails.
- At the planned S, a direct Monte Carlo of the raw ratio estimator reports the achieved RMSE with a 95% CI.
- Totals are reported as a vector, never a single scalar: shots, copies, two-qubit gates, Toffolis, duration and space-time volume (width × duration).

**Channel purification (VCP2).** This follows Liu et al., PRX Quantum 6, 020325. Each purified layer gets one gadget:
1. a |+⟩ control and an I/2ᵏ auxiliary on the layer's gate support (k qubits);
2. a compiled CSWAP;
3. the noisy layer applied to the main register **and** to the auxiliary;
4. a second compiled CSWAP;
5. an X readout.

**Purification policy:** only layers that contain a two-qubit gate are purified, because Liu-style VCP targets the noisy entangling channels. One-qubit-only layers run unpurified with their ε₁ = ε₂/10 noise; purifying them would add far more gadget error than it removes.
- The M = 2 entangling study circuit has 4 such layers out of 12.
- The M = 2 product circuit has none of its 8, and the M = 1 circuit none of its 6, so on those circuits VCP2 reduces to the unmitigated circuit.

Each purified layer gets its own freshly prepared control (with its auxiliary). These are drawn round robin from the smallest pool of physical control–auxiliary pairs that keeps measurement, reset and re-preparation off the main register's critical path: for the entangling study circuit, 2 pairs with fast measurement and 3 with slow.
- The main register never waits on a previous layer's control.
- The reported width counts every physical control and auxiliary qubit.
- The auxiliary is prepared by a random Pauli and needs no reset.

Resources counted include 2G_purified + G_unpurified noisy-gate executions, plus one auxiliary preparation, one control preparation and one measurement per purified layer.

With an ideal apparatus and no idle noise, VCP2 equals ideal channel purification to 10⁻¹²; a test checks this. With idle noise, the algorithm's idle errors on qubits outside a layer are not purified. "VCP2 at a cut" purifies the layers before the cut only.

## Studies (`run_simulations.py --task …`) and outputs

| Task / CSV | Figure | Content |
|---|---|---|
| `validation` | — | ideal R = ρ, W = ρᴺ for all six arms, M = 1, 2 and N = 2, 4, 8 |
| `readout`, `shots` | Fig A | readout visibility (1−2q)^k and the (1−2q)^(−2(N−2)) penalty; estimator Monte Carlo along q, with χ² bands and the ratio bias |
| `mechanisms` | Fig A2 | each mechanism swept from zero: ε₂, ε₁, q, the reset errors, idle φ and γ, t_meas, t_prep |
| `phase` | Figs B, B2, B3 | lowest-cost arm at RMSE 0.03 for four cost metrics with no width constraint (primary result); winners at explicit width budgets 5, 7, 9 and unconstrained; Monte Carlo validation of the planning |
| `placement` | Fig C | fixed total depth of the study circuit (`circuits.study_layers`: L = 12 entangling, L = 8 product, both divisible by 4), cut after exactly L_A/L ∈ {0, ¼, ½, ¾, 1}; every arm plus VCP2 at the same cut |
| `channel_physical` | Fig F | full layerwise VCP2 vs output-only state VD, under ideal, 10×-better and physical apparatus |
| `channel_proxy` | App. | ideal channel purification with no apparatus (never ranked) |
| `twirling`, `twirl_mc`, `twirl_budget` | Fig D | fixed K-frame sets (K = 1, 3, …, 243); fresh random frames per copy and per shot; noisy Cliffords; fixed-budget error before and after twirling |
| `cycles` | Fig E | repeated virtual cycles (local and global, N = 2, 4, 8): F_t, Z_t, and the copy budget N^t·S for RMSE 0.03 |
| `global_depol`, `global_boundaries` | Fig G | exact one-shot F_N, T_N and MSE; repeated fixed points; p_c(N) = 1−1/N; numerical p_SN(D,N) |
| `local_resource` | Fig H | F_N, T_N, leading and sampled MSE at copy budgets 10³, 10⁴, 10⁵; the optimal N; the entangled mismatch floor |

## Assumptions and limitations

1. **Gate model:**
   - one compiled gate set;
   - Toffolis are native three-qubit gates with ε₃ = ε₂;
   - there is no connectivity or routing model;
   - all gates take one unit.
2. **Observable readouts are ideal in all arms.** The observable is the known ideal target projector, measured perfectly on every measured register, including the derangement arm's mid-circuit readouts; its uncomputation cost is not included. Readout error `readout_q` acts only on recorded sign and control bits (tree ancillas, the coherent control, VCP2 controls).
3. **Data reset error** is phenomenological (|0…0⟩ substitution).
4. **VCP2 auxiliary:**
   - the I/2ᵏ auxiliary is prepared ideally;
   - VCP2 is order two only;
   - idle amplitude damping makes the VCP2 identity approximate (it is simulated exactly regardless).
5. **No hardware calibration:** results hold for the stated generic parameters. The figures show how conclusions move with the latency profile, ε₂, the readout error and the idle rate.
6. **Twirl Monte Carlo** uses an ideal SWAP instrument and noisy Clifford frames. Product targets factorize per qubit, so M = 5 is exact.
7. **Scope:** no QEC memory, no decoding, no non-Markovian or correlated noise, and no coherent over-rotations.

## Tests

```bash
python -m pytest -q pqes_new_numerics/tests/
```

The suite covers:
- **ideal identities** for every arm;
- **oracle comparisons** under all noise mechanisms with non-trivial timings, for:
  - sequential trees, N = 2–4, M = 1, 2;
  - parallel and recycled trees, N = 4;
  - coherent and wide arms, N = 2–4;
  - the derangement arm (spent registers read out at once, ideally), N = 2, 3;
- **clock tests:** hand-derived durations and idle units;
- **reset counts** per actual reuse;
- **regressions** for each defect the October 2026 audit found:
  - recycled-tree clock artifact;
  - reset counts;
  - real-dtype crash;
  - N = 1 readout;
  - VCP accounting;
  - seed nesting across shot counts;
- **regressions** for the review fixes:
  - spent derangement registers read out at once (no idle until the final readout), with ideal observable readouts;
  - VCP2 purifying only layers with a two-qubit gate;
  - one fresh control per VCP2 layer, with the main register never waiting;
  - placement cut fractions realized exactly;
  - repeated-cycle copy budgets undefined when the bias exceeds the target;
  - apparatus columns recording the hardware actually used;
  - leading MSE undefined outside the delta-method regime;
- **noise structure:** compiled two-qubit noise on the data pair biasing the estimator and separating the trees;
- **estimator checks:** delta-method and derangement moments against sampling, and shot planning reaching the target RMSE;
- **study formulas** against brute force.
