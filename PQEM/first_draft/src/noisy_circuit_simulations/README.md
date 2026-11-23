# Noisy circuit simulations

This package is the circuit-level counterpart of
`src/simulation/rho2_sims`. New code stays here; the older simulation
directories are not modified.

## What is modeled

The simulator evolves two operators:

- `U`, the ordinary physical average over all SWAP-test branches.
- `S`, the unnormalized signed PQES operator obtained by weighting the
  complete SWAP-outcome record by its parity.

For an ideal SWAP gadget and identical inputs, the implementation satisfies

```text
U_out = rho
S_out = rho^2
```

to numerical precision. The mitigated state used for fidelity and observable
estimation is `S / Tr(S)`, and the leading sampling-overhead diagnostic is
`1 / |Tr(S)|^2`. The implementation rescales `S` between purification
levels and carries its absolute magnitude in log space, so deep trees remain
numerically stable even when their estimator sampling cost is enormous.

The hardware model includes independently configurable:

- imperfect data and ancilla reset;
- one-qubit gate errors;
- two-qubit workload-gate errors;
- native Fredkin/controlled-SWAP errors;
- memory noise during circuit blocks and SWAP gadgets;
- ancilla readout errors.

Every ideal workload gate is executed explicitly. Depolarizing noise uses a
local Qiskit quantum channel on the gate support. Dephasing applies an
independent phase-flip channel to each qubit in that support.

The initial state is prepared by explicit Hadamard gates. Available workload
blocks are:

- `brickwork`: reproducible random `RY`, `RZ`, and alternating CNOT layers;
- `echo`: nontrivial gate pairs whose ideal action is the identity;
- `idle`: memory-noise-only control.

The compared matched-depth schedules are:

- `interleaved`: apply `ell` SWAP/PQES levels after every circuit block;
- `end_only`: apply the same total `cycles * ell` levels after the circuit;
- `none`: an unmitigated baseline.

## Output organization

```text
src/noisy_circuit_simulations/       code and tests
data/noisy_circuit_simulations/      CSV data tables and manifest.json
figures/noisy_circuit_simulations/   PDF and PNG figures
```

The `steps.csv` and `finals.csv` tables retain the important legacy
`rho2_sims` columns and add schedule, circuit, hardware-location,
physical-branch, signed-normalization, sampling-overhead, and resource fields.
Both ordinary and log-space signed-trace/overhead columns are included; use
the log-space columns for deep purification trees where floating-point
sampling overhead can overflow.

`schedule_comparison.csv` contains one row per matched interleaved/end-only
pair, including fidelity differences, gains over the unmitigated baseline,
sampling-overhead differences, physical gate counts, and the schedule with
higher final fidelity.

## Running

Activate the repository's `swap_qec` environment first.

Quick validation grid:

```bash
python -m src.noisy_circuit_simulations.grid_run --quick
```

Full default grid:

```bash
python -m src.noisy_circuit_simulations.grid_run
```

Generate all figures:

```bash
python -m src.noisy_circuit_simulations.plots
```

Rebuild only the matched-schedule comparison table from an existing
`finals.csv`:

```bash
python -m src.noisy_circuit_simulations.analysis
```

Run model tests:

```bash
python -m pytest src/noisy_circuit_simulations/test_models.py
```

## Error-rate convention

The swept `p` is the one-qubit location error. By default, all other
locations use the same probability. Multipliers permit a hardware-weighted
model:

```bash
python -m src.noisy_circuit_simulations.grid_run \
  --p2-multiplier 5 \
  --cswap-multiplier 10 \
  --memory-multiplier 0.2 \
  --measure-multiplier 2
```

All realized rates are stored explicitly in the CSV files.

## Scaling limit

One noisy SWAP instrument uses `1 + 2M` qubits and an exact dense
density-operator calculation. Grid runs therefore enforce `M <= 4`.
The default full sweep stops at `M=3`; larger-register studies should use
tensor-network or trajectory methods.
