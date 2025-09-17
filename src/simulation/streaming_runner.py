"""
Streaming (O(log N)) purification runner using Qiskit circuits.

This module ties together:
  - target state preparation (state_factory.build_target),
  - noisy copy generation (noise_engine.build_noisy_copy), and
  - two-copy purification (amplified_swap.purify_two_from_density),
then logs per-merge metrics and a final summary compatible with your figure
scripts.

Notes on scalability
--------------------
The purification merge operates on 1+2M qubits with a *density matrix* backend
(needed for CPTP channels). Density-matrix size grows as 4^{(1+2M)}; in
practice, M ≲ 5-6 is comfortable on a laptop. For larger M (e.g., 8-10), a
Monte Carlo trajectory mode is advisable (TODO: add in follow-up).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector

# Aer simulator import compatible across Qiskit versions
try:  # qiskit-aer >= 0.12
    from qiskit_aer import AerSimulator
except Exception:  # fallback for older installs
    from qiskit.providers.aer import AerSimulator  # type: ignore

from .configs import RunSpec, NoiseMode
from .state_factory import build_target
from src.simulation.noise_engine import build_noisy_copy, sample_error_pattern
from src.simulation.amplified_swap import purify_two_from_density


# -----------------------------
# Backend helpers
# -----------------------------

def _make_backend(method: str):
    return AerSimulator(method=method)


def _density_from_circuit(qc, backend) -> DensityMatrix:
    qc2 = qc.copy()
    qc2.save_density_matrix()
    t = transpile(qc2, backend)
    res = backend.run(t, shots=1024).result()
    data0 = res.data(0)
    # 'density_matrix' key holds the complex array in Aer
    rho = data0.get("density_matrix")
    if rho is None:
        # older versions might use 'statevector' if method not honored
        rho = data0.get("statevector")
    return DensityMatrix(rho)


# -----------------------------
# Metric helpers
# -----------------------------

def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    v = psi.data.reshape((-1, 1))
    return float(np.real(np.conj(v).T @ (rho.data @ v)))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    proj = np.outer(psi.data, np.conj(psi.data))
    diff = rho.data - proj
    # Hermitian; trace norm is sum of absolute eigenvalues
    evals = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def _purity(rho: DensityMatrix) -> float:
    return float(np.real(np.trace(rho.data @ rho.data)))


# -----------------------------
# Runner
# -----------------------------

def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a streaming purification experiment per 'spec'.

    Returns
    -------
    (steps, finals):
        steps  — one row per merge (purification step), with depth, metrics, etc.
        finals — one row summary for the run (final eps_L, reduction ratio, ...).
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backend
    backend = _make_backend(spec.backend_method)

    # Target |psi> and its prep circuit
    prep, psi = build_target(spec.target)
    M = spec.target.M

    # Build a prototype noisy copy circuit (iid_p can reuse it)
    iid_cached_dm: Optional[DensityMatrix] = None
    if spec.noise.mode == NoiseMode.iid_p:
        qc_noisy, _ = build_noisy_copy(prep, spec.noise)
        iid_cached_dm = _density_from_circuit(qc_noisy, backend)

    # Initial (noisy) single-register density for baseline metrics
    rho_init = iid_cached_dm
    if rho_init is None:
        # Build one sample copy for baseline in exact_k mode as well
        qc_copy, _ = build_noisy_copy(prep, spec.noise, seed=spec.target.seed)
        rho_init = _density_from_circuit(qc_copy, backend)

    # baseline metrics before any purification
    F_init = _fidelity_to_pure(rho_init, psi)
    eps_init = _trace_distance_to_pure(rho_init, psi)

    # Streaming stacks: one slot per depth (number of inputs represented = 2^depth)
    slots: Dict[int, DensityMatrix] = {}
    counts: Dict[int, int] = {}  # number of raw inputs represented at each slot

    steps_rows: List[Dict] = []

    def _log_step(depth: int, rho_out: DensityMatrix, meta: Dict):
        row = {
            "run_id": spec.synthesize_run_id(),
            "M": M,
            "depth": depth,
            "copies_used": 2 ** depth,
            "N_so_far": sum(counts.values()),
            "noise": spec.noise.noise_type.value,
            "mode": spec.noise.mode.value,
            "delta": spec.noise.delta,
            "p_channel": spec.noise.kraus_p(),
            "P_success": meta.get("P_success"),
            "grover_iters": meta.get("grover_iters"),
            "fidelity": _fidelity_to_pure(rho_out, psi),
            "eps_L": _trace_distance_to_pure(rho_out, psi),
            "purity": _purity(rho_out),
        }
        steps_rows.append(row)

    def _get_new_copy_dm(i: int) -> DensityMatrix:
        # iid_p: reuse cached
        if iid_cached_dm is not None:
            return iid_cached_dm
        # exact_k: build per-copy (pattern can be shared by pair at merge-time; here we sample per copy)
        qc_copy, _ = build_noisy_copy(prep, spec.noise, seed=(spec.target.seed or 0) + i)
        return _density_from_circuit(qc_copy, backend)

    # Process N incoming copies
    for i in range(spec.N):
        rho_new = _get_new_copy_dm(i)
        level = 0
        carry_dm = rho_new
        carry_count = 1
        while True:
            if level not in slots:
                slots[level] = carry_dm
                counts[level] = carry_count
                break
            else:
                # Merge with existing slot at this level
                left = slots.pop(level)
                left_count = counts.pop(level)
                # Purify two (identical interface accepts both copies if extended in future)
                rho_out, meta = purify_two_from_density(left, carry_dm, spec.aa)  # identical copies interface
                # NOTE: The above purifies two identical copies 'left' with itself.
                # To merge two *different* copies, replace with: purify_two(left, carry_dm, spec.aa)
                _log_step(depth=level + 1, rho_out=rho_out, meta=meta)
                carry_dm = rho_out
                carry_count = left_count + carry_count
                level += 1

    # Final output = deepest slot
    if not counts:
        raise ValueError("No data was processed; N must be >= 1")
    max_level = max(counts.keys())
    rho_final = slots[max_level]

    F_final = _fidelity_to_pure(rho_final, psi)
    eps_final = _trace_distance_to_pure(rho_final, psi)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    finals_row = {
        "run_id": spec.synthesize_run_id(),
        "M": M,
        "N": spec.N,
        "noise": spec.noise.noise_type.value,
        "mode": spec.noise.mode.value,
        "delta": spec.noise.delta,
        "p_channel": spec.noise.kraus_p(),
        "fidelity_init": F_init,
        "fidelity_final": F_final,
        "eps_L_init": eps_init,
        "eps_L_final": eps_final,
        "error_reduction_ratio": reduction,
        "max_depth": int(np.log2(counts[max_level])),
    }

    steps_df = pd.DataFrame(steps_rows)
    finals_df = pd.DataFrame([finals_row])
    return steps_df, finals_df


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Run a single spec and append results to CSVs under spec.out_dir.

    Returns the (steps_path, finals_path).
    """
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_path = out_dir / "steps_circuit.csv"
    finals_path = out_dir / "finals_circuit.csv"

    # Append or create
    if steps_path.exists():
        prev = pd.read_csv(steps_path)
        steps_df = pd.concat([prev, steps_df], ignore_index=True)
    if finals_path.exists():
        prev = pd.read_csv(finals_path)
        finals_df = pd.concat([prev, finals_df], ignore_index=True)

    steps_df.to_csv(steps_path, index=False)
    finals_df.to_csv(finals_path, index=False)

    print(f"Saved {steps_path} and {finals_path}")
    return steps_path, finals_path


__all__ = ["run_streaming", "run_and_save"]
