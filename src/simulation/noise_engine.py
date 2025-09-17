"""
Noise engine for the SWAP-based purification simulator (Qiskit).

This module builds *noisy input copies* rho from a given target preparation
circuit U_psi. Two modes are supported:

(A) iid_p      — Apply a CPTP channel independently to each qubit with
                 probability p (maps manuscript's δ to p via configs).
(B) exact_k    — Deterministically inject exactly k single-qubit Pauli faults
                 (Z/X for dephasing, uniform {X,Y,Z} for depolarizing).

Returned objects are *circuits on M data qubits* that prepare the noisy state
from |0...0>. The caller can compose these into larger circuits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus
# from qiskit.quantum_info.operators.channel import DepolarizingChannel

from .configs import NoiseMode, NoiseSpec, NoiseType, delta_to_kraus_p

def _kraus_depolarizing(p: float) -> Kraus:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Ks = [np.sqrt(1.0 - p) * I, np.sqrt(p / 3) * X, np.sqrt(p / 3) * Y, np.sqrt(p / 3) * Z]
    return Kraus(Ks)

# -----------------------------
# Error pattern (for exact_k)
# -----------------------------
@dataclass(frozen=True)
class ErrorOp:
    qubit: int
    pauli: str  # one of {"X","Y","Z"}


ErrorPattern = Tuple[ErrorOp, ...]


def sample_error_pattern(
    M: int,
    noise_type: NoiseType,
    k: int,
    seed: Optional[int] = None,
) -> ErrorPattern:
    """Sample a deterministic pattern of exactly k single-qubit faults.

    For dephase_z: only Z faults
    For dephase_x: only X faults
    For depolarizing: uniform over {X,Y,Z}
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return tuple()
    if k > M:
        raise ValueError("k cannot exceed M for single-qubit faults")

    rng = np.random.default_rng(seed)
    qubits = rng.choice(M, size=k, replace=False)
    ops: List[ErrorOp] = []
    for q in qubits:
        if noise_type == NoiseType.dephase_z:
            ops.append(ErrorOp(int(q), "Z"))
        elif noise_type == NoiseType.dephase_x:
            ops.append(ErrorOp(int(q), "X"))
        else:  # depolarizing
            pauli = rng.choice(["X", "Y", "Z"])  # uniform
            ops.append(ErrorOp(int(q), str(pauli)))
    # sort by qubit index for determinism
    ops.sort(key=lambda e: e.qubit)
    return tuple(ops)


def apply_error_pattern(qc: QuantumCircuit, pattern: ErrorPattern) -> None:
    """Append the specified single-qubit Pauli gates to the circuit."""
    for op in pattern:
        if op.pauli == "X":
            qc.x(op.qubit)
        elif op.pauli == "Y":
            qc.y(op.qubit)
        elif op.pauli == "Z":
            qc.z(op.qubit)
        else:
            raise ValueError(f"Unknown Pauli '{op.pauli}' in pattern")


# -----------------------------
# IID CPTP channels per qubit
# -----------------------------

def _kraus_z_dephase(p: float) -> Kraus:
    # Kraus operators: sqrt(1-p) I, sqrt(p) Z  =>  rho -> (1-p)rho + p Z rho Z
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Ks = [np.sqrt(1.0 - p) * I, np.sqrt(p) * Z]
    return Kraus(Ks)


def _kraus_x_dephase(p: float) -> Kraus:
    # Kraus operators: sqrt(1-p) I, sqrt(p) X  =>  rho -> (1-p)rho + p X rho X
    import numpy as np

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Ks = [np.sqrt(1.0 - p) * I, np.sqrt(p) * X]
    return Kraus(Ks)


def _append_channel_per_qubit(qc: QuantumCircuit, chan_instr, M: int) -> None:
    for q in range(M):
        qc.append(chan_instr, [q])


def build_copy_iid_p(prep: QuantumCircuit, noise: NoiseSpec) -> QuantumCircuit:
    M = prep.num_qubits
    p = noise.kraus_p()
    qc = prep.copy(name=f"noisy_{noise.noise_type.value}_iid")

    if noise.noise_type == NoiseType.depolarizing:
        try:
            # Works on some Qiskit versions
            from qiskit.quantum_info import DepolarizingChannel  # local import
            chan_instr = DepolarizingChannel(p).to_instruction()
        except Exception:
            # Robust fallback: exact same channel via Kraus ops
            chan_instr = _kraus_depolarizing(p).to_instruction()
        for q in range(M):
            qc.append(chan_instr, [q])

    elif noise.noise_type == NoiseType.dephase_z:
        chan_instr = _kraus_z_dephase(p).to_instruction()
        for q in range(M):
            qc.append(chan_instr, [q])

    elif noise.noise_type == NoiseType.dephase_x:
        chan_instr = _kraus_x_dephase(p).to_instruction()
        for q in range(M):
            qc.append(chan_instr, [q])

    else:
        raise ValueError(f"Unsupported noise type: {noise.noise_type}")

    return qc



def build_copy_exact_k(prep: QuantumCircuit, pattern: ErrorPattern) -> QuantumCircuit:
    """Return a circuit that prepares |psi> and then injects the given pattern for k qubits.

    The pattern specifies *deterministic* Pauli faults to apply after preparation.
    """
    qc = prep.copy(name="noisy_exact_k")
    apply_error_pattern(qc, pattern)
    return qc


def build_noisy_copy(
    prep: QuantumCircuit,
    noise: NoiseSpec,
    seed: Optional[int] = None,
    shared_pattern: Optional[ErrorPattern] = None,
) -> Tuple[QuantumCircuit, Optional[ErrorPattern]]:
    """Factory that returns a noisy-copy circuit and the pattern used (if any).

    Parameters
    ----------
    prep : QuantumCircuit
        Preparation circuit for |psi> on M qubits.
    noise : NoiseSpec
        Noise configuration (type, mode, delta, exact_k, etc.).
    seed : Optional[int]
        RNG seed for sampling patterns (exact_k) when not provided.
    shared_pattern : Optional[ErrorPattern]
        If provided (and mode == exact_k), this pattern is used instead of
        sampling — useful for 'identical_pattern=True' across two copies.

    Returns
    -------
    (qc, pattern)
        qc: noisy-copy circuit on M qubits starting from |0...0>.
        pattern: the ErrorPattern used (None for iid_p mode).
    """
    if noise.mode == NoiseMode.iid_p:
        return build_copy_iid_p(prep, noise), None

    # exact_k mode
    if shared_pattern is not None:
        pattern = shared_pattern
    else:
        pattern = sample_error_pattern(
            M=prep.num_qubits,
            noise_type=noise.noise_type,
            k=noise.exact_k,
            seed=seed,
        )
    return build_copy_exact_k(prep, pattern), pattern


__all__ = [
    "ErrorOp",
    "ErrorPattern",
    "sample_error_pattern",
    "apply_error_pattern",
    "build_copy_iid_p",
    "build_copy_exact_k",
    "build_noisy_copy",
]
