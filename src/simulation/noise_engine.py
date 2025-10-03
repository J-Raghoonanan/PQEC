"""
Noise engine for the SWAP-based purification simulator (Qiskit).

This module builds *noisy input copies* ρ from a given target preparation
circuit U_psi. Two modes are supported:

(A) iid_p      — Apply a CPTP channel independently to each qubit with
                 probability p (maps manuscript's δ to p via configs).
(B) exact_k    — Deterministically inject exactly k single-qubit Pauli faults
                 (Z/X for dephasing, uniform {X,Y,Z} for depolarizing).

CRITICAL: We use ONLY explicit Kraus operators to ensure consistency with the
manuscript's noise model definitions. We never use Qiskit's DepolarizingChannel
as its parameter convention may differ across versions.

Returned objects are *circuits on M data qubits* that prepare the noisy state
from |0...0>.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus

from .configs import NoiseMode, NoiseSpec, NoiseType

logger = logging.getLogger(__name__)

# -----------------------------
# Explicit Kraus operators
# -----------------------------

def _kraus_depolarizing(p: float) -> Kraus:
    """Depolarizing channel with Kraus form:
    E0 = sqrt(1-p) I, Ej = sqrt(p/3) σj for j ∈ {X,Y,Z}
    
    This gives: ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    
    Manuscript relation: δ = 4p/3, so p = 3δ/4.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    Ex = np.sqrt(p / 3.0) * X
    Ey = np.sqrt(p / 3.0) * Y
    Ez = np.sqrt(p / 3.0) * Z
    
    logger.debug(f"Depolarizing channel: p={p:.4f}, δ_equiv={4*p/3:.4f}")
    return Kraus([E0, Ex, Ey, Ez])


def _kraus_z_dephase(p: float) -> Kraus:
    """Pure Z-dephasing channel:
    E0 = sqrt(1-p) I, E1 = sqrt(p) Z
    
    This gives: ρ → (1-p)ρ + p ZρZ
    
    For manuscript: δ = p (we take δ directly as the dephasing probability).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Z-dephasing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    E1 = np.sqrt(p) * Z
    
    logger.debug(f"Z-dephasing channel: p={p:.4f}")
    return Kraus([E0, E1])


def _kraus_x_dephase(p: float) -> Kraus:
    """Pure X-dephasing channel:
    E0 = sqrt(1-p) I, E1 = sqrt(p) X
    
    This gives: ρ → (1-p)ρ + p XρX
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"X-dephasing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    E1 = np.sqrt(p) * X
    
    logger.debug(f"X-dephasing channel: p={p:.4f}")
    return Kraus([E0, E1])


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
    
    NOTE: This generates a SINGLE pattern that should be shared between
    both copies entering the SWAP test to ensure they are identical.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return tuple()
    if k > M:
        raise ValueError(f"k={k} cannot exceed M={M} for single-qubit faults")

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
    
    logger.debug(f"Sampled error pattern (k={k}): {[(e.qubit, e.pauli) for e in ops]}")
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

def build_copy_iid_p(prep: QuantumCircuit, noise: NoiseSpec) -> QuantumCircuit:
    """Build a noisy copy by applying i.i.d. CPTP channels to each qubit.
    
    CRITICAL: Uses only explicit Kraus operators, never Qiskit's 
    DepolarizingChannel to ensure consistency with manuscript definitions.
    """
    M = prep.num_qubits
    p = noise.kraus_p()
    qc = prep.copy(name=f"noisy_{noise.noise_type.value}_iid")

    logger.debug(f"Building iid_p copy: M={M}, noise={noise.noise_type.value}, p={p:.4f}")

    if noise.noise_type == NoiseType.depolarizing:
        chan_instr = _kraus_depolarizing(p).to_instruction()
    elif noise.noise_type == NoiseType.dephase_z:
        chan_instr = _kraus_z_dephase(p).to_instruction()
    elif noise.noise_type == NoiseType.dephase_x:
        chan_instr = _kraus_x_dephase(p).to_instruction()
    else:
        raise ValueError(f"Unsupported noise type: {noise.noise_type}")

    # Apply channel to each qubit independently
    for q in range(M):
        qc.append(chan_instr, [q])

    return qc


def build_copy_exact_k(prep: QuantumCircuit, pattern: ErrorPattern) -> QuantumCircuit:
    """Return a circuit that prepares |ψ⟩ and then injects the given pattern.

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
        Preparation circuit for |ψ⟩ on M qubits.
    noise : NoiseSpec
        Noise configuration (type, mode, delta, exact_k, etc.).
    seed : Optional[int]
        RNG seed for sampling patterns (exact_k) when not provided.
    shared_pattern : Optional[ErrorPattern]
        If provided (and mode == exact_k), this pattern is used instead of
        sampling — CRITICAL for ensuring identical copies in SWAP test.

    Returns
    -------
    (qc, pattern)
        qc: noisy-copy circuit on M qubits starting from |0...0⟩.
        pattern: the ErrorPattern used (None for iid_p mode).
    """
    if noise.mode == NoiseMode.iid_p:
        logger.debug("Building iid_p noisy copy")
        return build_copy_iid_p(prep, noise), None

    # exact_k mode
    if shared_pattern is not None:
        logger.debug(f"Using shared pattern with {len(shared_pattern)} errors")
        pattern = shared_pattern
    else:
        logger.debug(f"Sampling new error pattern (k={noise.exact_k})")
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