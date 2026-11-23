"""Branch-resolved noisy SWAP instrument for circuit-level PQES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from qiskit.circuit.library import CSwapGate, HGate
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from .channels import (
    apply_local_error,
    apply_memory_noise,
    evolve_unitary,
)
from .config import HardwareNoiseSpec


def _symmetrize(operator: np.ndarray) -> np.ndarray:
    return (operator + operator.conj().T) / 2.0


@dataclass(frozen=True)
class MergeDiagnostics:
    """Physical branch and gate counts for one register-level SWAP gadget."""

    branch_probability_plus: float
    branch_probability_minus: float
    h_gates: int
    cswap_gates: int
    memory_locations: int


def _project_and_reduce(
    joint_operator: np.ndarray,
    *,
    M: int,
    outcome: int,
) -> np.ndarray:
    """Project ancilla q0 and retain register A on q1..qM."""
    projector = np.array(
        [[1.0, 0.0], [0.0, 0.0]]
        if outcome == 0
        else [[0.0, 0.0], [0.0, 1.0]],
        dtype=complex,
    )
    projected = DensityMatrix(joint_operator).evolve(
        Operator(projector),
        qargs=[0],
    )
    register_b = list(range(1 + M, 1 + 2 * M))
    reduced = partial_trace(projected, qargs=[0] + register_b)
    return _symmetrize(reduced.data)


def noisy_swap_instrument(
    left: np.ndarray,
    right: np.ndarray,
    spec: HardwareNoiseSpec,
) -> Tuple[np.ndarray, np.ndarray, MergeDiagnostics]:
    """
    Apply a noisy SWAP gadget and return unnormalized true +/- branches.

    Qiskit's subsystem order is q0=ancilla, q1..qM=register A, and
    q(M+1)..q(2M)=register B.  The dense matrix is therefore assembled as
    B tensor A tensor ancilla.
    """
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("left and right must be square operators of equal shape")

    dimension = left.shape[0]
    M = int(round(np.log2(dimension)))
    if 2**M != dimension:
        raise ValueError("Register dimension must be a power of two")

    ancilla = np.diag([1.0 - spec.p_reset, spec.p_reset]).astype(complex)
    joint = np.kron(right, np.kron(left, ancilla))

    # First ancilla Hadamard and its gate error.
    joint = evolve_unitary(joint, HGate(), (0,))
    joint = apply_local_error(
        joint,
        qargs=(0,),
        p=spec.p_1q,
        family=spec.family,
    )

    # Serial native Fredkin gates.  One memory interval is charged after
    # every Fredkin to all live qubits, including the coherent ancilla.
    all_live_qubits = tuple(range(1 + 2 * M))
    for qubit in range(M):
        support = (0, 1 + qubit, 1 + M + qubit)
        joint = evolve_unitary(joint, CSwapGate(), support)
        joint = apply_local_error(
            joint,
            qargs=support,
            p=spec.p_cswap,
            family=spec.family,
        )
        joint = apply_memory_noise(
            joint,
            qargs=all_live_qubits,
            spec=spec,
        )

    # Closing ancilla Hadamard and gate error.
    joint = evolve_unitary(joint, HGate(), (0,))
    joint = apply_local_error(
        joint,
        qargs=(0,),
        p=spec.p_1q,
        family=spec.family,
    )

    plus = _project_and_reduce(joint, M=M, outcome=0)
    minus = _project_and_reduce(joint, M=M, outcome=1)
    diagnostics = MergeDiagnostics(
        branch_probability_plus=float(np.real(np.trace(plus))),
        branch_probability_minus=float(np.real(np.trace(minus))),
        h_gates=2,
        cswap_gates=M,
        memory_locations=M * (1 + 2 * M),
    )
    return plus, minus, diagnostics


def merge_branch_aggregates(
    unsigned: np.ndarray,
    signed: np.ndarray,
    spec: HardwareNoiseSpec,
) -> tuple[np.ndarray, np.ndarray, MergeDiagnostics]:
    """
    Merge identical branch ensembles without enumerating their records.

    ``unsigned`` is the physical branch average. ``signed`` is the
    unnormalized parity-weighted PQES operator.  Linearity gives

        U' = Phi_+(U tensor U) + Phi_-(U tensor U)
        S' = (1-2 p_meas) [Phi_+(S tensor S) - Phi_-(S tensor S)].

    This exactly retains the full SWAP-outcome record at the operator level.
    """
    unsigned_plus, unsigned_minus, diagnostics = noisy_swap_instrument(
        unsigned,
        unsigned,
        spec,
    )
    signed_plus, signed_minus, _ = noisy_swap_instrument(
        signed,
        signed,
        spec,
    )

    unsigned_out = _symmetrize(unsigned_plus + unsigned_minus)
    measurement_visibility = 1.0 - 2.0 * spec.p_measure
    signed_out = _symmetrize(
        measurement_visibility * (signed_plus - signed_minus)
    )
    return unsigned_out, signed_out, diagnostics


__all__ = [
    "MergeDiagnostics",
    "merge_branch_aggregates",
    "noisy_swap_instrument",
]
