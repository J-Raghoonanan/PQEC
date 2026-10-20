"""Exact density-operator evolution with gate-local hardware noise."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer.noise import depolarizing_error, pauli_error

from .config import HardwareNoiseSpec, NoiseFamily


def _symmetrize(operator: np.ndarray) -> np.ndarray:
    return (operator + operator.conj().T) / 2.0


@lru_cache(maxsize=None)
def _depolarizing_channel(p: float, num_qubits: int):
    return depolarizing_error(float(p), num_qubits).to_quantumchannel()


@lru_cache(maxsize=None)
def _phase_flip_channel(p: float):
    return pauli_error(
        [("I", 1.0 - float(p)), ("Z", float(p))]
    ).to_quantumchannel()


def evolve_unitary(
    operator: np.ndarray,
    gate: object,
    qargs: Iterable[int],
) -> np.ndarray:
    """Apply a gate to a possibly unnormalized signed operator."""
    evolved = DensityMatrix(operator).evolve(Operator(gate), qargs=list(qargs))
    return _symmetrize(evolved.data)


def apply_local_error(
    operator: np.ndarray,
    *,
    qargs: Tuple[int, ...],
    p: float,
    family: NoiseFamily,
) -> np.ndarray:
    """Apply the configured channel to a gate support."""
    if p <= 0.0:
        return operator

    result = DensityMatrix(operator)
    if family == NoiseFamily.DEPOLARIZING:
        channel = _depolarizing_channel(float(p), len(qargs))
        result = result.evolve(channel, qargs=list(qargs))
    elif family == NoiseFamily.DEPHASING:
        channel = _phase_flip_channel(float(p))
        for qubit in qargs:
            result = result.evolve(channel, qargs=[qubit])
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"Unsupported noise family: {family}")
    return _symmetrize(result.data)


def apply_memory_noise(
    operator: np.ndarray,
    *,
    qargs: Iterable[int],
    spec: HardwareNoiseSpec,
) -> np.ndarray:
    """Apply one local memory-noise interval to each listed qubit."""
    result = operator
    for qubit in qargs:
        result = apply_local_error(
            result,
            qargs=(int(qubit),),
            p=spec.p_memory,
            family=spec.family,
        )
    return result


def apply_noisy_circuit(
    operator: np.ndarray,
    circuit: QuantumCircuit,
    spec: HardwareNoiseSpec,
) -> tuple[np.ndarray, int, int]:
    """Execute a unitary circuit and add noise after every physical gate."""
    result = operator
    one_qubit_count = 0
    two_qubit_count = 0

    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"barrier", "delay"}:
            continue

        qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        result = evolve_unitary(result, operation, qargs)
        if len(qargs) == 1:
            one_qubit_count += 1
            gate_error = spec.p_1q
        elif len(qargs) == 2:
            two_qubit_count += 1
            gate_error = spec.p_2q
        else:
            raise ValueError(
                f"Unsupported workload gate {operation.name!r} on {len(qargs)} qubits"
            )

        result = apply_local_error(
            result,
            qargs=qargs,
            p=gate_error,
            family=spec.family,
        )

    return result, one_qubit_count, two_qubit_count


def reset_product_state(M: int, p_reset: float) -> np.ndarray:
    """Independent imperfect resets: (1-p)|0><0| + p|1><1|."""
    single = np.diag([1.0 - p_reset, p_reset]).astype(complex)
    state = np.array([[1.0]], dtype=complex)
    for _ in range(M):
        state = np.kron(single, state)
    return state


__all__ = [
    "apply_local_error",
    "apply_memory_noise",
    "apply_noisy_circuit",
    "evolve_unitary",
    "reset_product_state",
]
