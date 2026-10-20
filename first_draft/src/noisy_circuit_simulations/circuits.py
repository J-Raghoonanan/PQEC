"""Preparation and reproducible circuit blocks for the noisy analysis."""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit

from .config import CircuitKind, CircuitSpec


def build_preparation_circuit(M: int) -> QuantumCircuit:
    """Prepare the legacy |+>^M initial state using explicit H gates."""
    circuit = QuantumCircuit(M, name="prepare_plus")
    for qubit in range(M):
        circuit.h(qubit)
    return circuit


def _build_echo_block(M: int, cycle: int) -> QuantumCircuit:
    """An ideal identity block containing nontrivial 1q and 2q gates."""
    circuit = QuantumCircuit(M, name=f"echo_{cycle}")
    theta = 0.23 + 0.07 * cycle
    for qubit in range(M):
        circuit.rx(theta, qubit)
        circuit.rx(-theta, qubit)

    offset = cycle % 2
    for left in range(offset, M - 1, 2):
        circuit.cx(left, left + 1)
        circuit.cx(left, left + 1)
    return circuit


def _build_brickwork_block(
    M: int,
    cycle: int,
    rng: np.random.Generator,
    rotation_scale: float,
) -> QuantumCircuit:
    """One hardware-efficient random rotation and CNOT brickwork layer."""
    circuit = QuantumCircuit(M, name=f"brickwork_{cycle}")
    for qubit in range(M):
        theta = rotation_scale * rng.uniform(-np.pi, np.pi)
        phi = rotation_scale * rng.uniform(-np.pi, np.pi)
        circuit.ry(float(theta), qubit)
        circuit.rz(float(phi), qubit)

    offset = cycle % 2
    for left in range(offset, M - 1, 2):
        circuit.cx(left, left + 1)
    return circuit


def build_workload_blocks(spec: CircuitSpec) -> List[QuantumCircuit]:
    """Return exactly ``spec.cycles`` ideal workload blocks."""
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    blocks: List[QuantumCircuit] = []

    for cycle in range(spec.cycles):
        if spec.kind == CircuitKind.IDLE:
            block = QuantumCircuit(spec.M, name=f"idle_{cycle}")
        elif spec.kind == CircuitKind.ECHO:
            block = _build_echo_block(spec.M, cycle)
        elif spec.kind == CircuitKind.BRICKWORK:
            block = _build_brickwork_block(
                spec.M,
                cycle,
                rng,
                spec.rotation_scale,
            )
        else:  # pragma: no cover - Enum exhaustiveness guard
            raise ValueError(f"Unsupported circuit kind: {spec.kind}")
        blocks.append(block)

    return blocks


def count_workload_gates(circuit: QuantumCircuit) -> tuple[int, int]:
    """Return (one-qubit gates, two-qubit gates), excluding directives."""
    one_qubit = 0
    two_qubit = 0
    for instruction in circuit.data:
        if instruction.operation.name in {"barrier", "delay"}:
            continue
        arity = len(instruction.qubits)
        if arity == 1:
            one_qubit += 1
        elif arity == 2:
            two_qubit += 1
        else:
            raise ValueError(
                f"Workload gate {instruction.operation.name!r} has arity {arity}; "
                "only one- and two-qubit workload gates are supported."
            )
    return one_qubit, two_qubit


__all__ = [
    "build_preparation_circuit",
    "build_workload_blocks",
    "count_workload_gates",
]
