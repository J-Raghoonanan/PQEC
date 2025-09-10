"""
State preparation utilities for the SWAP-based purification simulator (Qiskit).

This module produces a *preparation circuit* U_psi on M qubits such that
U_psi |0...0> = |psi>, and also returns the corresponding reference
Statevector |psi> for metrics.

Supported target kinds (see configs.StateKind):
- manual:     user supplies a circuit or statevector
- haar:       Haar-random pure state
- random_circuit: shallow random rotations + entanglers
- hadamard:   (H|0>)^{\otimes M}
- ghz:        (|0...0> + |1...1>)/sqrt(2)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector

from .configs import StateKind, TargetSpec


def _ensure_num_qubits(circ: QuantumCircuit, M: int) -> QuantumCircuit:
    if circ.num_qubits != M:
        raise ValueError(
            f"Manual circuit has {circ.num_qubits} qubits but target M={M}. "
            "Please supply a circuit defined on exactly M qubits."
        )
    return circ


def _prep_from_statevector(psi: Statevector) -> QuantumCircuit:
    """Return an Initialize-based circuit that prepares 'psi' from |0...0>.

    Notes
    -----
    - Initialize synthesizes a preparation; depth will scale with M.
    - For larger systems, you may prefer supplying a custom circuit.
    """
    M = int(round(np.log2(psi.dim)))
    init = Initialize(psi.data)
    qc = QuantumCircuit(M, name="prep_psi")
    qc.append(init, range(M))
    return qc


def _build_hadamard(M: int) -> Tuple[QuantumCircuit, Statevector]:
    qc = QuantumCircuit(M, name="prep_hadamard")
    for q in range(M):
        qc.h(q)
    psi = Statevector.from_instruction(qc)
    return qc, psi


def _build_ghz(M: int) -> Tuple[QuantumCircuit, Statevector]:
    if M < 1:
        raise ValueError("M must be >= 1 for GHZ")
    qc = QuantumCircuit(M, name="prep_ghz")
    qc.h(0)
    for q in range(1, M):
        qc.cx(0, q)
    psi = Statevector.from_instruction(qc)
    return qc, psi


def _build_haar(M: int, seed: Optional[int]) -> Tuple[QuantumCircuit, Statevector]:
    rng = np.random.default_rng(seed)
    # Generate a random statevector and synthesize via Initialize
    coeffs = (rng.normal(size=2**M) + 1j * rng.normal(size=2**M)).astype(complex)
    coeffs /= np.linalg.norm(coeffs)
    psi = Statevector(coeffs)
    qc = _prep_from_statevector(psi)
    qc.name = "prep_haar"
    return qc, psi


def _build_random_circuit(M: int, layers: int, seed: Optional[int]) -> Tuple[QuantumCircuit, Statevector]:
    """Random shallow circuit: RX/RY/RZ on all qubits + CZ entangler ring per layer."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(M, name="prep_random")
    for _ in range(max(1, layers)):
        # single-qubit random rotations
        for q in range(M):
            qc.rx(float(2 * np.pi * rng.random()), q)
            qc.ry(float(2 * np.pi * rng.random()), q)
            qc.rz(float(2 * np.pi * rng.random()), q)
        # entanglers (CZ ring)
        for q in range(M):
            qc.cz(q, (q + 1) % M)
    psi = Statevector.from_instruction(qc)
    return qc, psi


def build_target(spec: TargetSpec) -> Tuple[QuantumCircuit, Statevector]:
    """Construct a preparation circuit and reference |psi> per TargetSpec.

    Returns
    -------
    (qc_prep, psi):
        - qc_prep: QuantumCircuit on 'spec.M' qubits, such that qc_prep |0...0> = |psi>.
        - psi:     Statevector(|psi>) for metrics (fidelity, trace distance).
    """
    if spec.M <= 0:
        raise ValueError("M must be positive")

    if spec.kind == StateKind.manual:
        if spec.manual_statevector is not None:
            psi = spec.manual_statevector
            if int(round(np.log2(psi.dim))) != spec.M:
                raise ValueError(
                    f"Manual statevector dimension {psi.dim} incompatible with M={spec.M}."
                )
            qc = _prep_from_statevector(psi)
            qc.name = "prep_manual_sv"
            return qc, psi
        if spec.manual_circuit is not None:
            qc = _ensure_num_qubits(spec.manual_circuit, spec.M).copy()
            qc.name = "prep_manual_circ"
            psi = Statevector.from_instruction(qc)
            return qc, psi
        raise ValueError(
            "StateKind.manual requires either manual_statevector or manual_circuit"
        )

    if spec.kind == StateKind.hadamard:
        return _build_hadamard(spec.M)
    if spec.kind == StateKind.ghz:
        return _build_ghz(spec.M)
    if spec.kind == StateKind.haar:
        return _build_haar(spec.M, spec.seed)
    if spec.kind == StateKind.random_circuit:
        return _build_random_circuit(spec.M, spec.random_layers, spec.seed)

    raise ValueError(f"Unsupported StateKind: {spec.kind}")


__all__ = ["build_target"]
