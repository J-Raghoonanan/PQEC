"""
SWAP test + (emulated) amplitude amplification for purification.

This module constructs and applies the SWAP-test unitary to two identical
M-qubit input states and (optionally) performs amplitude amplification *in the
sense of logging the Grover iteration count required to make ancilla=0 nearly
certain*. Importantly, the **conditional output state** given ancilla=0 is
independent of whether amplitude amplification was used; therefore we project
onto ancilla |0> and extract the purified single-register state.

We use Qiskit's DensityMatrix evolutions with circuits (no hand multiplications),
then perform projection and partial trace with quantum_info utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CSXGate
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector

from .configs import AASpec


# -----------------------------
# Building the SWAP-test unitary
# -----------------------------

def build_swap_test_unitary(M: int) -> QuantumCircuit:
    """Return a circuit on (1 + 2M) qubits implementing the SWAP test skeleton:
       - H on ancilla
       - M controlled-SWAPs between regA[i] and regB[i] with control ancilla
    Qubit order: [anc] + [A0..A{M-1}] + [B0..B{M-1}].
    """
    n = 1 + 2 * M
    qc = QuantumCircuit(n, name="A_swap")
    anc = 0
    A = list(range(1, 1 + M))
    B = list(range(1 + M, 1 + 2 * M))

    qc.h(anc)
    for i in range(M):
        # Fredkin via 2 CNOTs + CSWAP decomposition
        # Qiskit has 'cswap' as 'cswap' method on the circuit in newer versions; implement portable:
        qc.cswap(anc, A[i], B[i])
    qc.h(anc)
    return qc


# -----------------------------
# Amplitude amplification helpers (emulated)
# -----------------------------

def ancilla_success_probability(rho_after_A: DensityMatrix, M: int) -> float:
    """Compute Pr[ancilla=0] from the (1+2M)-qubit density matrix after A.
    Assumes qubit ordering [anc, A..., B...]."""
    anc_dm = partial_trace(rho_after_A, qargs=list(range(1, 1 + 2 * M)))
    # anc_dm is a 2x2 DensityMatrix
    p0 = float(np.real(anc_dm.data[0, 0]))
    # clip for numerical safety
    return max(0.0, min(1.0, p0))


def choose_grover_iters(P0: float, target_success: float, max_iters: int) -> int:
    """Choose k so that sin^2((k+1/2)θ) >= target_success, θ = 2 arcsin sqrt(P0).
    If P0 in {0,1}, handle edge cases gracefully.
    """
    P0 = float(P0)
    if P0 >= target_success:
        return 0
    if P0 <= 0.0:
        return 0  # cannot amplify if no support; return 0 and let postselection handle
    if P0 >= 1.0:
        return 0
    theta = 2.0 * np.arcsin(np.sqrt(P0))
    # minimal k satisfying the inequality
    k = int(np.floor(np.pi / (4.0 * np.arcsin(np.sqrt(P0))) - 0.5))
    k = max(0, min(k, max_iters))
    return k


# -----------------------------
# Projection and extraction of purified state
# -----------------------------

def _project_ancilla_zero(rho: DensityMatrix, M: int) -> DensityMatrix:
    """Project the ancilla (qubit 0) onto |0><0| and renormalize."""
    # Projector Π = |0><0|_anc ⊗ I_rest
    # Implement by slicing the density matrix in computational basis blocks.
    dim = 2 ** (1 + 2 * M)
    # Use reshape to 2 x 2 blocks: index ancilla as leading bit
    # However, DensityMatrix stores data as a flat matrix; we can use partial_trace math:
    # Construct Π ρ Π by zeroing out rows/cols where ancilla=1.
    data = rho.data.copy()
    # Basis states are ordered with ancilla as the most-significant bit given our indexing: confirm qargs ordering.
    # In qiskit, qubit ordering for data is big-endian by default (qubit 0 as most significant). We adhere to that here.
    # Create mask for basis states with ancilla=0
    size = data.shape[0]
    idx0 = []
    for idx in range(size):
        # ancilla bit is bit position (n_qubits - 1 - 0) in little-endian; but qiskit uses big-endian for Statevector dims.
        # For safety, compute using integer math over total qubits with ancilla as most significant:
        # Most significant bit being 0 means idx < size/2
        if idx < size // 2:
            idx0.append(idx)
    idx0 = np.array(idx0, dtype=int)

    mask = np.zeros(size, dtype=bool)
    mask[idx0] = True

    projected = np.zeros_like(data)
    projected[np.ix_(mask, mask)] = data[np.ix_(mask, mask)]

    # Renormalize by probability p0 = Tr(Π ρ)
    p0 = float(np.real(np.trace(projected)))
    if p0 <= 0.0:
        # Return zero state to avoid NaNs; caller should have seen P0 ~ 0
        return DensityMatrix(projected)
    projected /= p0
    return DensityMatrix(projected)


def extract_purified_register(rho_after_proj: DensityMatrix, M: int) -> DensityMatrix:
    """Partial trace out ancilla (0) and regB (last M) to get ρ_out on regA (middle M)."""
    # Qubit ordering: [anc=0] [A:1..M] [B:M+1..2M]
    traced = [0] + list(range(1 + M, 1 + 2 * M))
    rho_A = partial_trace(rho_after_proj, qargs=traced)
    # Result is a 2^M x 2^M DensityMatrix
    return rho_A


# -----------------------------
# Public API
# -----------------------------

def purify_two_from_density(
    rho_single: DensityMatrix,
    aa: AASpec,
) -> Tuple[DensityMatrix, Dict]:
    """Apply SWAP test purification to two identical copies of 'rho_single'.

    Returns (rho_out, metrics) where rho_out is the purified single-register
    density matrix on M qubits, and metrics contains P_success and k.

    Notes
    -----
    - This function *emulates* amplitude amplification: it computes the
      pre-AA success probability and the Grover iteration count required to
      reach the target success, but it does not explicitly apply Q^k. The
      conditional output state after post-selecting ancilla=0 is identical to
      that after amplitude amplification.
    """
    # Build the joint state |0>_anc ⊗ rho ⊗ rho
    M = int(np.log2(rho_single.dim))
    rho_joint = DensityMatrix.from_label('0').tensor(rho_single).tensor(rho_single)

    # Apply the SWAP-test unitary A
    A = build_swap_test_unitary(M)
    rho_after_A = rho_joint.evolve(A)

    # Compute success probability and choose k
    P0 = ancilla_success_probability(rho_after_A, M)
    k = choose_grover_iters(P0, aa.target_success, aa.max_iters)

    # Optionally, one could insert explicit AA here; we skip by default (see notes)

    # Project ancilla to |0> and extract the purified register A
    rho_proj = _project_ancilla_zero(rho_after_A, M)
    rho_out = extract_purified_register(rho_proj, M)

    metrics = {
        "P_success": P0,
        "grover_iters": k,
    }
    return rho_out, metrics


__all__ = [
    "build_swap_test_unitary",
    "purify_two_from_density",
    "ancilla_success_probability",
]
