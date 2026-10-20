"""Known algorithm targets, physically noisy circuit blocks, and VCP proxy.

This code deliberately distinguishes TWO physically different uses of a state:
  1. The ideal TARGET projector P0 for benchmarking observable error.
  2. A noisy input density matrix rho_A prepared by the first circuit block.

A generic ideal circuit is separated into A and B. The real noisy B block is
CPTP and acts once on the surviving output for mid-circuit compression, but
N times on N separately prepared outputs for end-only purification.

Using a full known projector P0 is an analytical benchmark: it assumes we can
perform that final measurement perfectly (or equivalently uncompute the known
ideal preparation and measure all zeros). The cost/noise of uncomputation is
NOT included here; this assumption is explicitly noted in the README.

The optional VCP proxy is explicitly an IDEAL CHANNEL-LEVEL estimator under
stochastic Pauli noise, not a simulation of a noisy VCP implementation. Its
input-output fidelity must never be used as a matched physical-resource claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from core import I2,H,X,Y,Z,pure_dm,dag,site_unitary,depolarize_subset


def rotations(axis,theta):
    sig={'x':X,'y':Y,'z':Z}[axis]
    return np.cos(theta/2)*I2-1j*np.sin(theta/2)*sig


def CNOT():
    U=np.eye(4,dtype=complex)
    U[2,2]=U[3,3]=0
    U[2,3]=U[3,2]=1
    return U


def CZ():return np.diag([1,1,1,-1]).astype(complex)


@dataclass(frozen=True)
class Gate:
    name:str
    wires:tuple[int,...]
    matrix:np.ndarray


def algorithm_blocks(M:int, b_repeats:int=1):
    """Fixed, deterministic, reproducible entangling circuit family.

    M=1: no fake entanglers, but noncommuting single-qubit rotations.
    M=2: A includes H, Ry, CZ, Rz; B includes Ry, CX, Rx, CZ, Rz.
    M=3: the same two-qubit interactions on (0,1) plus extra (1,2) CZ.
    B_repeats varies the downstream gate count WITHOUT changing the
    meaning of which ideal target the plot is evaluated against.
    """
    if M not in (1,2,3):raise ValueError('M=1..3')
    A=[Gate('H0',(0,),H),Gate('RY0',(0,),rotations('y',.66))]
    B=[Gate('RX0',(0,),rotations('x',.45)),Gate('RZ0',(0,),rotations('z',-.37))]
    if M>1:
        A.extend([Gate('RY1',(1,),rotations('y',.58)),Gate('CZ01',(0,1),CZ()),
                  Gate('RZ1',(1,),rotations('z',.35))])
        B.extend([Gate('RY1',(1,),rotations('y',-.71)),Gate('CX01',(0,1),CNOT()),
                  Gate('RX1',(1,),rotations('x',.31)),Gate('CZ01',(0,1),CZ())])
    if M>2:
        A.append(Gate('CZ12',(1,2),CZ()))
        B.append(Gate('RX2',(2,),rotations('x',.28)))
    return A,B*b_repeats


def apply_gates(rho,gates,M:int,noise_p:float):
    """Noisy deterministic implementation of a known unitary circuit.

    Every 1- or 2-qubit ideal gate is followed by a depolarizing channel
    acting on *exactly the same wires*. This is a gate-local channel; its
    probability noise_p is separate from the Fredkin error probability.
    """
    for g in gates:
        U=site_unitary(g.matrix,g.wires,M)
        rho=U@rho@dag(U)
        if noise_p:rho=depolarize_subset(rho,g.wires,M,noise_p)
    return rho


def fixed_circuit_case(M=2,b_repeats=1,algorithm_p=.025):
    A,B=algorithm_blocks(M,b_repeats)
    ground=np.zeros((1<<M,1<<M),complex);ground[0,0]=1
    rho_ideal_A=apply_gates(ground,A,M,0)
    rho_ideal_output=apply_gates(rho_ideal_A,B,M,0)
    rho_noisy_A=apply_gates(ground,A,M,algorithm_p)
    rho_noisy_output=apply_gates(rho_noisy_A,B,M,algorithm_p)
    return dict(M=M,b_repeats=b_repeats,A=A,B=B,ideal=rho_ideal_output,
                input_A=rho_noisy_A,noisy_output=rho_noisy_output,
                algorithm_gate_p=algorithm_p)


def apply_downstream(rho,case):
    return apply_gates(rho,case['B'],case['M'],case['algorithm_gate_p'])


def apply_ideal_pauli_channel(rho,wires,p,M,power=1):
    """Ideal VCP-style Pauli channel with weights raised to power N.

    Only a controlled *theoretical proxy*. Does not simulate physical VCP
    ancillas, Choi preparation, normalization readout noise or gate overhead.
    For a one-qubit Pauli stochastic channel, the Choi eigenvalues are
    (1-p,p/3,p/3,p/3), and ideal Nth-order channel purification yields
    normalized powers. We apply this per error location, not to a general
    coherent or nonunital channel. Eq. consistent with VCP (Liu et al.,
    PRX Quantum 6, 020325 (2025)).
    """
    probabilities=np.array([1-p,p/3,p/3,p/3],float)**power
    probabilities/=probabilities.sum()
    for wire in wires:
        ps=(I2,X,Y,Z)
        out=np.zeros_like(rho)
        for prob,P in zip(probabilities,ps):
            U=site_unitary(P,(wire,),M)
            out+=prob*(U@rho@dag(U))
        rho=out
    return rho


def vcp_pauli_proxy(M=2,b_repeats=1,physical_p=.025,power=2):
    """Ideal layerwise VCP diagnostic; NOT included in matched-cost rankings.

    Here every physical gate's local depolarizing noise is replaced by a
    tensor product of ideal purified *single-qubit* Pauli channels. Thus the
    comparison uses the same location-wise stochastic Pauli noise assumption
    for different powers, but it DOES NOT model a physical VCP circuit or
    its sampling variance. The metric is an optimistic channel-level target.
    """
    A,B=algorithm_blocks(M,b_repeats)
    ground=np.zeros((1<<M,1<<M),complex);ground[0,0]=1
    target=apply_gates(ground,A+B,M,0)
    rho=ground
    for g in A+B:
        U=site_unitary(g.matrix,g.wires,M)
        rho=U@rho@dag(U)
        rho=apply_ideal_pauli_channel(rho,g.wires,physical_p,M,power)
    return float(np.trace(rho@target).real)
