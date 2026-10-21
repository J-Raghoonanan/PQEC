"""Known algorithm targets, layered noisy circuits, and the ideal channel proxy.

Two physically different uses of a state are kept apart:
  1. the ideal TARGET projector P0 used to benchmark observable error;
  2. noisy density matrices produced by running the noisy circuit.

Every algorithm gate is followed by the SAME replacement depolarizing channel
used for the SWAP gadgets (core.gate_noise: eps1 after one-qubit gates, eps2
after two-qubit gates). Gates are grouped into ASAP layers of one time unit;
qubits not acted on in a layer receive the shared idle channel for that unit.

Using a full known projector P0 is an analytical benchmark: it assumes the
final measurement is perfect (or the known ideal preparation is uncomputed).
That cost is NOT included; the README states this assumption.

The channel PROXY applies ideal order-N channel purification to each gate's
Pauli channel with no apparatus at all. It is a reference curve only and is
never ranked against physical circuits.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core import (I2,H,X,Y,Z,pure_dm,dag,site_unitary,conj_local,gate_noise,apply_idle,
                  Hardware,Ledger,depolarize_subset)


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


def algorithm_blocks(M:int, b_repeats:int=1, family:str='entangling'):
    """Fixed, deterministic, reproducible circuit family U=U_B^r U_A.

    M=1: noncommuting single-qubit rotations only.
    M=2: A has H, Ry, CZ, Rz; B has Ry, CX, Rx, CZ, Rz.
    M=3: the same two-qubit interactions on (0,1) plus a (1,2) CZ.
    family='product' drops every two-qubit gate (controlled product-state
    benchmark: the ideal output is a product state, and so is every noisy one).
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
    if family=='product':
        A=[g for g in A if len(g.wires)==1];B=[g for g in B if len(g.wires)==1]
    elif family!='entangling':raise ValueError(family)
    return A,B*b_repeats


def has_two_qubit_gate(layer):
    """Channel-purification policy shared by VCP2 and its proxy."""
    return any(len(g.wires)==2 for g in layer)


def layer_gates(gates):
    """ASAP layering: a gate goes in the first layer after the last layer that
    touches any of its wires. Gate order on each wire is preserved."""
    layers=[];last={}
    for g in gates:
        k=max((last.get(w,-1) for w in g.wires),default=-1)+1
        if k==len(layers):layers.append([])
        layers[k].append(g)
        for w in g.wires:last[w]=k
    return layers


def full_layers(M,family='entangling',b_repeats=2):
    A,B=algorithm_blocks(M,b_repeats,family)
    return layer_gates(A+B)


# Fixed total depths of the study circuits. For M=2 they are divisible by 4 so
# that L_A/L in {0,1/4,1/2,3/4,1} is realized exactly: the entangling circuit is
# the first 12 of the 13 layers of U_B^2 U_A; the product circuit is U_B^3 U_A
# (8 layers).
STUDY_DEPTH={(2,'entangling'):(2,12),(2,'product'):(3,8)}


def study_layers(M,family='entangling'):
    """The fixed circuit used by the placement and channel-purification studies."""
    reps,L=STUDY_DEPTH.get((M,family),(2,None))
    layers=full_layers(M,family,reps)
    return layers[:L] if L else layers


def apply_layer(rho,layer,M,hw:Hardware,offset=0,n=None,ledger=None):
    """One noisy time unit: ideal gates, their gate noise, idle on untouched qubits.
    `ledger` (if given) is charged the layer's gates and idle units."""
    n=M if n is None else n
    busy=set()
    for g in layer:
        wires=tuple(offset+w for w in g.wires)
        rho=conj_local(rho,g.matrix,wires,n)
        rho=gate_noise(rho,wires,n,hw)
        busy.update(wires)
    idle=[offset+w for w in range(M) if offset+w not in busy]
    rho=apply_idle(rho,idle,n,1,hw)
    if ledger is not None:
        ledger.data_idle+=len(idle)
        ledger.gates1+=sum(1 for g in layer if len(g.wires)==1)
        ledger.cnot+=sum(1 for g in layer if len(g.wires)==2)
    return rho


def run_layers(rho,layers,M,hw:Hardware,ledger=None):
    for layer in layers:rho=apply_layer(rho,layer,M,hw,ledger=ledger)
    return rho


def ground(M):
    g=np.zeros((1<<M,1<<M),complex);g[0,0]=1;return g


def ideal_output(layers,M):
    return run_layers(ground(M),layers,M,Hardware())


def count_gates(layers):
    one=sum(1 for L in layers for g in L if len(g.wires)==1)
    two=sum(1 for L in layers for g in L if len(g.wires)==2)
    return one,two


def cut_case(M,family,f,hw:Hardware|None=None):
    """Fixed total depth L (study_layers); compression after the first f*L layers.

    f*L must be an integer, so the requested fraction is realized exactly.
    `source` is the ledger of one run of the pre-cut layers (one copy of rho_A).
    """
    hw=Hardware() if hw is None else hw
    layers=study_layers(M,family);L=len(layers)
    cut=int(round(f*L))
    if abs(cut-f*L)>1e-9:raise ValueError(f'L_A/L={f} not realizable at L={L}')
    pre,post=layers[:cut],layers[cut:]
    source=Ledger()
    rhoA=run_layers(ground(M),pre,M,hw,ledger=source)
    target=ideal_output(layers,M)
    return dict(M=M,family=family,f=f,L=L,cut=cut,pre=pre,post=post,rho_A=rhoA,source=source,target=target,
                gates_pre=count_gates(pre),gates_post=count_gates(post))


def pauli_channel_probs(eps,k):
    """Replacement depolarizing D_eps on k qubits as a Pauli channel."""
    pI=1-eps+eps/4**k;pP=eps/4**k
    return pI,pP


def apply_ideal_purified_gate_noise(rho,wires,eps,M,power=1):
    """Ideal order-`power` purification of D_eps on `wires` (proxy only).

    D_eps is the Pauli channel with weights (1-eps+eps/4^k, eps/4^k,...); ideal
    channel purification raises the weights to `power` and renormalizes
    (Liu et al., PRX Quantum 6, 020325, Eq. 2). No ancilla, no sampling cost.
    """
    k=len(wires);pI,pP=pauli_channel_probs(eps,k)
    wI,wP=pI**power,pP**power;Z_=wI+(4**k-1)*wP
    # sum_P P rho P over all 4^k Paulis = 4^k * (I/d (x) Tr_wires rho)
    full=depolarize_subset(rho,tuple(wires),M,1.0)*4**k
    return (wI*rho+wP*(full-rho))/Z_


def vcp_pauli_proxy(M=2,b_repeats=2,eps2=.01,power=2,family='entangling',eps1=None):
    """Ideal layerwise channel-purification reference fidelity (NOT physical).
    Only layers containing a two-qubit gate are purified, as in vcp_physical.
    b_repeats=None uses the study circuit (study_layers)."""
    eps1=eps2/10 if eps1 is None else eps1
    layers=study_layers(M,family) if b_repeats is None else full_layers(M,family,b_repeats)
    target=ideal_output(layers,M)
    rho=ground(M)
    for layer in layers:   # same policy as physical VCP2: purify layers with a two-qubit gate only
        pw=power if has_two_qubit_gate(layer) else 1
        for g in layer:
            rho=conj_local(rho,g.matrix,g.wires,M)
            rho=apply_ideal_purified_gate_noise(rho,g.wires,eps1 if len(g.wires)==1 else eps2,M,pw)
    return float(np.trace(rho@target).real)
