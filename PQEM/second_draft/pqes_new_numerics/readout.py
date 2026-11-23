"""THE OBSERVABLE-READOUT RULE, APPLIED THE SAME WAY TO EVERY ARM AND THE BASELINE.

Published rule (the Hardware defaults): O is measured ideally on every measured
register, with no basis change; readout_q acts only on the recorded sign and
control bits; each spent derangement register is read right after its CSWAP.

Uniform rule (Hardware fields obs_readout_q, obs_basis_change,
obs_flip_correction, derangement_readout), on every readout of O: every arm's
final readout, every spent derangement register and the unmitigated baseline.
  1. BASIS CHANGE. The compiled unitary V that rotates O's eigenbasis to the
     computational basis runs on the register as native gates through
     circuits.apply_layer: eps1 after each one-qubit gate, eps2 after each CNOT,
     the idle channel on the register's qubits that wait. Its depth is added to
     the makespan once (registers are read in parallel) and its gates and idle
     units to the ledger for every register read. With obs_basis_change off, V
     is ideal and free; it still defines which bits the flips act on.
  2. FLIPS. Each measured data qubit's computational-basis bit flips
     independently with probability obs_readout_q. This is simulated exactly
     as an X error just before an ideal Z measurement, so the joint statistics
     of all registers (the derangement cross terms included) are exact.
  3. CORRECTION (obs_flip_correction). The reported value of an outcome
     string b' is f(b') = [(A^-1)^{(x)M} o](b'), A=[[1-q,q],[q,1-q]], o the
     eigenvalue of each basis string: the standard inverse-confusion estimate.
     It is exactly unbiased, E[f]=Tr(O rho_before_flips); its second moment
     enters m2, hence v and the shot plan. Without it the reported value is
     o(b') and the flips bias every arm by about -2q*mu for a two-qubit projector.
  4. DERANGEMENT TIMING (core.run_derangement): 'end' idles every spent
     register until the last CSWAP ends; then the basis change and flips above.

BASIS-CHANGE COMPILATIONS (V maps the target |psi> to |0...0>, so V O V^dagger
is diagonal and O's value is 1 on |0...0> only):
  bell_tilted, M=2: |psi>=cos(pi/6)|00>+sin(pi/6)|11>.
     CNOT(0->1) gives (cos(pi/6)|0>+sin(pi/6)|1>)|0>, then Ry(-pi/3) on qubit 0
     gives |00>. Two layers, [CX01] then [RY0]; qubit 1 idles in the second.
     Per register read: 1 CNOT, 1 one-qubit gate, 1 idle unit, 2 time units.
  tilted, M=1: |psi>=cos(pi/6)|0>+e^{i pi/4}sin(pi/6)|1>.
     Rz(-pi/4) removes the phase (up to a global phase), then Ry(-pi/3) gives
     |0>. Two one-qubit layers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core import Hardware,Ledger,X,dag,conj_local,site_unitary
from circuits import Gate,CNOT,rotations,apply_layer


@dataclass(frozen=True)
class ObservableReadout:
    """A target observable O and the compiled basis change V that reads it."""
    O:np.ndarray
    layers:tuple
    M:int

    def unitary(self):
        """Ideal V (big-endian wires)."""
        V=np.eye(1<<self.M,dtype=complex)
        for layer in self.layers:
            for g in layer:V=site_unitary(g.matrix,g.wires,self.M)@V
        return V

    def values(self):
        """Eigenvalue of O on each computational-basis string after V, checked
        to be the whole of V O V^dagger."""
        V=self.unitary();Oc=V@self.O@dag(V)
        if np.abs(Oc-np.diag(np.diag(Oc))).max()>1e-12:raise AssertionError('V does not diagonalize O')
        return np.real(np.diag(Oc))


BASIS_CHANGE={
    ('bell_tilted',2):((Gate('CX01',(0,1),CNOT()),),(Gate('RY0',(0,),rotations('y',-np.pi/3)),)),
    ('tilted',1):((Gate('RZ0',(0,),rotations('z',-np.pi/4)),),(Gate('RY0',(0,),rotations('y',-np.pi/3)),)),
}


def target_readout(kind:str,M:int,O:np.ndarray)->ObservableReadout:
    """The compiled readout of the target projector O of core.prep_target(M,kind)."""
    if (kind,M) not in BASIS_CHANGE:raise ValueError(f'no compiled basis change for {kind}, M={M}')
    ro=ObservableReadout(np.asarray(O,complex),BASIS_CHANGE[(kind,M)],M)
    ro.values()
    return ro


def uses_readout(hw:Hardware)->bool:
    """True when the readout of O differs from the ideal published readout."""
    return bool(hw.obs_readout_q or hw.obs_basis_change or hw.obs_flip_correction)


def reported_observable(ro:ObservableReadout,hw:Hardware)->np.ndarray:
    """Diagonal operator whose eigenvalues are the reported values of each string."""
    o=ro.values();q=hw.obs_readout_q
    if hw.obs_flip_correction and q:
        Ainv=np.array([[1-q,-q],[-q,1-q]])/(1-2*q);K=np.array([[1.]])
        for _ in range(ro.M):K=np.kron(K,Ainv)
        o=K@o
    return np.diag(o).astype(complex)


def read_out(ops,ro:ObservableReadout,n:int,offsets,hw:Hardware,ledger:Ledger|None=None):
    """Apply the readout channel to every register (first wire at each of
    `offsets`) of each operator in `ops` (n wires). Returns (ops, O_reported).

    The ledger is charged once per register (gates and idle) and once for the
    basis-change depth (registers are read in parallel)."""
    M=ro.M;gate_hw=hw if hw.obs_basis_change else Hardware()
    out=[]
    for i,A in enumerate(ops):
        led=ledger if (i==0 and hw.obs_basis_change) else None
        for off in offsets:
            for layer in ro.layers:A=apply_layer(A,layer,M,gate_hw,offset=off,n=n,ledger=led)
            if hw.obs_readout_q:
                q=hw.obs_readout_q
                for w in range(off,off+M):A=(1-q)*A+q*conj_local(A,X,(w,),n)
        out.append(A)
    if ledger is not None and hw.obs_basis_change:ledger.duration+=len(ro.layers)
    return out,reported_observable(ro,hw)
