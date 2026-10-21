r"""PHYSICAL order-two Virtual Channel Purification (VCP2), Liu-style.

Liu, Zhang, Fei and Cai, "Virtual Channel Purification", PRX Quantum 6,
020325 (2025) (arXiv:2402.07866): the auxiliary register starts maximally
mixed, the NOISY GATE acts on both the main and the auxiliary register between
two controlled register SWAPs, the control (|+>) is measured in X, and
<X (x) O>/<X (x) I> = Tr[O E^(2) U(rho)] with E^(2)=sum_i p_i^2 E_i./sum p_i^2
for Pauli noise. VCP may act on a SUBSET of qubits (their Sec. III.1.3,
Fig. 2d) and one control can purify a whole layer of independent channels
(their Eq. 5: (E1 (x) E2)^(2) = E1^(2) (x) E2^(2)).

IMPLEMENTATION (one gadget per purified circuit LAYER):
  control |+>, auxiliary register on the layer's support S (k=|S| qubits) in I/2^k
  -> compiled CSWAP(c; S_main, aux)            (core.cswap_gadget, k+2 units)
  -> the layer's gates on main AND on aux, each followed by its gate noise
  -> compiled CSWAP(c; S_main, aux)
  -> X readout of the control (readout error), aux discarded.
Main qubits outside S idle during the gadget. Each layer has its own freshly
prepared control: the controls (with their auxiliaries) come round robin from
the smallest pool of physical pairs that keeps each pair's measurement, reset
and re-preparation off the main register's critical path (vcp_pool_size), so
the main register never waits for a previous layer's control. The auxiliary
I/2^k is prepared by a uniformly random Pauli on the spent register (exact for
any leftover state, so no reset). Signs only multiply classically.

The ideal identity W = prod_layers P2 * E^(2) holds for Pauli gate noise with
idle noise switched off; amplitude damping makes the auxiliary state and the
idle noise non-Pauli, so the identity is then only approximate (simulated
exactly anyway).

RESOURCES (per shot): 1 main preparation; every purified gate executes twice
(main + aux) so noisy-gate executions = 2 G_purified + G_unpurified; one
control preparation, one auxiliary preparation and one measurement per
purified layer; 2(2k) CNOTs and 2k Toffolis per purified layer; width
M + pool*(1+k_max) counting every physical control and auxiliary qubit.
"""
from __future__ import annotations
import numpy as np
from core import (Hardware,Ledger,PLUS_DM,dag,tensor,conj_local,gate_noise,apply_idle,
                  cswap_gadget,cswap_duration,gadget_counts,ancilla_plus,observed_branches,_x_branches,_herm)
from circuits import apply_layer,ground,count_gates


def vcp2_layer_gadget(x,layer,M,hw:Hardware,reused_control:bool,alg_hw:Hardware|None=None):
    """Order-two VCP gadget for one layer on UNNORMALIZED x (M-qubit main register).

    Returns reported (plus, minus) on the main register and the gadget duration.
    `alg_hw` sets the noise of the algorithm gates being purified (default: hw);
    hw sets the apparatus (CSWAPs, control, readout, idle).
    """
    alg_hw=hw if alg_hw is None else alg_hw
    S=sorted({w for g in layer for w in g.wires});k=len(S)
    n=1+M+k;aux=[1+M+i for i in range(k)];amap={w:1+M+i for i,w in enumerate(S)}
    state=tensor(ancilla_plus(hw,reused_control),x,np.eye(1<<k,dtype=complex)/(1<<k))
    outside=[1+w for w in range(M) if w not in S]
    Tc=cswap_duration(k,hw)
    state,_=cswap_gadget(state,n,0,[1+w for w in S],aux,hw)
    state=apply_idle(state,outside,n,Tc,hw)
    for g in layer:
        for wires in (tuple(1+w for w in g.wires),tuple(amap[w] for w in g.wires)):
            state=conj_local(state,g.matrix,wires,n)
            state=gate_noise(state,wires,n,alg_hw)
    state=apply_idle(state,[0],n,1,hw)            # control waits: apparatus idle
    state=apply_idle(state,outside,n,1,alg_hw)    # untouched main qubits in this layer: algorithm idle, as in run_layers
    state,_=cswap_gadget(state,n,0,[1+w for w in S],aux,hw)
    state=apply_idle(state,outside,n,Tc,hw)
    d=1<<M;e=1<<k
    p,m=_x_branches(state,d,e)
    return observed_branches(p,m,hw.readout_q),2*Tc+1,k


def gadget_duration(layer,hw:Hardware):
    k=len({w for g in layer for w in g.wires})
    return 2*cswap_duration(k,hw)+1


def vcp_pool_size(layers_purified,hw:Hardware):
    """Smallest number of (control, auxiliary) pairs, used round robin, such that
    a pair's measurement, reset and |+>/twirl re-preparation (t_meas+t_reset+1)
    always finishes before its next layer starts with gadgets run back to back.
    The main register then never waits on a previous layer's control."""
    D=[gadget_duration(L,hw) for L in layers_purified];need=hw.t_meas+hw.t_reset+1
    for n in range(1,len(D)+1):
        if all(sum(D[l+1:l+n])>=need for l in range(len(D)-n)):return n
    return max(len(D),1)


def simulate_layerwise_vcp2(M,layers_purified,layers_plain,hw:Hardware,initial=None,ledger=None,alg_hw=None):
    """Purify `layers_purified` layer by layer, then run `layers_plain` unpurified.

    Every layer gets its own freshly prepared control (and auxiliary) from a
    pool of vcp_pool_size() physical pairs, so the main register never waits
    for a previous control's measurement or reset. Unnormalized parity-resolved
    states are propagated: feeding a normalized weighted operator forward as a
    physical state would hide the signed-record sampling cost. Returns (R, W)
    and fills `ledger` (width counts every physical control and auxiliary).
    """
    led=ledger if ledger is not None else Ledger()
    alg_hw=hw if alg_hw is None else alg_hw
    d=1<<M
    init=ground(M) if initial is None else initial
    states={1:init.astype(complex),-1:np.zeros((d,d),complex)}
    pool=vcp_pool_size(layers_purified,hw)
    pfree=[0]*pool;pused=[False]*pool
    t=0;kmax=0
    led.preparations=1
    for ell,layer in enumerate(layers_purified):
        c=ell%pool;reused=pused[c]
        start=max(t,pfree[c]+((hw.t_reset+1) if reused else 0))
        if start>t:     # cannot happen with the chosen pool; kept so the clock stays honest
            states={s:apply_idle(states[s],range(M),M,start-t,hw) for s in (1,-1)}
            led.data_idle+=M*(start-t)
        nxt={1:np.zeros((d,d),complex),-1:np.zeros((d,d),complex)}
        for s in (1,-1):
            if not np.any(states[s]):continue
            (pl,mi),dur,k=vcp2_layer_gadget(states[s],layer,M,hw,reused,alg_hw)
            nxt[s]+=pl;nxt[-s]+=mi
        k=len({w for g in layer for w in g.wires});dur=gadget_duration(layer,hw)
        kmax=max(kmax,k)
        if reused:led.ancilla_resets+=1
        gadget_counts(k,hw,led);gadget_counts(k,hw,led)
        led.gates1+=1+sum(2 for g in layer if len(g.wires)==1)
        led.cnot+=sum(2 for g in layer if len(g.wires)==2)
        led.control_idle+=1;led.data_idle+=(M-k)*dur
        led.measurements+=1
        states=nxt;t=start+dur
        pfree[c]=t+hw.t_meas;pused[c]=True
    for layer in layers_plain:
        states={s:apply_layer(states[s],layer,M,alg_hw) for s in (1,-1)}
        led.gates1+=sum(1 for g in layer if len(g.wires)==1)
        led.cnot+=sum(1 for g in layer if len(g.wires)==2)
        led.data_idle+=M-len({w for g in layer for w in g.wires})
        t+=1
    led.duration=t+(hw.t_meas if layers_purified else 0)
    led.peak_qubits=M+pool*(1+kmax) if layers_purified else M
    R=states[1]+states[-1];W=states[1]-states[-1]
    return _herm(R),_herm(W)


def vcp_accounting(layers_purified,layers_plain):
    """Static resource vector of layerwise VCP2 (checked against the Ledger in tests)."""
    g1p,g2p=count_gates(layers_purified);g1u,g2u=count_gates(layers_plain)
    nl=len(layers_purified)
    return dict(main_preparations=1,aux_preparations=nl,control_preparations=nl,
                ancilla_measurements=nl,noisy_gate_executions=2*(g1p+g2p)+g1u+g2u,
                purified_gates=g1p+g2p,purified_layers=nl,
                cswap_toffolis=2*sum(len({w for g in L for w in g.wires}) for L in layers_purified))
