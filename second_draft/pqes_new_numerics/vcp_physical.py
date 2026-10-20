r"""PHYSICAL order-two Virtual Channel Purification (VCP) comparator.

Circuit implements an experimentally specified SUPERINSTRUMENT for a
stochastic PAULI channel (not arbitrary coherent/nonunital noise):

  control |+>, data A in sigma, auxiliary data B in I/d
  -> controlled REG-SWAP S
  -> independent identical noisy channels E on A and E on B
  -> controlled REG-SWAP S
  -> measure control X, discard B, retain A.

For orthogonal Pauli Kraus operators K_i=sqrt(p_i) P_i, the IDEAL signed
operator of this physical instrument is

           W_out = sum_i p_i**2 P_i sigma P_i.

The normal output is the ORIGINAL noisy channel E(sigma) for unital Pauli
noise. The equality follows because, in the ancilla off-diagonal block,

    Tr_B[(K_i sigma K_j^dag) \otimes (K_j K_i^dag/d)]
      = delta_ij p_i**2 P_i sigma P_i

when Tr(P_i^dag P_j)/d=delta_ij. This derivation is channel purification,
not state rho**2 filtering. See Liu et al., PRX Quantum 6, 020325 (2025),
for the architecture and its scope. We implement the ideal Pauli-channel
superinstrument directly with a noisy Fredkin/readout model, so the
comparison does not grant VCP an imaginary zero-overhead channel gate.

IMPORTANT RESOURCE MODEL:
- At EACH noisy Pauli channel location, VCP uses two instances of E, one on
  the main register and one on the maximally mixed auxiliary register.
- Two full controlled REG-SWAP passes are needed, each M native Fredkins.
- One ancilla X readout and its classical parity sign are recorded per layer.
- One main quantum input state is propagated between layers.
- For a multi-gate algorithm, apply the intended ideal algorithmic gate on
  the main register, then the above VCP instrument on its associated noise
  channel. Signed weights from all previous layers multiply the new sign.
- Noise on the VCP Fredkins is included; the auxiliary I/d preparation,
  ideal algorithm gate and final target-projector measurement are assumed
  perfect. Gate errors under study are stochastic Pauli noise in E itself.
- This is a PHYSICAL instrument simulator for order-2 VCP under those stated
  assumptions. We do not claim it models higher-order VCP, optimal VCP,
  coherent/non-Pauli noise, or a hardware-calibrated gate decomposition.

The comparisons involving VCP have a different resource vector from state VD.
We report both state preparations and channel calls, not just number of qubits.
"""
from __future__ import annotations
import numpy as np
from core import (Noise,PLUS_DM,I2,X,Y,Z,dag,tensor,embed_pauli,
                  controlled_register_swap,observed_branches,metrics,simulate,resources)
from circuits import (algorithm_blocks,apply_ideal_pauli_channel,site_unitary)


def _local_pauli_noise(full,wires,p,M):
    """Apply a local Pauli channel E_p to wires in the 1+2M-qubit full state."""
    if p==0:return full
    n=1+2*M
    for wire in wires:
        temp=(1-p)*full
        for P in 'xyz':
            op=embed_pauli(P,wire,n)
            temp+=(p/3)*op@full@dag(op)
        full=temp
    return full


def vcp2_branch(x,gate_wires,p,M,model:Noise):
    """Order-two physical VCP ancilla gadget on UNNORMALIZED positive input x.

    Assumes the Pauli channel acts on gate_wires. Both registers experience
    independent Kraus noise E_p on those wires. Return physical outcomes +/-
    AFTER assignment flips. Summing them is physical channel E_p(x).
    """
    d=1<<M
    if x.shape!=(d,d):raise ValueError('Bad input dimension')
    aux=np.eye(d,dtype=complex)/d
    state=tensor(PLUS_DM,x,aux)
    state=controlled_register_swap(state,M,model)
    state=_local_pauli_noise(state,tuple(1+w for w in gate_wires),p,M)
    state=_local_pauli_noise(state,tuple(1+M+w for w in gate_wires),p,M)
    state=controlled_register_swap(state,M,model)
    # X basis control, trace B, just as for the state SWAP instrument.
    blocks=state.reshape((2,d,d,2,d,d))
    diag=np.trace(blocks[0,:,:,0,:,:]+blocks[1,:,:,1,:,:],axis1=1,axis2=3)
    offdiag=np.trace(blocks[0,:,:,1,:,:]+blocks[1,:,:,0,:,:],axis1=1,axis2=3)
    pstate=(diag+offdiag)/2
    mstate=(diag-offdiag)/2
    return observed_branches((pstate+dag(pstate))/2,
                             (mstate+dag(mstate))/2,model.readout_q)


def simulate_layerwise_vcp2(M,p,b_repeats,model):
    """Apply ideal U for each gate then a VCP2 instrument to its Pauli noise.

    Propagate UNNORMALIZED parity-resolved states through the entire circuit.
    This is crucial: inserting the normalized weighted operator as a physical
    next-cycle state would incorrectly omit the signed-record sampling cost.
    """
    gatesA,gatesB=algorithm_blocks(M,b_repeats)
    allgates=gatesA+gatesB
    d=1<<M
    initial=np.zeros((d,d),complex);initial[0,0]=1
    states={1:initial,-1:np.zeros_like(initial)}
    for index,gate in enumerate(allgates):
        u=site_unitary(gate.matrix,gate.wires,M)
        nextstates={1:np.zeros_like(initial),-1:np.zeros_like(initial)}
        for old_sign in (1,-1):
            x=u@states[old_sign]@dag(u)
            plus,minus=vcp2_branch(x,gate.wires,p,M,model)
            nextstates[old_sign]+=plus
            nextstates[-old_sign]+=minus
        states=nextstates
    R=states[1]+states[-1]
    W=states[1]-states[-1]
    return (R+dag(R))/2,(W+dag(W))/2,len(allgates)


def noisy_algorithm_pauli(M,p,b_repeats):
    """Conventional output-only state VD sees the *same* Pauli noise at every gate."""
    gatesA,gatesB=algorithm_blocks(M,b_repeats)
    gates=gatesA+gatesB
    d=1<<M
    ground=np.zeros((d,d),complex);ground[0,0]=1
    state=ground.copy()
    target=ground.copy()
    for gate in gates:
        u=site_unitary(gate.matrix,gate.wires,M)
        target=u@target@dag(u)
        state=u@state@dag(u)
        state=apply_ideal_pauli_channel(state,gate.wires,p,M,power=1)
    return state,target,len(gates)


def compare_vcp2_statevd(M,p,repeats,noise_model):
    """Report matched input algorithm and honest accounting of all channels.

    Two target-state preparations for state VD2 vs one for layerwise VCP2;
    both consume two noisy channel copies per algorithm gate. However,
    layerwise VCP2 also consumes 2*M native Fredkins per gate, whereas
    state VD2 needs only M Fredkins after TWO complete noisy circuits.
    """
    rho,target,G=noisy_algorithm_pauli(M,p,repeats)
    out=[]
    for arch in ('tree_sequential','coherent_request'):
        R,W=simulate(rho,2,arch,noise_model)
        met=metrics(R,W,target,1.)
        out.append(dict(protocol='stateVD2_'+arch,M=M,N=2,b_repeats=repeats,p_channel=p,
                        gates_in_algorithm=G,source_state_preparations=2,
                        noisy_channel_calls=2*G,algorithm_unitary_gate_calls=2*G,
                        native_fredkins=M,ancilla_measurements=1,
                        peak_qubits=2*M+1,**met))
    R,W,ng=simulate_layerwise_vcp2(M,p,repeats,noise_model)
    met=metrics(R,W,target,1.)
    out.append(dict(protocol='layerwise_physical_VCP2',M=M,N=2,b_repeats=repeats,p_channel=p,
                    gates_in_algorithm=G,source_state_preparations=1,
                    noisy_channel_calls=2*G,algorithm_unitary_gate_calls=G,
                    native_fredkins=2*M*G,ancilla_measurements=G,
                    peak_qubits=2*M+1,**met))
    return out
