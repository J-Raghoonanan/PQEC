"""PQES / virtual-distillation physics engine (independent of the legacy code).

THE QUANTUM INSTRUMENT, NOT AN IMAGINARY DETERMINISTIC RHO-SQUARING GATE
============================================================================
This engine treats the quantum processing as a completely positive instrument.
A parity-resolved run returns *unnormalized*, positive operators tau[+/-] on
one surviving M-qubit register. Their sum R is the physical reduced state and
their difference W is the signed virtual state. Ideally

               R = rho,                    W = rho**N.

The normalized virtual observable is Tr(O W)/Tr(W), which is NOT, in general,
a prepared quantum state. Computational squaring is used only for ground truth.

Four physically specified schedules use the same primitive controlled-SWAP:
  tree_parallel    balanced, measurements and disposal at each node;
  tree_recycled    balanced depth-first, left subtree waits for right subtree;
  tree_sequential  measured SWAP accumulator with a fresh copy each step;
  coherent_request coherent-control reset/streaming cyclic-permutation VD.

The final architecture keeps one coherent control qubit through all swaps;
the measured architectures reset/read the ancilla at each node. This is a
specific REQUEST-like circuit realization, NOT every existing REQUEST variant.

Each Fredkin acts on (control, A[j], B[j]); a logical register swap involves M
Fredkins. Noise after Fredkin is a three-qubit depolarizing channel; dephasing
of control and stored data is independently configurable. These are explicitly
stated native-gate assumptions, not a physical calibration for any device.

No external code from the user's earlier repository is imported.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1.,-1.]).astype(complex)
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
PLUS=np.array([[1.],[1.]])/np.sqrt(2)
PLUS_DM=PLUS@PLUS.T
PROJ0=np.diag([1.,0.]).astype(complex)


def dag(a): return a.conj().T

def tensor(*ops):
    y=np.array([[1.]],complex)
    for op in ops:y=np.kron(y,op)
    return y


def normalized_density(a, atol=1e-9):
    a=np.asarray(a,complex)
    assert a.ndim==2 and a.shape[0]==a.shape[1]
    assert abs(np.trace(a)-1)<atol, f'Invalid trace {np.trace(a)}'
    assert np.linalg.norm(a-dag(a)) < atol
    assert np.linalg.eigvalsh((a+dag(a))/2).min()>-atol
    return (a+dag(a))/2


def pure_dm(vec):
    v=np.asarray(vec,complex).reshape(-1)
    v=v/np.linalg.norm(v)
    return np.outer(v,v.conj())


def site_unitary(U:np.ndarray,targets:tuple[int,...],n:int)->np.ndarray:
    """Embed a k-qubit operator in BIG-ENDIAN wire numbering, arbitrary order.

    E.g. (control, A[j], B[j]) targets=(0, 1+j, 1+M+j).
    The operator's first tensor factor acts on targets[0]. Only n<=5 is used.
    """
    targets=tuple(targets)
    k=len(targets); dim=1<<n
    if len(set(targets))!=k or max(targets,default=0)>=n:raise ValueError(targets)
    if U.shape!=(1<<k,1<<k):raise ValueError((U.shape,targets))
    out=np.zeros((dim,dim),complex)
    masks=[1<<(n-1-t) for t in targets]
    for col in range(dim):
        local_in=sum(int(bool(col&m))<<(k-1-j) for j,m in enumerate(masks))
        for local_out in range(1<<k):
            row=col
            for j,m in enumerate(masks):
                row= (row | m) if local_out & (1<<(k-1-j)) else (row & ~m)
            out[row,col]=U[local_out,local_in]
    return out


@lru_cache(maxsize=128)
def embed_pauli(kind:str,wire:int,n:int):
    return site_unitary({'x':X,'y':Y,'z':Z,'h':H}[kind],(wire,),n)


def partial_trace(rho:np.ndarray, keep:tuple[int,...],n:int)->np.ndarray:
    """Trace out all wires except requested ordered wires, without qiskit."""
    keep=tuple(keep)
    if len(set(keep))!=len(keep):raise ValueError(keep)
    from string import ascii_letters
    if n>25:raise ValueError('Explicit matrix backend limited to <=25 wires')
    symbols=ascii_letters
    a=list(symbols[:n]); b=list(symbols[n:2*n]);
    for j in range(n):
        if j not in keep:b[j]=a[j]
    right=''.join(a[j] for j in keep)+''.join(b[j] for j in keep)
    return np.einsum(''.join(a+b)+'->'+right,rho.reshape((2,)*(2*n)),optimize=True).reshape((1<<len(keep),)*2)


def depolarize_subset(rho:np.ndarray, wires:tuple[int,...],n:int,p:float):
    """D_p(X)=(1-p)X+p (I/d)_{wires} ⊗ Tr_wires X.

    Works on *unnormalized* operators and ancilla/data coherences as well.
    This is a physical CPTP map on the specified subsystem, even when rho
    contains off-diagonal control coherences.
    """
    if p==0:return rho
    wires=tuple(sorted(wires));others=tuple(q for q in range(n) if q not in wires)
    reduced=partial_trace(rho,others,n)
    raw=np.kron(reduced,np.eye(1<<len(wires),dtype=complex)/(1<<len(wires)))
    # raw order is [others, wires]; permutation into original n-wire order.
    order=others+wires; position={old:i for i,old in enumerate(order)}
    perm=[position[q] for q in range(n)]
    mat=raw.reshape((2,)*(2*n)).transpose(perm+[n+j for j in perm]).reshape(rho.shape)
    return (1-p)*rho+p*mat


def phase_flip(rho, wires:tuple[int,...],n:int,p:float):
    """Independent stochastic Z errors with probability p; |coherence| factor 1-2p."""
    if p==0:return rho
    for wire in wires:
        z=embed_pauli('z',wire,n)
        rho=(1-p)*rho+p*(z@rho@z)
    return rho


def depolarize_qubits(rho,p,n):
    for i in range(n):rho=depolarize_subset(rho,(i,),n,p)
    return rho


def spectral_power(rho,N:int):
    """IDEAL virtual mathematical target; NEVER interpreted as a CPTP gate."""
    vals,vec=np.linalg.eigh(rho)
    vals=np.clip(vals,0,None)**N
    if vals.sum()<=0:raise ValueError('Numerical underflow in spectral power')
    return (vec*vals)@dag(vec)/vals.sum()


def prep_target(M:int,kind='plus'):
    """Return known ideal statevector. Big-endian wires 0..M-1."""
    if M<1 or M>3:raise ValueError('M=1..3 for this explicit-small-register code')
    if kind=='plus':return tensor(*([PLUS]*M)).reshape(-1)
    if kind=='tilted':
        v=np.array([np.cos(np.pi/6),np.exp(1j*np.pi/4)*np.sin(np.pi/6)])
        # Build a true tensor-product STATEVECTOR, not a classically cloned state.
        state=v.copy()
        for _ in range(M-1):
            state=np.kron(state,v)
        return state
    if kind=='ghz':
        v=np.zeros(1<<M,complex);v[0]=v[-1]=1/np.sqrt(2)
        if M==1:return PLUS.reshape(-1)
        return v
    if kind=='bell_tilted':
        if M!=2:raise ValueError('bell_tilted requires M=2')
        v=np.zeros(4,complex);v[0]=np.cos(np.pi/6);v[3]=np.sin(np.pi/6)
        return v
    if kind=='entangled':
        if M==1:return np.array([np.cos(.42),np.exp(1j*.35)*np.sin(.42)],complex)
        v=np.array([np.cos(.42),0,0,np.exp(1j*.35)*np.sin(.42)],complex)
        if M==2:return v
        return np.kron(v,PLUS.reshape(-1))
    raise ValueError(kind)


def noisy_input(M,kind,p_input=.12,input_noise='local_depol'):
    """Prepare an ideal benchmark target, then apply a specified local CPTP noise.
    For depol: (1-p)rho + p/3 sum_{P=X,Y,Z} P rho P, per qubit.
    For dephase: (1-p)rho + p Z rho Z, per qubit.
    p_input is NOT the SWAP-gate noise probability.
    """
    psi=prep_target(M,kind)
    rho=pure_dm(psi)
    for wire in range(M):
        if input_noise=='local_depol':
            rr=(1-p_input)*rho
            for j in 'xyz':
                P=embed_pauli(j,wire,M)
                rr+=(p_input/3)*P@rho@P
            rho=rr
        elif input_noise=='dephase_z':rho=phase_flip(rho,(wire,),M,p_input)
        elif input_noise=='none':pass
        else:raise ValueError(input_noise)
    return normalized_density(rho), pure_dm(psi)


@lru_cache(maxsize=1)
def fredkin_3():
    f=np.eye(8,dtype=complex)
    f[5,5]=f[6,6]=0
    f[5,6]=f[6,5]=1
    return f


@lru_cache(maxsize=8)
def fredkin_matrix(M:int,j:int):
    return site_unitary(fredkin_3(),(0,1+j,1+M+j),1+2*M)


@dataclass(frozen=True)
class Noise:
    # Zero is the exact ideal apparatus, regardless of input state's noise.
    gate_p:float=0.0      # after EACH native 3-qubit Fredkin (not per register)
    control_z_p:float=0.0 # after EACH Fredkin
    readout_q:float=0.0   # independent bit-flip on each recorded instrument outcome
    memory_z_p:float=0.0  # per stored DATA qubit per elementary SWAP gadget time unit
    control_idle_p:float=0.0 # EXTRA coherent-control Z error between register swaps
    reset_p:float=0.0     # imperfect re-preparation after recycling one data register

    def __post_init__(self):
        for val in vars(self).values():
            if not 0<=val<=.5:raise ValueError('noise parameters must be between 0 and 0.5')


def controlled_register_swap(state,M,model:Noise):
    """Apply M native Fredkins on control/A/B plus specified local noise.

    The entire physical state is (1+2M) qubits, with explicit data/control
    correlations. Noise is applied BETWEEN elementary Fredkins, not after
    a fictitious one-step register-SWAP gate. This distinction matters M>1.
    """
    n=1+2*M
    for j in range(M):
        F=fredkin_matrix(M,j)
        state=F@state@dag(F)
        if model.gate_p:state=depolarize_subset(state,(0,1+j,1+M+j),n,model.gate_p)
        if model.control_z_p:state=phase_flip(state,(0,),n,model.control_z_p)
    return state


def observed_branches(actual_plus,actual_minus,q):
    return ((1-q)*actual_plus+q*actual_minus,
            q*actual_plus+(1-q)*actual_minus)


def measure_swap(left,right,model:Noise):
    """Actual nondestructive SWAP instrument with physically measured ancilla.

    Input left,right may be unnormalized positive matrices. Return two
    unnormalized physical states of A with *reported* +/- sign; readout flips
    do not affect the underlying physical state, just its classical label.
    """
    d=left.shape[0];M=int(round(math.log2(d)))
    full=tensor(PLUS_DM,left,right)
    full=controlled_register_swap(full,M,model)
    blk=full.reshape((2,d,d,2,d,d))
    diag=np.trace(blk[0,:,:,0,:,:]+blk[1,:,:,1,:,:],axis1=1,axis2=3)
    cross=np.trace(blk[0,:,:,1,:,:]+blk[1,:,:,0,:,:],axis1=1,axis2=3)
    # X-control projectors. For perfect input, tau+/-=(rho +/- rho^2)/2.
    p=(diag+cross)/2; m=(diag-cross)/2
    return observed_branches((p+dag(p))/2,(m+dag(m))/2,model.readout_q)


def _as_parity(state):
    return {1:state,-1:np.zeros_like(state)}


def _merge_parity(left,right,model):
    out={1:np.zeros_like(left[1]),-1:np.zeros_like(left[1])}
    for l in (1,-1):
        for r in (1,-1):
            p,m=measure_swap(left[l],right[r],model)
            out[l*r]+=p
            out[-l*r]+=m
    return out


def _memory_on_parities(par,model,wait_steps):
    if model.memory_z_p==0 or wait_steps==0:return par
    # Independent Markovian Z flip repeated for wait_steps; composition exact.
    p_eff=(1-(1-2*model.memory_z_p)**wait_steps)/2
    M=int(round(math.log2(par[1].shape[0])))
    return {s:phase_flip(par[s],tuple(range(M)),M,p_eff) for s in (1,-1)}


def reset_input(rho,model,active):
    """Model imperfect reset-before-preparation: mixture with |0..0><0..0|.
    This phenomenological replacement is NOT equivalent to arbitrary gate noise.
    It is applied only when a register is reused to prepare a fresh copy.
    """
    if not active or not model.reset_p:return rho
    ground=np.zeros_like(rho);ground[0,0]=1
    return (1-model.reset_p)*rho+model.reset_p*ground


def run_measured(rho,N:int,architecture:str,model:Noise):
    """Return (R,W) for a measured SWAP schedule, including noisy branches.

    Tree shape matters under memory/reset/gate noise; ideal coarse-grained
    outcome (R,W) is independent of shape for identical input states.
    """
    if N<1:raise ValueError('N>=1')
    if architecture not in ('tree_parallel','tree_recycled','tree_sequential'):
        raise ValueError(architecture)
    if architecture!='tree_sequential' and N&(N-1):
        raise ValueError('Balanced tree N must be power of two')
    counter=[0]
    def leaf():
        reuse= counter[0]>0 and architecture!='tree_parallel'
        counter[0]+=1
        return _as_parity(reset_input(rho,model,reuse))
    if architecture=='tree_sequential':
        accumulator=leaf()
        for j in range(1,N):
            accumulator=_memory_on_parities(accumulator,model,1 if j>1 else 0)
            accumulator=_merge_parity(accumulator,leaf(),model)
    else:
        def build(count):
            if count==1:return leaf(),0
            L,dleft=build(count//2)
            R,dright=build(count//2)
            # Depth-first storage: the left computed subtree is held while
            # all operations in right subtree execute. Parallel mode avoids it.
            if architecture=='tree_recycled':L=_memory_on_parities(L,model,dright)
            out=_merge_parity(L,R,model)
            return out, 1+(dleft+dright if architecture=='tree_recycled' else max(dleft,dright))
        accumulator,_=build(N)
    R=accumulator[1]+accumulator[-1]
    W=accumulator[1]-accumulator[-1]
    return (R+dag(R))/2,(W+dag(W))/2


def run_coherent_request(rho,N:int,model:Noise):
    """One COHERENT ancilla, sequential stream/reset, ONE final X measurement.

    Proof sketch: starting from |+><+|_c ⊗ rho_A, each new independent rho_B
    is controlled-swapped with A and B is discarded. The ancilla off-diagonal
    block evolves rho^k -> rho^(k+1); after N-1 steps the final X outcome has
    tau+/-=(rho +/- rho^N)/2 in the ideal apparatus. We calculate all matrix
    blocks explicitly under noise rather than substituting that ideal identity.
    """
    if N<1:raise ValueError('N>=1')
    M=int(round(math.log2(rho.shape[0])));d=1<<M
    state=np.kron(PLUS_DM,rho) # (control,A)
    for step in range(1,N):
        if step>1:
            # extra coherent ancilla idle between measured-module durations
            if model.control_idle_p:state=phase_flip(state,(0,),1+M,model.control_idle_p)
            if model.memory_z_p:state=phase_flip(state,tuple(range(1,M+1)),1+M,model.memory_z_p)
        fresh=reset_input(rho,model,True)
        full=np.kron(state,fresh)
        full=controlled_register_swap(full,M,model)
        state=partial_trace(full,tuple(range(1+M)),1+2*M)
    blk=state.reshape((2,d,2,d))
    diag=blk[0,:,0,:]+blk[1,:,1,:]
    cross=blk[0,:,1,:]+blk[1,:,0,:]
    p=(diag+cross)/2;m=(diag-cross)/2
    obsplus,obsminus=observed_branches((p+dag(p))/2,(m+dag(m))/2,model.readout_q)
    return (obsplus+obsminus+dag(obsplus+obsminus))/2,(obsplus-obsminus+dag(obsplus-obsminus))/2


def simulate(rho,N,architecture,model):
    if architecture=='coherent_request':return run_coherent_request(rho,N,model)
    return run_measured(rho,N,architecture,model)


def metrics(R,W,O,ideal_expect):
    """Unified *noisy-instrument* bias and leading shot-noise coefficient.

    Let Omega=+/- be the ACTUALLY REPORTED parity. tau_Omega are positive.
       R=tau_+ + tau_- ; W=tau_+ - tau_- ; t=Tr(W)
       mu=Tr(O W)/t ; v=[Tr(O^2 R)-2mu Tr(O R)+mu^2]/t^2
    The unconditional measurement of O is on R; no physical rho**N is assumed.
    If t≈0, ratio estimation is numerically ill conditioned; return NaNs.
    """
    trR=float(np.trace(R).real)
    trW=float(np.trace(W).real)
    if abs(trR-1)>1e-7:raise AssertionError(f'non-trace preserving {trR}')
    if np.linalg.eigvalsh(R).min()<-1e-8:raise AssertionError('R not PSD')
    if abs(trW)<1e-11:
        return dict(normalization=trW,estimate=np.nan,bias=np.nan,variance_coeff=np.inf,
                    ordinary_expect=float(np.trace(O@R).real),ratio_bias_coeff=np.nan)
    mu=float((np.trace(O@W)/trW).real)
    ordinary=float(np.trace(O@R).real)
    moment2=float(np.trace((O@O)@R).real)
    numer=moment2-2*mu*ordinary+mu**2
    if numer< -1e-8:raise AssertionError('negative variance')
    v=max(numer,0)/trW**2
    return dict(normalization=trW,estimate=mu,bias=mu-ideal_expect,variance_coeff=v,
                ordinary_expect=ordinary,ratio_bias_coeff=(mu-ordinary)/trW**2)


def shot_requirements(row,epsilon=.1,confidence=.95):
    """Leading delta-method threshold, NOT an exact finite-shot guarantee.
    MSE approx bias**2 + [v +2*bias*ratio_bias_coeff]/shots.
    Conservative implementation figures show v-only criterion and flag when
    asymptotic denominator is poorly resolved. Simulation validates ratios.
    """
    b=row['bias'];v=row['variance_coeff']
    if not np.isfinite(v) or abs(b)>=epsilon:return np.inf
    return max(1,math.ceil(v/(epsilon**2-b*b)))


def resources(M,N,architecture,model):
    """Explicit simple clock model: each register-level SWAP is one unit.
    Width values count control ancillas but exclude compilation ancillas.
    Parallel stage count=log2 N; other variants schedule N-1 stages.
    At fixed total copy count, PREPARATION COUNT is N for all protocols.
    Counts assume M native 3q Fredkins per register exchange.
    """
    l=math.ceil(math.log2(N)) if N>1 else 0
    width={'tree_parallel':M*N+N//2 if N>1 else M,
           'tree_recycled':M*(l+1)+1 if N>1 else M,
           'tree_sequential':2*M+1 if N>1 else M,
           'coherent_request':2*M+1 if N>1 else M}[architecture]
    depth=l if architecture=='tree_parallel' else max(N-1,0)
    return dict(peak_qubits=width,register_swaps=N-1,native_fredkins=M*(N-1),
                controls_measured=N-1 if architecture.startswith('tree_') else (1 if N>1 else 0),
                fresh_preparations=N,critical_swap_stages=depth,
                approximate_gate_time=M*depth,
                reset_events=(N-1 if architecture!='tree_parallel' else 0))


def sample_ratio(R,W,O,shots,seed=1):
    """Physical +/- instrument data plus final observable measurement.
    Generate outcomes using tau±=(R±W)/2 and observable spectral projectors.
    Single-shot observable values are sampled, NEVER replaced by conditional means.
    """
    rng=np.random.default_rng(seed)
    eigenvalues,U=np.linalg.eigh(O)
    probabilities=[];results=[]
    for parity in (1,-1):
        branch=(R+parity*W)/2
        p=np.real(np.diag(dag(U)@branch@U))
        for value,prob in zip(eigenvalues,p):
            probabilities.append(max(0,prob));results.append((parity,float(value.real)))
    probabilities=np.array(probabilities,float)
    probabilities/=probabilities.sum()
    sample=rng.choice(len(results),size=shots,p=probabilities)
    s=np.array([results[i][0] for i in sample],float)
    outcomes=np.array([results[i][1] for i in sample],float)
    den=s.sum()
    return dict(empirical=float(np.dot(s,outcomes)/den) if den!=0 else np.nan,
                signed_normalization=float(den/shots),shots=shots,
                zero_denominator=bool(den==0))
