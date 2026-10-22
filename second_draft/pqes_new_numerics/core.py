"""PQES / virtual-distillation physics engine (independent of the legacy code).

THE QUANTUM INSTRUMENT, NOT AN IMAGINARY DETERMINISTIC RHO-SQUARING GATE
============================================================================
The quantum processing is a completely positive instrument. A parity-resolved
run returns *unnormalized*, positive operators tau[+/-] on the measured data
register(s). Their sum R is the physical reduced state and their difference W
is the signed virtual state. Ideally R = rho and W = rho**N. The normalized
virtual observable Tr(O W)/Tr(W) is NOT a prepared quantum state.

SIX ARCHITECTURES, ONE HARDWARE MODEL, ONE CLOCK
================================================
  tree_parallel     balanced tree, N registers, N/2 reusable SWAP ancillas
  tree_recycled     balanced depth-first tree, l+1 registers, 1 ancilla
  tree_sequential   measured SWAP accumulator, 2 registers, 1 ancilla
  coherent_request  one coherent control, streaming: B is reset and reprepared
  wide_cyclic       one coherent control, all N registers present, no resets
  derangement       wide_cyclic circuit; O measured on ALL N registers and
                    averaged (cyclic-derangement estimator, Koczor 2021). Each
                    spent register is read out mid-circuit right after its
                    CSWAP (t_meas); observable readouts are ideal in all arms

Every register-level controlled SWAP is COMPILED: CSWAP(c;a,b) =
CNOT(b->a) CCX(c,a->b) CNOT(b->a). For a register of M pairs the CNOTs of
different pairs commute, so the gadget is one CNOT layer, M sequential
Toffolis sharing the control, and one CNOT layer (M+2 time units). After each
CNOT the draft's two-qubit depolarizing channel D_eps2 (Eq.
noisy_entangling_channel) acts on (a,b); after each Toffoli the same-form
three-qubit channel D_eps3 acts on (c,a,b). Every qubit that waits in a time
unit - data or control, inside or between gadgets - receives the same idle
channel: amplitude damping idle_gamma and Z flip idle_phi per unit.

THE CLOCK (one rule set for all arms; see Schedule below)
  * gate = 1 unit; a fresh source preparation takes t_prep units (or the depth
    of the actual source circuit); mid-circuit measurement t_meas; reset t_reset.
  * The first use of any register (data or ancilla) starts from the perfect
    initial reset at t=0; every actual mid-circuit REUSE pays t_reset and the
    reset error (data_reset_p / ancilla_reset_p).
  * Preparations (and ancilla |+> preparations: reset + H, one extra unit) are
    scheduled just in time, so a fresh leaf never idles; a merged survivor
    that has to wait does idle. A first-use ancilla is ready at t=0.
  * PQES needs no feed-forward: a surviving register never waits for its own
    ancilla's measurement. Only ancilla REUSE (measure + reset) blocks the next
    gadget that needs that ancilla.

No external code from the user's earlier repository is imported.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import math
import numpy as np

I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1.,-1.]).astype(complex)
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
S_GATE=np.diag([1,1j]).astype(complex)
PLUS=np.array([[1.],[1.]])/np.sqrt(2)
PLUS_DM=(PLUS@PLUS.T).astype(complex)
MINUS_DM=np.array([[.5,-.5],[-.5,.5]],complex)
PROJ0=np.diag([1.,0.]).astype(complex)
CNOT_M=np.eye(4,dtype=complex);CNOT_M[2:,2:]=X   # control = first listed wire
CCX_M=np.eye(8,dtype=complex);CCX_M[6:,6:]=X      # controls = first two wires

ARCH=('tree_parallel','tree_recycled','tree_sequential','coherent_request','wide_cyclic','derangement')
MEASURED=('tree_parallel','tree_recycled','tree_sequential')


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

    The operator's first tensor factor acts on targets[0]. Dense reference
    embedding (used for small registers and as an independent check of the
    tensor-contraction routines below).
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


def conj_local(rho,U,wires,n):
    """U_wires rho U_wires^dagger by tensor contraction (no dense embedding)."""
    wires=list(wires);k=len(wires)
    t=np.asarray(rho,complex).reshape((2,)*(2*n))
    Ut=np.asarray(U,complex).reshape((2,)*(2*k))
    t=np.tensordot(Ut,t,axes=(list(range(k,2*k)),wires))
    t=np.moveaxis(t,list(range(k)),wires)
    t=np.tensordot(Ut.conj(),t,axes=(list(range(k,2*k)),[n+w for w in wires]))
    t=np.moveaxis(t,list(range(k)),[n+w for w in wires])
    return t.reshape(rho.shape)


def kraus_local(rho,ops,wires,n):
    out=np.zeros_like(rho,dtype=complex)
    for K in ops:out=out+conj_local(rho,K,wires,n)
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
    """D_p(X)=(1-p)X+p (I/d)_{wires} (x) Tr_wires X  (draft Eq. noisy_entangling_channel).

    Works on *unnormalized* operators and ancilla/data coherences as well.
    """
    if p==0:return rho
    wires=tuple(sorted(wires));others=tuple(q for q in range(n) if q not in wires)
    reduced=partial_trace(rho,others,n)
    raw=np.kron(reduced,np.eye(1<<len(wires),dtype=complex)/(1<<len(wires)))
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


@lru_cache(maxsize=256)
def idle_kraus(units:int,gamma:float,phi:float):
    """Exact composition of `units` idle steps: amplitude damping then Z flip.

    The two channels commute, and each composes in closed form:
    gamma_eff=1-(1-gamma)**k,  phi_eff=(1-(1-2 phi)**k)/2.
    """
    g=1-(1-gamma)**units; f=(1-(1-2*phi)**units)/2
    K0=np.diag([1,np.sqrt(1-g)]).astype(complex)
    K1=np.array([[0,np.sqrt(g)],[0,0]],complex)
    ops=[np.sqrt(1-f)*K0,np.sqrt(1-f)*K1,np.sqrt(f)*Z@K0,np.sqrt(f)*Z@K1]
    return tuple(K for K in ops if np.abs(K).max()>0)


def spectral_power(rho,N:int):
    """IDEAL virtual mathematical target; NEVER interpreted as a CPTP gate."""
    vals,vec=np.linalg.eigh(rho)
    vals=np.clip(vals,0,None)**N
    if vals.sum()<=0:raise ValueError('Numerical underflow in spectral power')
    return (vec*vals)@dag(vec)/vals.sum()


def prep_target(M:int,kind='plus'):
    """Return known ideal statevector. Big-endian wires 0..M-1."""
    if M<1 or M>5:raise ValueError('M=1..5 for this explicit-small-register code')
    if kind=='plus':return tensor(*([PLUS]*M)).reshape(-1)
    if kind=='tilted':
        v=np.array([np.cos(np.pi/6),np.exp(1j*np.pi/4)*np.sin(np.pi/6)])
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
        return np.kron(v,tensor(*([PLUS]*(M-2))).reshape(-1))
    raise ValueError(kind)


def noisy_input(M,kind,p_input=.12,input_noise='local_depol'):
    """Prepare an ideal benchmark target, then apply a specified local CPTP noise.
    For depol: (1-p)rho + p/3 sum_{P=X,Y,Z} P rho P, per qubit (Pauli form,
    manuscript Eq. pauli-kraus; recorded in CSVs as `source_pauli_p`).
    For dephase: (1-p)rho + p Z rho Z, per qubit.
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


@dataclass(frozen=True)
class Hardware:
    """All apparatus imperfections and durations, shared by every architecture.

    Gate noise uses ONE convention: the replacement depolarizing channel
    D_eps(X)=(1-eps)X+eps I/d (x) Tr X on the gate's wires (draft Eq.
    noisy_entangling_channel). For one qubit, eps corresponds to the Pauli-form
    probability p=3 eps/4.
    """
    eps1:float=0.0            # after every one-qubit gate (incl. ancilla H)
    eps2:float=0.0            # after every two-qubit gate (CNOT/CZ)
    eps3:float|None=None      # after every Toffoli on (c,a,b); None -> eps2
    readout_q:float=0.0       # independent flip of each recorded ancilla outcome
    ancilla_reset_p:float=0.0 # failed mid-circuit ancilla reset -> |-> instead of |+>
    data_reset_p:float=0.0    # failed mid-circuit data reset: fresh copy -> |0..0> (phenomenological)
    idle_gamma:float=0.0      # amplitude damping per idle unit, every waiting qubit
    idle_phi:float=0.0        # Z flip per idle unit, every waiting qubit
    t_prep:int=3              # duration of one abstract source preparation
    t_meas:int=1              # mid-circuit measurement duration
    t_reset:int=1             # mid-circuit reset duration

    def __post_init__(self):
        for key in ('eps1','eps2','readout_q','ancilla_reset_p','data_reset_p','idle_gamma','idle_phi'):
            val=getattr(self,key)
            if not 0<=val<=.5:raise ValueError(f'{key} must be between 0 and 0.5')
        if self.eps3 is not None and not 0<=self.eps3<=.5:raise ValueError('eps3')
        for key in ('t_prep','t_meas','t_reset'):
            if int(getattr(self,key))!=getattr(self,key) or getattr(self,key)<0:raise ValueError(key)

    @property
    def e3(self):return self.eps2 if self.eps3 is None else self.eps3

    def record(self):
        d=asdict(self);d['eps3']=self.e3;return d


@dataclass
class Ledger:
    """Per-shot schedule and resource record of one architecture run."""
    duration:int=0          # makespan including the final measurement
    data_idle:int=0         # sum over data qubits of idle units
    control_idle:int=0      # sum over control/ancilla qubits of idle units
    preparations:int=0      # fresh source-state preparations (copies)
    data_resets:int=0       # mid-circuit data-register reuses
    ancilla_resets:int=0    # mid-circuit ancilla reuses
    measurements:int=0      # recorded mid-circuit outcomes (ancillas; spent derangement registers)
    cnot:int=0
    ccx:int=0
    gates1:int=0
    peak_qubits:int=0

    def record(self,prefix=''):return {prefix+k:v for k,v in asdict(self).items()}


def apply_idle(rho,wires,n,units,hw:Hardware):
    if units<=0 or not wires or (hw.idle_gamma==0 and hw.idle_phi==0):return rho
    ops=idle_kraus(int(units),float(hw.idle_gamma),float(hw.idle_phi))
    for w in wires:rho=kraus_local(rho,ops,(w,),n)
    return rho


def gate_noise(rho,wires,n,hw:Hardware,eps=None):
    k=len(wires)
    p={1:hw.eps1,2:hw.eps2,3:hw.e3}[k] if eps is None else eps
    return depolarize_subset(rho,tuple(wires),n,p) if p else rho


def cswap_gadget(state,n,control,A,B,hw:Hardware,ledger:Ledger|None=None):
    """Compiled controlled register SWAP on an explicit n-qubit state.

    CNOT layer (b_j -> a_j for all j), M Toffolis CCX(c,a_j->b_j) in sequence,
    CNOT layer. Waiting qubits inside the gadget get the idle channel each unit.
    Returns (state, duration).
    """
    M=len(A);pairs=list(zip(A,B));data=list(A)+list(B)
    def cnot_layer(st):
        for a,b in pairs:
            st=conj_local(st,CNOT_M,(b,a),n)
            st=gate_noise(st,(b,a),n,hw,hw.eps2)
        return apply_idle(st,[control],n,1,hw)
    state=cnot_layer(state)
    for j,(a,b) in enumerate(pairs):
        state=conj_local(state,CCX_M,(control,a,b),n)
        state=gate_noise(state,(control,a,b),n,hw,hw.e3)
        state=apply_idle(state,[q for q in data if q not in (a,b)],n,1,hw)
    state=cnot_layer(state)
    if ledger is not None:gadget_counts(M,hw,ledger)
    return state,M+2


def cswap_duration(M,hw:Hardware):
    return M+2


def gadget_counts(M,hw:Hardware,ledger:Ledger):
    """Add one register-CSWAP gadget's gates and in-gadget idle units to `ledger`."""
    ledger.cnot+=2*M;ledger.ccx+=M;ledger.control_idle+=2;ledger.data_idle+=2*M*(M-1)


def ancilla_plus(hw:Hardware,reused:bool):
    """|+> after (re)initialization: failed reset flips it to |->; H carries eps1."""
    r=hw.ancilla_reset_p if reused else 0.
    a=(1-r)*PLUS_DM+r*MINUS_DM
    return (1-hw.eps1)*a+hw.eps1*I2/2


def observed_branches(actual_plus,actual_minus,q):
    return ((1-q)*actual_plus+q*actual_minus,
            q*actual_plus+(1-q)*actual_minus)


def reset_input(rho,model,active):
    """Imperfect mid-circuit data reset before re-preparation (phenomenological):
    with probability data_reset_p the fresh copy is replaced by |0..0><0..0|.
    Applied ONLY when a register is actually reused."""
    p=getattr(model,'data_reset_p',0.)
    if not active or not p:return rho
    ground=np.zeros_like(rho,dtype=complex);ground[0,0]=1
    return (1-p)*rho+p*ground


def _x_branches(state,keep_dims,trace_dims=None):
    """Control is wire 0 of `state`; return (tau+, tau-) on the remaining wires
    (tracing the trailing `trace_dims` dimension) after an X measurement."""
    d=keep_dims;e=trace_dims or 1
    blk=state.reshape((2,d,e,2,d,e))
    diag=np.trace(blk[0,:,:,0,:,:]+blk[1,:,:,1,:,:],axis1=1,axis2=3)
    cross=np.trace(blk[0,:,:,1,:,:]+blk[1,:,:,0,:,:],axis1=1,axis2=3)
    p=(diag+cross)/2;m=(diag-cross)/2
    return (p+dag(p))/2,(m+dag(m))/2


def _herm(a):return (a+dag(a))/2


def measure_swap(left,right,model:Hardware):
    """One SWAP gadget on (possibly unnormalized) inputs: reported (tau+, tau-)
    on the surviving register; ideal tau+/-=(X+Y +/- (XY+YX))/4."""
    d=left.shape[0];M=int(round(math.log2(d)));n=1+2*M
    full=tensor(ancilla_plus(model,False),left,right)
    full,_=cswap_gadget(full,n,0,list(range(1,1+M)),list(range(1+M,n)),model)
    p,m=_x_branches(full,d,d)
    return observed_branches(p,m,model.readout_q)





class _Pool:
    """Registers (data or ancilla) with next-free time and first-use flag."""
    def __init__(self,size):self.free=[0]*size;self.used=[False]*size
    def take(self):
        i=min(range(len(self.free)),key=lambda j:(self.free[j],j))
        return i,self.free[i],self.used[i]


def _width(M,N,arch):
    if N<=1:return M
    l=int(math.log2(N)) if arch in ('tree_parallel','tree_recycled') else None
    return {'tree_parallel':M*N+N//2,'tree_recycled':M*(l+1)+1 if l is not None else None,
            'tree_sequential':2*M+1,'coherent_request':2*M+1,
            'wide_cyclic':M*N+1,'derangement':M*N+1}[arch]


def run_measured(rho,N:int,architecture:str,model:Hardware,t_prep=None,ledger=None):
    """Measured SWAP trees on the shared clock. Returns (R,W) on the survivor.

    Each subtree output is a parity-resolved pair of unnormalized operators
    plus its ready time; idle noise is applied lazily to a survivor's pair just
    before the merge that consumes it (the idle channel is local and outcome-
    independent, so this equals applying it in real time).
    """
    hw=model;tp=hw.t_prep if t_prep is None else int(t_prep)
    if N<1:raise ValueError('N>=1')
    if architecture not in MEASURED:raise ValueError(architecture)
    if architecture!='tree_sequential' and N&(N-1):
        raise ValueError('Balanced tree N must be power of two')
    led=ledger if ledger is not None else Ledger()
    M=int(round(math.log2(rho.shape[0])));d=1<<M;n=1+2*M
    nreg={'tree_parallel':N,'tree_sequential':min(N,2),
          'tree_recycled':int(math.log2(N))+1 if N>1 else 1}[architecture]
    nanc={'tree_parallel':max(1,N//2),'tree_sequential':1,'tree_recycled':1}[architecture]
    regs=_Pool(nreg);ancs=_Pool(nanc);Tc=cswap_duration(M,hw)
    led.peak_qubits=_width(M,N,architecture)
    def leaf():
        i,free,used=regs.take()
        regs.free[i]=math.inf;led.preparations+=1
        if used:led.data_resets+=1
        state=reset_input(rho,hw,used)
        return dict(par={1:state.astype(complex),-1:np.zeros((d,d),complex)},
                    ready=free+(hw.t_reset if used else 0)+tp,reg=i,leaf=True)
    def merge(L,R):
        a,afree,aused=ancs.take()
        start=max(L['ready'],R['ready'],afree+((hw.t_reset+1) if aused else 0))
        for node in (L,R):
            wait=start-node['ready']
            if not node['leaf'] and wait>0:
                node['par']={s:apply_idle(node['par'][s],range(M),M,wait,hw) for s in (1,-1)}
                led.data_idle+=M*wait
        if aused:led.ancilla_resets+=1
        anc=ancilla_plus(hw,aused);led.gates1+=1
        out={1:np.zeros((d,d),complex),-1:np.zeros((d,d),complex)}
        for l in (1,-1):
            for r in (1,-1):
                if not np.any(L['par'][l]) or not np.any(R['par'][r]):continue
                full=tensor(anc,L['par'][l],R['par'][r])
                full,_=cswap_gadget(full,n,0,list(range(1,1+M)),list(range(1+M,1+2*M)),hw)
                p,m=_x_branches(full,d,d)
                p,m=observed_branches(p,m,hw.readout_q)
                out[l*r]+=p;out[-l*r]+=m
        gadget_counts(M,hw,led)
        end=start+Tc
        ancs.free[a]=end+hw.t_meas;ancs.used[a]=True
        regs.free[R['reg']]=end;regs.used[R['reg']]=True
        regs.used[L['reg']]=True
        led.measurements+=1
        return dict(par=out,ready=end,reg=L['reg'],leaf=False)
    if architecture=='tree_sequential':
        acc=leaf()
        for j in range(1,N):acc=merge(acc,leaf())
    elif architecture=='tree_parallel':
        level=[leaf() for _ in range(N)]
        while len(level)>1:level=[merge(level[2*i],level[2*i+1]) for i in range(len(level)//2)]
        acc=level[0]
    else:
        def build(count):
            if count==1:return leaf()
            L=build(count//2);R=build(count//2)
            return merge(L,R)
        acc=build(N)
    led.duration=acc['ready']+(hw.t_meas if N>1 else 0)
    R=acc['par'][1]+acc['par'][-1];W=acc['par'][1]-acc['par'][-1]
    return _herm(R),_herm(W)


def run_coherent(rho,N:int,model:Hardware,t_prep=None,ledger=None,wide=False):
    """One coherent control through N-1 compiled CSWAP(c;A,B_k); ONE X readout.

    The product of the transpositions (A B_1)...(A B_{N-1}) is a controlled
    N-cycle, so ideally tau+/-=(rho +/- rho^N)/2. wide=False is the streaming
    (REQUEST-like) circuit with two data registers: B is reset and reprepared
    while A and the control idle. wide=True keeps all N registers (MN+1
    qubits): every B_k is prepared just in time, nothing is reset, and the
    CSWAPs run back to back. In both, a spent B_k is never touched again, so it
    is traced out right after its CSWAP (exact).
    """
    hw=model;tp=hw.t_prep if t_prep is None else int(t_prep)
    if N<1:raise ValueError('N>=1')
    led=ledger if ledger is not None else Ledger()
    M=int(round(math.log2(rho.shape[0])));d=1<<M
    led.peak_qubits=_width(M,N,'wide_cyclic' if wide else 'coherent_request')
    led.preparations=N
    if N==1:
        led.duration=tp
        return _herm(rho.astype(complex)),_herm(rho.astype(complex))
    Tc=cswap_duration(M,hw)
    state=np.kron(ancilla_plus(hw,False),rho.astype(complex));led.gates1+=1
    ready=tp;bfree=0;bused=False
    for k in range(1,N):
        if wide:bready=tp
        else:
            bready=bfree+(hw.t_reset if bused else 0)+tp
            if bused:led.data_resets+=1
        start=max(ready,bready);wait=start-ready
        if wait>0:
            state=apply_idle(state,range(1+M),1+M,wait,hw)
            led.data_idle+=M*wait;led.control_idle+=wait
        fresh=reset_input(rho,hw,bused and not wide).astype(complex)
        full=np.kron(state,fresh)
        full,_=cswap_gadget(full,1+2*M,0,list(range(1,1+M)),list(range(1+M,1+2*M)),hw,led)
        state=partial_trace(full,tuple(range(1+M)),1+2*M)
        ready=start+Tc;bfree=ready;bused=True
    p,m=_x_branches(state,d)
    p,m=observed_branches(p,m,hw.readout_q)
    led.measurements+=1;led.duration=ready+hw.t_meas
    return _herm(p+m),_herm(p-m)


def derangement_moments(Rf,Wf,O,M,N):
    """Register-averaged estimator moments of the cyclic-derangement arm.

    Like every arm's final observable readout, each register's O readout is
    ideal (spent registers are read mid-circuit, charged t_meas only); only
    the sign/control readout carries readout_q.
    """
    n=M*N;regs=[tuple(range(M*i,M*(i+1))) for i in range(N)]
    R1=[partial_trace(Rf,r,n) for r in regs]
    num=np.mean([np.trace(O@partial_trace(Wf,r,n)).real for r in regs])
    m1=np.mean([np.trace(O@R1[i]).real for i in range(N)])
    pair=0.
    for i in range(N):
        for j in range(N):
            pair+=(np.trace(O@O@R1[i]).real if i==j else
                   np.trace(np.kron(O,O)@partial_trace(Rf,regs[i]+regs[j],n)).real)
    return dict(t=float(np.trace(Wf).real),num=float(num),m1=float(m1),m2=float(pair/N**2),trR=float(np.trace(Rf).real))


def derangement_table(Rf,Wf,O,M,N):
    """Joint (reported parity, register-averaged O outcome) distribution."""
    ev,U=np.linalg.eigh(_herm(np.asarray(O,complex)));d=1<<M
    UU=U
    for _ in range(N-1):UU=np.kron(UU,U)
    sg=[];vals=[];probs=[]
    for s in (1,-1):
        diag=np.clip(np.real(np.diag(UU.conj().T@((Rf+s*Wf)/2)@UU)),0,None).reshape((d,)*N)
        for idx in np.ndindex(*(d,)*N):
            sg.append(s);vals.append(float(np.mean([ev[k] for k in idx])));probs.append(diag[idx])
    probs=np.array(probs);probs/=probs.sum()
    return np.array(sg,float),np.array(vals,float),probs


def run_derangement(rho,N:int,model:Hardware,O,t_prep=None,ledger=None):
    """Cyclic-derangement estimator: wide_cyclic circuit, then O on ALL N registers.

    The per-shot numerator is Omega*(1/N)sum_i o_i. Each spent register B_k is
    measured as soon as its CSWAP ends (no later gate touches it), charged a
    mid-circuit measurement duration t_meas (off the critical path). As in
    every arm, observable readouts are ideal; readout_q acts on the control's
    sign readout only. Dense joint simulation: 1+MN<=11 qubits.
    Returns dict with t, the signed and unsigned moments needed by metrics().
    """
    hw=model;tp=hw.t_prep if t_prep is None else int(t_prep)
    M=int(round(math.log2(rho.shape[0])));d=1<<M
    if 1+M*N>11:raise ValueError('derangement arm limited to 1+MN<=11 qubits')
    led=ledger if ledger is not None else Ledger()
    led.peak_qubits=_width(M,N,'derangement');led.preparations=N
    Tc=cswap_duration(M,hw)
    state=np.kron(ancilla_plus(hw,False),rho.astype(complex));led.gates1+=1
    for k in range(1,N):
        nq=1+M*(k+1)
        state=np.kron(state,rho.astype(complex))
        state,_=cswap_gadget(state,nq,0,list(range(1,1+M)),list(range(1+M*k,1+M*(k+1))),hw,led)
    T=tp+(N-1)*Tc if N>1 else tp
    led.duration=T+hw.t_meas if N>1 else tp
    led.measurements+=(1+(N-1)) if N>1 else 0     # the control plus every spent register
    if N>1:
        p,m=_x_branches(state,d**N)
        p,m=observed_branches(p,m,hw.readout_q)
    else:p,m=_herm(rho.astype(complex)),np.zeros((d,d),complex)
    Rf=_herm(p+m);Wf=_herm(p-m)
    mom=derangement_moments(Rf,Wf,O,M,N)
    regs=[tuple(range(M*i,M*(i+1))) for i in range(N)]
    return dict(mom,R1=partial_trace(Rf,regs[0],M*N),W1=partial_trace(Wf,regs[0],M*N),Rfull=Rf,Wfull=Wf)


def simulate(rho,N,architecture,model,t_prep=None,ledger=None):
    """(R,W) for every single-register-readout architecture."""
    if architecture=='coherent_request':return run_coherent(rho,N,model,t_prep,ledger,wide=False)
    if architecture=='wide_cyclic':return run_coherent(rho,N,model,t_prep,ledger,wide=True)
    if architecture=='derangement':raise ValueError('use simulate_moments for derangement')
    return run_measured(rho,N,architecture,model,t_prep,ledger)


def simulate_moments(rho,N,architecture,model,O,t_prep=None):
    """Uniform interface: estimator moments plus the schedule ledger."""
    led=Ledger()
    if architecture=='derangement':
        mom=run_derangement(rho,N,model,O,t_prep,led)
        return dict(t=mom['t'],num=mom['num'],m1=mom['m1'],m2=mom['m2'],trR=mom['trR']),led,mom
    R,W=simulate(rho,N,architecture,model,t_prep,led)
    mom=dict(t=float(np.trace(W).real),num=float(np.trace(O@W).real),
             m1=float(np.trace(O@R).real),m2=float(np.trace(O@O@R).real),trR=float(np.trace(R).real))
    return mom,led,(R,W)


ZERO_DENOMINATOR=1e-11   # numerical cut on |Tr W|; below it the ratio is reported undefined


def moments_metrics(mom,ideal_expect):
    """Noisy-instrument bias, delta-method variance and leading ratio bias.

    With per-shot numerator X=Omega*obar and denominator Omega (Omega^2=1):
       t=E[Omega], mu=E[X]/t, m1=E[obar], m2=E[obar^2]
       v=[m2-2 mu m1+mu^2]/t^2 ,  ratio-bias coefficient c=(mu-m1)/t^2.
    For a single measured register obar=o, m1=Tr(O R), m2=Tr(O^2 R).
    """
    trW=mom['t']
    if abs(mom.get('trR',1)-1)>1e-7:raise AssertionError(f'non-trace preserving {mom["trR"]}')
    if abs(trW)<ZERO_DENOMINATOR:
        return dict(normalization=trW,estimate=np.nan,bias=np.nan,variance_coeff=np.inf,
                    ordinary_expect=mom['m1'],ratio_bias_coeff=np.nan)
    mu=mom['num']/trW
    numer=mom['m2']-2*mu*mom['m1']+mu**2
    if numer< -1e-8:raise AssertionError('negative variance')
    return dict(normalization=trW,estimate=mu,bias=mu-ideal_expect,variance_coeff=max(numer,0)/trW**2,
                ordinary_expect=mom['m1'],ratio_bias_coeff=(mu-mom['m1'])/trW**2)


def metrics(R,W,O,ideal_expect):
    """Single-register metrics from (R,W) (draft Eqs. noisy_estimator_mean-variance_coefficient)."""
    if np.linalg.eigvalsh(_herm(R)).min()<-1e-8:raise AssertionError('R not PSD')
    mom=dict(t=float(np.trace(W).real),num=float(np.trace(O@W).real),m1=float(np.trace(O@R).real),
             m2=float(np.trace(O@O@R).real),trR=float(np.trace(R).real))
    return moments_metrics(mom,ideal_expect)


GUARD_ST2=50.   # minimum S*t^2 for the delta-method planning to be trusted (validated in tests)


def plan_shots(row,epsilon=.03,guard=GUARD_ST2):
    """Shots for target RMSE epsilon from the leading MSE b^2+(v+2bc)/S.

    Infeasible when b^2>=epsilon^2. The guard enforces S*t^2>=guard so that the
    random denominator is resolved (below it the delta method fails: see
    tests and RESULTS_OVERVIEW). Returns dict(feasible, shots, guard_binding).
    """
    b,v,c,t=row['bias'],row['variance_coeff'],row['ratio_bias_coeff'],row['normalization']
    if not np.isfinite(v) or not np.isfinite(b) or b*b>=epsilon**2:
        return dict(feasible=False,shots=np.nan,guard_binding=False,shots_delta=np.nan)
    S0=max(1,math.ceil(max(v+2*b*c,0.)/(epsilon**2-b*b)))
    Sg=math.ceil(guard/t**2)
    return dict(feasible=True,shots=max(S0,Sg),guard_binding=bool(Sg>S0),shots_delta=S0)


def resources(M,N,architecture,model=None):
    """Static per-shot counts (the dynamic clock record is the Ledger).

    Reset counts follow the first-use rule: only actual mid-circuit REUSES.
    """
    l=int(math.log2(N)) if N>1 and N&(N-1)==0 else None
    width=_width(M,N,architecture)
    data_resets={'tree_parallel':0,'tree_recycled':(N-(l+1) if l is not None else None),
                 'tree_sequential':max(N-2,0),'coherent_request':max(N-2,0),
                 'wide_cyclic':0,'derangement':0}[architecture] if N>1 else 0
    anc_resets={'tree_parallel':max(N//2-1,0),'tree_recycled':max(N-2,0),'tree_sequential':max(N-2,0),
                'coherent_request':0,'wide_cyclic':0,'derangement':0}[architecture] if N>1 else 0
    depth=(l if architecture=='tree_parallel' else max(N-1,0))
    return dict(peak_qubits=width,register_swaps=max(N-1,0),toffolis=M*max(N-1,0),
                cnots=2*M*max(N-1,0),
                controls_measured=(max(N-1,0) if architecture in MEASURED else (1 if N>1 else 0)),
                fresh_preparations=N,critical_swap_stages=depth,
                data_reset_events=data_resets,ancilla_reset_events=anc_resets)


def _rng(seed):
    if isinstance(seed,np.random.Generator):return seed
    return np.random.default_rng(np.random.SeedSequence(seed if isinstance(seed,(list,tuple)) else [int(seed)]))


def outcome_table(R,W,O):
    """(signs, values, probabilities) of the joint (reported parity, O outcome)."""
    eigenvalues,U=np.linalg.eigh(O)
    probs=[];sg=[];vals=[]
    for parity in (1,-1):
        branch=(R+parity*W)/2
        p=np.real(np.diag(dag(U)@branch@U))
        for value,prob in zip(eigenvalues,p):
            probs.append(max(0.,prob));sg.append(parity);vals.append(float(value.real))
    probs=np.array(probs,float);probs/=probs.sum()
    return np.array(sg,float),np.array(vals,float),probs


def sample_ratio(R,W,O,shots,seed=1):
    """Physical +/- instrument data plus final observable measurement.

    `seed` is an int, a SeedSequence entropy tuple, or a Generator. Use a tuple
    that includes `shots` so different shot counts are independent draws.
    """
    rng=_rng(seed)
    sg,vals,probs=outcome_table(R,W,O)
    counts=rng.multinomial(shots,probs)
    den=float(counts@sg)
    return dict(empirical=float(counts@(sg*vals)/den) if den!=0 else np.nan,
                signed_normalization=den/shots,shots=shots,zero_denominator=bool(den==0))


def mc_rmse(sg,vals,probs,mu0,shots,reps,seed):
    """Monte Carlo RMSE of the raw ratio estimator at fixed shots, with a 95% CI.

    Draws with a zero denominator are counted and excluded (reported).
    The CI uses the normal approximation for the mean squared error.
    """
    rng=_rng(seed)
    c=rng.multinomial(int(shots),probs,size=int(reps))
    num=c@(sg*vals);den=c@sg
    ok=den!=0;est=num[ok]/den[ok];se2=(est-mu0)**2
    mse=se2.mean();h=1.96*se2.std(ddof=1)/np.sqrt(len(se2))
    return dict(rmse=float(np.sqrt(mse)),rmse_lo=float(np.sqrt(max(mse-h,0))),rmse_hi=float(np.sqrt(mse+h)),
                p_zero_den=float(1-ok.mean()),mc_bias=float(est.mean()-mu0),mc_reps=int(reps))
