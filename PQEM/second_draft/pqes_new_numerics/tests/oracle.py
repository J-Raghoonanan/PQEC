"""Independent brute-force oracle (shares NO code with core.py/circuits.py).

Ported from the audit's bf.py/bf_arch.py. Every live register is kept in ONE
joint density matrix, every ancilla is measured with explicit X projectors,
every true outcome string and every REPORTED string (independent flips) is
enumerated, and every channel is an explicit Kraus/Pauli sum:
  * depolarizing D_eps on k qubits = (1-eps) X + eps/4^k sum_P P X P;
  * idle unit = amplitude damping (gamma) then Z flip (phi), one unit at a time;
  * compiled CSWAP = CNOT(b->a) layer, M Toffolis, CNOT(b->a) layer.
Idle waits between gadgets are passed in explicitly (the tests derive them by
hand from the clock rules), so the oracle also checks the schedule.
Qubit 0 is the most significant tensor factor.
"""
from __future__ import annotations
import itertools
import numpy as np

I2=np.eye(2,dtype=complex)
PX=np.array([[0,1],[1,0]],complex)
PY=np.array([[0,-1j],[1j,0]],complex)
PZ=np.array([[1,0],[0,-1]],complex)
PAULIS=(I2,PX,PY,PZ)
PLUSDM=np.full((2,2),.5,complex)
MINUSDM=np.array([[.5,-.5],[-.5,.5]],complex)


def apply_op(rho,op,qubits,n):
    """op_q rho op_q^dagger via tensordot and an explicit axis permutation."""
    k=len(qubits);qubits=list(qubits)
    t=rho.reshape((2,)*(2*n));opr=op.reshape((2,)*(2*k))
    t=np.tensordot(opr,t,axes=(list(range(k,2*k)),qubits))
    rest=[a for a in range(2*n) if a not in qubits];cur=qubits+rest
    t=np.transpose(t,[cur.index(a) for a in range(2*n)])
    cols=[n+q for q in qubits]
    t=np.tensordot(t,op.conj().reshape((2,)*(2*k)),axes=(cols,list(range(k,2*k))))
    rest=[a for a in range(2*n) if a not in cols];cur=rest+cols
    t=np.transpose(t,[cur.index(a) for a in range(2*n)])
    return t.reshape(rho.shape)


def kraus(rho,ops,qubits,n):return sum(apply_op(rho,K,qubits,n) for K in ops)


def pauli_strings(k):
    for idx in itertools.product(range(4),repeat=k):
        P=np.array([[1]],complex)
        for i in idx:P=np.kron(P,PAULIS[i])
        yield idx,P


def depol(rho,qubits,n,p):
    if p==0:return rho
    k=len(qubits);ops=[]
    for idx,P in pauli_strings(k):
        w=(1-p+p/4**k) if all(i==0 for i in idx) else p/4**k
        ops.append(np.sqrt(w)*P)
    return kraus(rho,ops,list(qubits),n)


def idle(rho,qubits,n,units,gamma,phi):
    K0=np.diag([1,np.sqrt(1-gamma)]).astype(complex);K1=np.array([[0,np.sqrt(gamma)],[0,0]],complex)
    for _ in range(int(units)):
        for q in qubits:
            if gamma:rho=kraus(rho,[K0,K1],[q],n)
            if phi:rho=kraus(rho,[np.sqrt(1-phi)*I2,np.sqrt(phi)*PZ],[q],n)
    return rho


def perm_unitary(k,fn):
    U=np.zeros((2**k,2**k),complex)
    for bits in itertools.product(range(2),repeat=k):
        out=fn(bits);U[int(''.join(map(str,out)),2),int(''.join(map(str,bits)),2)]=1
    return U


CNOT=perm_unitary(2,lambda b:(b[0],b[1]^b[0]))           # control first
CCX=perm_unitary(3,lambda b:(b[0],b[1],b[2]^(b[0]&b[1])))
FRED=perm_unitary(3,lambda b:(b[0],b[2],b[1]) if b[0] else b)


def ptrace_keep(rho,keep,n):
    keep=list(keep);t=rho.reshape((2,)*(2*n));cur=list(range(n))
    for q in sorted([q for q in range(n) if q not in keep],reverse=True):
        i=cur.index(q);nn=len(cur);t=np.trace(t,axis1=i,axis2=nn+i);cur.pop(i)
    nn=len(cur);perm=[cur.index(q) for q in keep]
    t=np.transpose(t,perm+[nn+p for p in perm]);d=2**len(keep)
    return t.reshape(d,d)


def project_x(rho,q,n,outcome):
    v=np.array([1,1],complex)/np.sqrt(2) if outcome==0 else np.array([1,-1],complex)/np.sqrt(2)
    return apply_op(rho,np.outer(v,v.conj()),[q],n)


class Reg:
    def __init__(self):self.names=[];self.rho=np.array([[1]],complex)
    @property
    def n(self):return len(self.names)
    def idx(self,nm):return self.names.index(nm)
    def add(self,names,rho):self.names+=list(names);self.rho=np.kron(self.rho,rho)
    def drop(self,names):
        keep=[i for i,nm in enumerate(self.names) if nm not in names]
        self.rho=ptrace_keep(self.rho,keep,self.n);self.names=[self.names[i] for i in keep]
    def copy(self):r=Reg();r.names=list(self.names);r.rho=self.rho.copy();return r


def gadget(reg,c,A,B,hw):
    """Compiled CSWAP with gate noise and in-gadget idle units."""
    g,f=hw['gamma'],hw['phi']
    def cnots():
        for a,b in zip(A,B):
            qs=[reg.idx(b),reg.idx(a)]
            reg.rho=apply_op(reg.rho,CNOT,qs,reg.n);reg.rho=depol(reg.rho,qs,reg.n,hw['eps2'])
        reg.rho=idle(reg.rho,[reg.idx(c)],reg.n,1,g,f)
    cnots()
    for a,b in zip(A,B):
        qs=[reg.idx(c),reg.idx(a),reg.idx(b)]
        reg.rho=apply_op(reg.rho,CCX,qs,reg.n);reg.rho=depol(reg.rho,qs,reg.n,hw['eps3'])
        others=[reg.idx(q) for q in list(A)+list(B) if q not in (a,b)]
        reg.rho=idle(reg.rho,others,reg.n,1,g,f)
    cnots()


def anc_state(hw,reused):
    r=hw['ra'] if reused else 0
    a=(1-r)*PLUSDM+r*MINUSDM
    return (1-hw['eps1'])*a+hw['eps1']*I2/2


def leaf(rho,hw,reused):
    if not reused:return rho
    g=np.zeros_like(rho);g[0,0]=1
    return (1-hw['rd'])*rho+hw['rd']*g


def names(tag,M):return [f'{tag}{j}' for j in range(M)]


def merge(branches,a,b,tag,M,hw,anc_reused,waits):
    """waits: dict register-tag -> idle units applied before this merge."""
    out=[]
    for bits,reg in branches:
        reg=reg.copy()
        for t,u in waits.items():reg.rho=idle(reg.rho,[reg.idx(q) for q in names(t,M)],reg.n,u,hw['gamma'],hw['phi'])
        reg.add([tag],anc_state(hw,anc_reused))
        gadget(reg,tag,names(a,M),names(b,M),hw)
        for o in (0,1):
            r2=reg.copy();r2.rho=project_x(r2.rho,r2.idx(tag),r2.n,o);r2.drop([tag]+names(b,M))
            out.append((bits+(o,),r2))
    return out


def reported_RW(branches,q):
    R=W=None
    for bits,reg in branches:
        for rep in itertools.product((0,1),repeat=len(bits)):
            w=1.
            for b_,r_ in zip(bits,rep):w*=(1-q) if b_==r_ else q
            om=(-1)**sum(rep)
            R=w*reg.rho if R is None else R+w*reg.rho
            W=w*om*reg.rho if W is None else W+w*om*reg.rho
    return (R+R.conj().T)/2,(W+W.conj().T)/2


def sequential(rho,M,N,hw,wait):
    reg=Reg();reg.add(names('d0_',M),rho);br=[((),reg)]
    for j in range(1,N):
        nb=[]
        for bits,r in br:
            r=r.copy();r.add(names(f'f{j}_',M),leaf(rho,hw,j>1));nb.append((bits,r))
        br=merge(nb,'d0_',f'f{j}_',f'c{j}',M,hw,j>1,{'d0_':wait} if j>1 else {})
    return reported_RW(br,hw['q'])


def parallel4(rho,M,hw,wait2):
    reg=Reg()
    for k in range(4):reg.add(names(f'd{k}_',M),rho)
    br=[((),reg)]
    br=merge(br,'d0_','d1_','c1',M,hw,False,{})
    br=merge(br,'d2_','d3_','c2',M,hw,False,{})
    br=merge(br,'d0_','d2_','c3',M,hw,True,{'d0_':wait2,'d2_':wait2})
    return reported_RW(br,hw['q'])


def recycled4(rho,M,hw,w_left,w_right):
    reg=Reg();reg.add(names('d0_',M),rho);reg.add(names('d1_',M),rho);br=[((),reg)]
    br=merge(br,'d0_','d1_','c1',M,hw,False,{})
    nb=[]
    for bits,r in br:
        r=r.copy();r.add(names('d2_',M),rho);r.add(names('d3_',M),leaf(rho,hw,True));nb.append((bits,r))
    br=merge(nb,'d2_','d3_','c2',M,hw,True,{})
    br=merge(br,'d0_','d2_','c3',M,hw,True,{'d0_':w_left,'d2_':w_right})
    return reported_RW(br,hw['q'])


def coherent(rho,M,N,hw,wait,wide=False,measure_all=False,tail=0):
    """One control; CSWAP(c;A,B_k). wait: idle on control+A before steps k>=2.
    measure_all keeps every B_k (derangement) and idles B_k for (N-1-k)*tail."""
    reg=Reg();reg.add(['c'],anc_state(hw,False));reg.add(names('A',M),rho)
    for k in range(1,N):
        if k>1 and wait:
            reg.rho=idle(reg.rho,[reg.idx('c')]+[reg.idx(q) for q in names('A',M)],reg.n,wait,hw['gamma'],hw['phi'])
        reg.add(names(f'B{k}_',M),leaf(rho,hw,(k>1) and not wide and not measure_all))
        gadget(reg,'c',names('A',M),names(f'B{k}_',M),hw)
        if not measure_all:reg.drop(names(f'B{k}_',M))
    if measure_all:
        for k in range(1,N):
            reg.rho=idle(reg.rho,[reg.idx(q) for q in names(f'B{k}_',M)],reg.n,(N-1-k)*tail,hw['gamma'],hw['phi'])
    br=[]
    for o in (0,1):
        r2=reg.copy();r2.rho=project_x(r2.rho,r2.idx('c'),r2.n,o);r2.drop(['c']);br.append(((o,),r2))
    return reported_RW(br,hw['q'])
