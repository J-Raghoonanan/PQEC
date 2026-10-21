"""Independent physics and regression checks (run: pytest -q tests/).

A failed exact identity is an implementation error. The oracle (tests/oracle.py)
shares no code with the simulator; clock tests use waits derived by hand from
the schedule rules in core.py's docstring.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import math
import numpy as np
import pytest
import core
from core import (Hardware,Ledger,site_unitary,conj_local,partial_trace,depolarize_subset,
                  measure_swap,simulate,simulate_moments,noisy_input,metrics,moments_metrics,
                  resources,plan_shots,sample_ratio,outcome_table,mc_rmse,PLUS_DM,X,Y,Z)
import circuits
import vcp_physical as vp
import oracle

SINGLE=('tree_parallel','tree_recycled','tree_sequential','coherent_request','wide_cyclic')
rng0=np.random.default_rng(11)


def rand_dm(M):
    d=2**M;z=rng0.normal(size=(d,d))+1j*rng0.normal(size=(d,d));r=z@z.conj().T+.3*np.eye(d)
    return r/np.trace(r)


# ---------------------------------------------------------------- ideal identities
@pytest.mark.parametrize('M,kind',[(1,'plus'),(1,'tilted'),(2,'bell_tilted'),(2,'entangled')])
@pytest.mark.parametrize('N',[2,4,8])
def test_all_ideal_instruments(M,kind,N):
    rho,O=noisy_input(M,kind,.16)
    power=np.linalg.matrix_power(rho,N)
    for arch in SINGLE:
        R,W=simulate(rho,N,arch,Hardware())
        np.testing.assert_allclose(R,rho,atol=2e-12)
        np.testing.assert_allclose(W,power,atol=2e-12)
        for sign in (1,-1):
            assert np.linalg.eigvalsh((R+sign*W)/2).min()>-1e-12
    if 1+M*N<=9:
        mom,_,full=simulate_moments(rho,N,'derangement',Hardware(),O)
        assert abs(mom['t']-np.trace(power).real)<1e-12
        assert abs(mom['num']-np.trace(O@power).real)<1e-12
        assert abs(mom['m1']-np.trace(O@rho).real)<1e-12


def test_ideal_cyclic_permutation_unitary_equals_coherent_arm():
    for M,N in [(1,3),(2,2),(2,3)]:
        rho=rand_dm(M);d=2**M;D=d**N
        P=np.zeros((D,D),complex)
        for idx in np.ndindex(*(d,)*N):
            src=int(np.ravel_multi_index(idx,(d,)*N));dst=int(np.ravel_multi_index(idx[1:]+idx[:1],(d,)*N))
            P[dst,src]=1
        CU=np.block([[np.eye(D),np.zeros((D,D))],[np.zeros((D,D)),P]])
        full=CU@np.kron(PLUS_DM,np.kron(np.kron(rho,rho),rho) if N==3 else np.kron(rho,rho))@CU.conj().T
        blk=full.reshape(2,d,D//d,2,d,D//d)
        cross=np.trace(blk[0,:,:,1]+blk[1,:,:,0],axis1=1,axis2=3)
        _,W=simulate(rho,N,'coherent_request',Hardware())
        np.testing.assert_allclose(cross,W,atol=1e-12)


def test_swap_branch_qubit_formula():
    rho,_=noisy_input(1,'tilted',.12)
    p,m=measure_swap(rho,rho,Hardware())
    np.testing.assert_allclose(p,(rho+rho@rho)/2,atol=2e-13)
    np.testing.assert_allclose(m,(rho-rho@rho)/2,atol=2e-13)


def test_sequential_arbitrary_N_not_power_of_two():
    rho,_=noisy_input(1,'plus',.11)
    for N in (1,3,5,7):
        for arch in ('tree_sequential','coherent_request','wide_cyclic'):
            R,W=simulate(rho,N,arch,Hardware())
            np.testing.assert_allclose(W,np.linalg.matrix_power(rho,N),atol=2e-12)


# ---------------------------------------------------------------- primitives
def test_conj_local_matches_dense_embedding():
    n=4;rho=rand_dm(n)
    for wires in [(0,),(2,),(3,1),(0,2,3),(2,0,1)]:
        k=len(wires);U=np.linalg.qr(rng0.normal(size=(2**k,2**k))+1j*rng0.normal(size=(2**k,2**k)))[0]
        E=site_unitary(U,wires,n)
        np.testing.assert_allclose(conj_local(rho,U,wires,n),E@rho@E.conj().T,atol=1e-12)


def test_depolarize_subset_equals_pauli_twirl_form():
    rho=rand_dm(3)
    for wires in [(0,),(1,2),(0,1,2)]:
        np.testing.assert_allclose(depolarize_subset(rho,wires,3,.3),oracle.depol(rho,list(wires),3,.3),atol=1e-12)


def test_idle_composition_exact():
    rho=rand_dm(2);hw=Hardware(idle_gamma=.07,idle_phi=.04)
    np.testing.assert_allclose(core.apply_idle(rho,[0,1],2,5,hw),oracle.idle(rho,[0,1],2,5,.07,.04),atol=1e-12)


def test_partial_trace_product():
    a,_=noisy_input(1,'tilted',.22);b,_=noisy_input(1,'plus',.12)
    np.testing.assert_allclose(partial_trace(np.kron(a,b),(0,),2),a,atol=1e-12)
    np.testing.assert_allclose(partial_trace(np.kron(a,b),(1,),2),b,atol=1e-12)


# ---------------------------------------------------------------- oracle: noisy circuits and the clock
HWO=dict(eps1=.03,eps2=.04,eps3=.05,ra=.06,rd=.05,q=.07,gamma=.03,phi=.02,tp=2,tm=3,tr=1)


def _hw():
    o=HWO
    return Hardware(eps1=o['eps1'],eps2=o['eps2'],eps3=o['eps3'],readout_q=o['q'],ancilla_reset_p=o['ra'],
                    data_reset_p=o['rd'],idle_gamma=o['gamma'],idle_phi=o['phi'],t_prep=o['tp'],t_meas=o['tm'],t_reset=o['tr'])


def _ohw():return dict(eps1=HWO['eps1'],eps2=HWO['eps2'],eps3=HWO['eps3'],ra=HWO['ra'],rd=HWO['rd'],
                       q=HWO['q'],gamma=HWO['gamma'],phi=HWO['phi'])


@pytest.mark.parametrize('M,N',[(1,2),(1,3),(1,4),(2,2),(2,3)])
def test_oracle_sequential(M,N):
    rho=rand_dm(M);tp,tm,tr=HWO['tp'],HWO['tm'],HWO['tr']
    wait=max(tr+tp,tm+tr+1)       # next leaf: reset+prep; ancilla: measure+reset+H
    Rb,Wb=oracle.sequential(rho,M,N,_ohw(),wait)
    R,W=simulate(rho,N,'tree_sequential',_hw())
    assert max(np.abs(Rb-R).max(),np.abs(Wb-W).max())<1e-13


def test_oracle_parallel_and_recycled():
    M=1;rho=rand_dm(M);tp,tm,tr=HWO['tp'],HWO['tm'],HWO['tr'];Tc=M+2
    Rb,Wb=oracle.parallel4(rho,M,_ohw(),tm+tr+1)
    R,W=simulate(rho,4,'tree_parallel',_hw())
    assert max(np.abs(Rb-R).max(),np.abs(Wb-W).max())<1e-13
    w=max(tr+tp,tm+tr+1)
    Rb,Wb=oracle.recycled4(rho,M,_ohw(),w+Tc+tm+tr+1,tm+tr+1)
    R,W=simulate(rho,4,'tree_recycled',_hw())
    assert max(np.abs(Rb-R).max(),np.abs(Wb-W).max())<1e-13


@pytest.mark.parametrize('M,N',[(1,2),(1,3),(1,4),(2,2),(2,3)])
def test_oracle_coherent_and_wide(M,N):
    rho=rand_dm(M);tp,tr=HWO['tp'],HWO['tr']
    Rb,Wb=oracle.coherent(rho,M,N,_ohw(),tr+tp)
    R,W=simulate(rho,N,'coherent_request',_hw())
    assert max(np.abs(Rb-R).max(),np.abs(Wb-W).max())<1e-13
    Rb,Wb=oracle.coherent(rho,M,N,_ohw(),0,wide=True)
    R,W=simulate(rho,N,'wide_cyclic',_hw())
    assert max(np.abs(Rb-R).max(),np.abs(Wb-W).max())<1e-13


@pytest.mark.parametrize('N',[2,3])
def test_oracle_derangement(N):
    M=1;rho,O=noisy_input(1,'tilted',.2)
    Rb,Wb=oracle.coherent(rho,M,N,_ohw(),0,measure_all=True,tail=M+2)
    mom,_,full=simulate_moments(rho,N,'derangement',_hw(),O)
    assert max(np.abs(Rb-full['Rfull']).max(),np.abs(Wb-full['Wfull']).max())<1e-13


# ---------------------------------------------------------------- clock and reset regressions (audit T1-3, T2-3)
def test_clock_hand_computed():
    hw=Hardware(t_prep=3,t_meas=1,t_reset=1);M=2;Tc=M+2
    rho=np.eye(4,dtype=complex)/4
    def run(arch,N):
        led=Ledger();simulate(rho,N,arch,hw,ledger=led);return led
    w=max(1+3,1+1+1)
    seq=run('tree_sequential',4)
    assert seq.duration==3+3*Tc+2*w+1 and seq.data_idle==3*2*M*(M-1)+2*w*M
    coh=run('coherent_request',4)
    assert coh.duration==seq.duration and coh.control_idle==3*2+2*(1+3)
    wide=run('wide_cyclic',4)
    assert wide.duration==3+3*Tc+1 and wide.data_idle==3*2*M*(M-1) and wide.control_idle==3*2
    par=run('tree_parallel',4)
    assert par.duration==3+2*Tc+(1+1+1)+1 and par.data_idle==3*2*M*(M-1)+2*M*(1+1+1)
    rec=run('tree_recycled',4)
    left=w+Tc+(1+1+1);right=1+1+1
    assert rec.data_idle==3*2*M*(M-1)+M*(left+right)
    assert rec.duration==3+Tc+w+Tc+right+Tc+1


@pytest.mark.parametrize('N',[2,4,8])
def test_reset_counts_first_use_rule(N):
    rho=np.eye(2,dtype=complex)/2;l=int(math.log2(N))
    want={'tree_parallel':(0,N//2-1),'tree_recycled':(N-l-1,N-2),'tree_sequential':(N-2,N-2),
          'coherent_request':(N-2,0),'wide_cyclic':(0,0)}
    for arch,(dr,ar) in want.items():
        led=Ledger();simulate(rho,N,arch,Hardware(),ledger=led)
        assert (led.data_resets,led.ancilla_resets)==(dr,ar),arch
        res=resources(1,N,arch)
        assert (res['data_reset_events'],res['ancilla_reset_events'])==(dr,ar),arch
        assert led.preparations==N and led.peak_qubits==res['peak_qubits']


def test_recycled_tree_has_no_free_memory_advantage():
    """Under the shared clock the recycled survivor waits for the right subtree's
    preparations as well as its SWAPs (the audit's T1-3 artifact)."""
    rho,O=noisy_input(2,'bell_tilted',.13)
    hw=Hardware(idle_phi=.02,t_prep=3)
    v={a:metrics(*simulate(rho,4,a,hw),O,1)['variance_coeff'] for a in ('tree_recycled','tree_sequential')}
    led={a:Ledger() for a in v}
    for a in v:simulate(rho,4,a,hw,ledger=led[a])
    assert led['tree_recycled'].data_idle>led['tree_sequential'].data_idle
    assert v['tree_recycled']>v['tree_sequential']


# ---------------------------------------------------------------- bug regressions (audit T6)
def test_real_dtype_input_does_not_crash():
    rho=np.eye(4)/4
    for arch in SINGLE:
        R,W=simulate(rho,2,arch,Hardware(eps2=.01))
        assert abs(np.trace(R)-1)<1e-12


def test_N1_has_no_readout():
    rho,O=noisy_input(1,'tilted',.1)
    for arch in SINGLE:
        led=Ledger();R,W=simulate(rho,1,arch,Hardware(readout_q=.1),ledger=led)
        np.testing.assert_allclose(W,rho,atol=1e-14)
        assert led.measurements==0 and resources(1,1,arch)['controls_measured']==0


# ---------------------------------------------------------------- readout and noise-model structure
@pytest.mark.parametrize('N',[2,4,8])
@pytest.mark.parametrize('q',[0.01,0.04,0.12])
def test_readout_attenuation(N,q):
    rho,O=noisy_input(1,'tilted',.1)
    for arch in SINGLE:
        R,W=simulate(rho,N,arch,Hardware(readout_q=q))
        factor=(1-2*q)**(N-1 if arch in core.MEASURED else 1)
        np.testing.assert_allclose(W,factor*np.linalg.matrix_power(rho,N),atol=1e-12)


def test_variance_penalty_readout_only():
    rho,O=noisy_input(1,'plus',.1);N=8;q=.06
    a,b=[simulate(rho,N,s,Hardware(readout_q=q)) for s in ('tree_sequential','coherent_request')]
    va=metrics(*a,O,1)['variance_coeff'];vb=metrics(*b,O,1)['variance_coeff']
    assert abs(va/vb-(1-2*q)**(-2*(N-2)))<1e-10


def test_native_fredkin_ablation_is_pure_visibility():
    """Audit T2-1: depolarizing that includes the control only rescales W."""
    rho,O=noisy_input(2,'bell_tilted',.13)
    hw=Hardware(eps3=.03,readout_q=.04,gate_model='native_fredkin')
    for arch in SINGLE:
        for N in (2,4):
            R,W=simulate(rho,N,arch,hw);P=np.linalg.matrix_power(rho,N)
            kappa=np.trace(W).real/np.trace(P).real
            np.testing.assert_allclose(W,kappa*P,atol=1e-13)


def test_compiled_gate_noise_biases_estimator_and_separates_trees():
    """Two-qubit errors on the data pair (not the control) bias the estimator,
    and balanced trees accumulate fewer merges on the surviving path."""
    rho,O=noisy_input(2,'bell_tilted',.13)
    ideal={a:metrics(*simulate(rho,8,a,Hardware()),O,1)['bias'] for a in SINGLE}
    hw=Hardware(eps2=.01,eps3=0.)
    extra={a:metrics(*simulate(rho,8,a,hw),O,1)['bias']-ideal[a] for a in SINGLE}
    assert all(e<-5e-3 for e in extra.values())
    assert abs(extra['tree_parallel'])<abs(extra['tree_sequential'])
    assert abs(extra['tree_recycled'])<abs(extra['coherent_request'])


# ---------------------------------------------------------------- estimator statistics
def test_delta_method_moments_match_sampling():
    rho,O=noisy_input(1,'tilted',.25)
    R,W=simulate(rho,4,'tree_sequential',Hardware(readout_q=.05,eps2=.01))
    m=metrics(R,W,O,1.);sg,vals,probs=outcome_table(R,W,O)
    S=4000;c=np.random.default_rng(3).multinomial(S,probs,size=200000);est=(c@(sg*vals))/(c@sg)
    assert abs(est.var()*S/m['variance_coeff']-1)<.05
    assert abs((est.mean()-m['estimate'])*S/m['ratio_bias_coeff']-1)<.25


def test_derangement_moments_match_joint_sampling():
    rho,O=noisy_input(1,'tilted',.25);N=3
    mom,_,full=simulate_moments(rho,N,'derangement',Hardware(eps2=.01,readout_q=.03),O)
    met=moments_metrics(mom,1.)
    Rf,Wf=full['Rfull'],full['Wfull']
    ev,U=np.linalg.eigh(O);UU=np.kron(np.kron(U,U),U)
    probs=[];num=[];den=[]
    for s in (1,-1):
        diag=np.real(np.diag(UU.conj().T@((Rf+s*Wf)/2)@UU))
        for idx,pv in enumerate(diag):
            bits=np.unravel_index(idx,(2,)*N);probs.append(max(pv,0));den.append(s);num.append(s*np.mean([ev[b] for b in bits]))
    probs=np.array(probs)/sum(probs);S=3000
    c=np.random.default_rng(4).multinomial(S,probs,size=100000);est=(c@np.array(num))/(c@np.array(den))
    assert abs(est.mean()-met['estimate'])<5e-3
    assert abs(est.var()*S/met['variance_coeff']-1)<.06


def test_plan_shots_guard_and_feasibility():
    row=dict(bias=.01,variance_coeff=2.,ratio_bias_coeff=1.,normalization=.1)
    p=plan_shots(row,.03)
    assert p['feasible'] and p['shots']*.1**2>=50
    assert not plan_shots(dict(row,bias=.04),.03)['feasible']


def test_plan_shots_reaches_target_rmse():
    rho,O=noisy_input(2,'bell_tilted',.14)
    R,W=simulate(rho,4,'tree_sequential',Hardware(eps2=.005,readout_q=.02))
    row=metrics(R,W,O,1.);p=plan_shots(row,.03)
    sg,vals,probs=outcome_table(R,W,O)
    mc=mc_rmse(sg,vals,probs,1.,p['shots'],20000,(5,))
    assert mc['rmse_lo']<=.03*1.03 and mc['rmse_hi']>=.03*.85


def test_seed_streams_independent_across_shots():
    """Audit T3-2: shot counts must not share nested draws."""
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import run_simulations as rs
    a=np.random.default_rng(rs.seed_for(31,'shots','tree_sequential',4,256,0)).random(256)
    b=np.random.default_rng(rs.seed_for(31,'shots','tree_sequential',4,1024,0)).random(1024)
    assert not np.allclose(a,b[:256])
    assert rs.seed_for(31,'twirl',.1,2,64,0)!=rs.seed_for(31,'twirl',.45,8,8,4)


# ---------------------------------------------------------------- channel purification (audit T1-4)
def test_physical_vcp2_equals_ideal_proxy_for_ideal_apparatus():
    for M,fam in [(1,'entangling'),(2,'entangling'),(2,'product')]:
        layers=circuits.full_layers(M,fam,2);alg=Hardware(eps1=.004,eps2=.03)
        R,W=vp.simulate_layerwise_vcp2(M,layers,[],Hardware(),alg_hw=alg)
        rho=circuits.run_layers(circuits.ground(M),layers,M,alg)
        np.testing.assert_allclose(R,rho,atol=1e-12)
        target=circuits.ideal_output(layers,M)
        mu=np.trace(target@W).real/np.trace(W).real
        assert abs(mu-circuits.vcp_pauli_proxy(M,2,.03,2,fam,eps1=.004))<1e-12


def test_vcp_accounting_matches_ledger_and_counts_aux_gate_runs():
    M=2;layers=circuits.full_layers(M,'entangling',2);cut=7
    hw=Hardware(eps2=.01,eps1=.001,readout_q=.01,idle_phi=.001,t_meas=2,t_reset=1)
    led=Ledger();vp.simulate_layerwise_vcp2(M,layers[:cut],layers[cut:],hw,ledger=led)
    acc=vp.vcp_accounting(layers[:cut],layers[cut:])
    G=sum(len(L) for L in layers[:cut])
    assert acc['noisy_gate_executions']==2*G+sum(len(L) for L in layers[cut:])
    assert led.measurements==acc['ancilla_measurements']==cut
    assert acc['aux_preparations']==acc['control_preparations']==cut
    assert led.ancilla_resets==cut-1 and led.ccx==acc['cswap_toffolis']


def test_vcp_at_full_cut_equals_full_layerwise_and_zero_cut_is_unmitigated():
    M=2;layers=circuits.full_layers(M,'entangling',2);hw=Hardware(eps2=.01,eps1=.001)
    R1,W1=vp.simulate_layerwise_vcp2(M,layers,[],hw)
    case=circuits.cut_case(M,'entangling',1.0,2,hw)
    R2,W2=vp.simulate_layerwise_vcp2(M,case['pre'],case['post'],hw)
    np.testing.assert_allclose(W1,W2,atol=1e-14)
    R0,W0=vp.simulate_layerwise_vcp2(M,[],layers,hw)
    np.testing.assert_allclose(R0,W0,atol=1e-14)
    np.testing.assert_allclose(R0,circuits.run_layers(circuits.ground(M),layers,M,hw),atol=1e-14)


# ---------------------------------------------------------------- placement
def test_gate_placement_ideal_unitary_equivalence():
    M=2;hw0=Hardware();case=circuits.cut_case(M,'entangling',.5,2,Hardware(eps2=.02,eps1=.002))
    for arch in ('tree_sequential','coherent_request'):
        Rm,Wm=simulate(case['rho_A'],4,arch,hw0)
        Rm=circuits.run_layers(Rm,case['post'],M,hw0);Wm=circuits.run_layers(Wm,case['post'],M,hw0)
        out=circuits.run_layers(case['rho_A'],case['post'],M,hw0)
        Ro,Wo=simulate(out,4,arch,hw0)
        np.testing.assert_allclose(Wm,Wo,atol=2e-12)


def test_noisy_placement_is_not_commutative():
    M=2;hw=Hardware(eps2=.05,eps1=.005);case=circuits.cut_case(M,'entangling',.5,2,hw)
    _,W=simulate(case['rho_A'],4,'coherent_request',Hardware())
    Wmid=circuits.run_layers(W,case['post'],M,hw)
    _,Wout=simulate(circuits.run_layers(case['rho_A'],case['post'],M,hw),4,'coherent_request',Hardware())
    assert np.linalg.norm(Wmid-Wout)>1e-3


def test_independent_vs_shared_frame_average_noncommuting_nonlinearity():
    p=.3;project=PLUS_DM
    states=[(1-p)*project+p*U@project@U for U in (X,Y,Z)]
    avg=sum(states)/3
    F=lambda z:np.trace(project@z).real/np.trace(z).real
    assert abs(F(np.linalg.matrix_power(avg,2))-F(sum(np.linalg.matrix_power(r,2) for r in states)/3))>1e-3
    np.testing.assert_allclose(avg,(1-4*p/3)*project+(4*p/3)*np.eye(2)/2)


def test_ideal_apparatus_vcp_sees_the_same_algorithm_idle_noise_as_the_source():
    """Regression: with an ideal apparatus, VCP2's unsigned output must equal the
    noisy algorithm run with the same algorithm idle noise (matched comparison)."""
    M=2;layers=circuits.full_layers(M,'entangling',2)
    alg=Hardware(eps1=.002,eps2=.02,idle_gamma=2e-3,idle_phi=2e-3)
    R,W=vp.simulate_layerwise_vcp2(M,layers,[],Hardware(),alg_hw=alg)
    np.testing.assert_allclose(R,circuits.run_layers(circuits.ground(M),layers,M,alg),atol=1e-12)
