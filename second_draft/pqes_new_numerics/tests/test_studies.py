"""Checks of the study-level formulas against brute-force matrices."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import itertools
import math
import numpy as np
import pytest
import run_simulations as rs
import circuits
from core import simulate,simulate_moments,Hardware,metrics,noisy_input


@pytest.mark.parametrize('D',[2,4,8])
@pytest.mark.parametrize('N',[1,2,4,8])
def test_global_closed_forms_match_matrices(D,N):
    for p in (.1,.5,.9):
        P=np.zeros((D,D));P[0,0]=1;rho=(1-p)*P+p*np.eye(D)/D
        W=np.linalg.matrix_power(rho,N);F,T,v,a=rs.global_closed_forms(D,N,p)
        m=metrics(rho.astype(complex),W.astype(complex),P,1.)
        assert abs(F-W[0,0]/np.trace(W))<1e-12 and abs(T-np.trace(W))<1e-12
        assert abs(v/m['variance_coeff']-1)<1e-9


def test_global_leading_mse_includes_ratio_bias_cross_term():
    """Regression: global_depol mse_lead_C* must be the suite's leading MSE
    b^2+(v+2bc)/S at S=floor(C/N), with b, v, c from brute-force matrices."""
    rows,_=rs.global_depol_study('quick')
    for r in rows:
        D,N,p=r['D'],r['N'],r['p']
        P=np.zeros((D,D));P[0,0]=1;rho=(1-p)*P+p*np.eye(D)/D
        m=metrics(rho.astype(complex),np.linalg.matrix_power(rho,N).astype(complex),P,1.)
        for C in rs.BUDGETS:
            ref=rs.leading_rmse(m,math.floor(C/N))**2;got=r[f'mse_lead_C{int(C)}']
            assert np.isnan(got)==np.isnan(ref)
            assert np.isnan(ref) or abs(got-ref)<=1e-9*ref


@pytest.mark.parametrize('D',[2,3,4,8])
def test_saddle_node_matches_closed_form_for_N2(D):
    assert abs(rs.p_saddle_node(D,2)-D/(D+2*np.sqrt(D-1)))<2e-3


def test_qubit_global_boundary_is_one_minus_one_over_N():
    for N in (2,4,8):
        assert abs(rs.p_saddle_node(2,N)-(1-1/N))<2e-3


def test_local_product_closed_forms():
    rows=rs.local_resource_study('quick',3)
    for r in rows:
        if r['target']!='plus' or r['M']>2:continue
        _,P0=noisy_input(r['M'],'plus',0);rho,_=noisy_input(r['M'],'plus',r['p'])
        W=np.linalg.matrix_power(rho,r['N'])
        assert abs(r['T_N']-np.trace(W).real)<1e-12
        assert abs(r['F_N']-np.trace(P0@W).real/np.trace(W).real)<1e-12


def test_sequential_jordan_words_reproduce_tree_on_nonidentical_inputs():
    rng=np.random.default_rng(2)
    def rd():
        z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2));r=z@z.conj().T;return r/np.trace(r)
    for N in (2,3,4):
        rhos=[rd() for _ in range(N)]
        J=rhos[0];A=rhos[0]
        for r in rhos[1:]:J=(J@r+r@J)/2;A=(A+r)/2
        words,wR=rs._seq_words(N)
        Wexp=sum(np.linalg.multi_dot([rhos[j] for j in w]) if len(w)>1 else rhos[w[0]] for w in words)/2**(N-1)
        np.testing.assert_allclose(Wexp,J,atol=1e-13)
        np.testing.assert_allclose(sum(w*r for w,r in zip(wR,rhos)),A,atol=1e-13)


def test_frame_states_average_is_local_depolarizing():
    for theta in (np.pi/2,np.pi/3):
        tgt,fs=rs.frame_states(theta,.3,0.)
        bar=sum(fs)/3;dep=(1-.3)*tgt+.1*sum(P@tgt@P for P in (rs.X,rs.Y,rs.Z))
        np.testing.assert_allclose(bar,dep,atol=1e-13)


def test_per_qubit_factorization_of_random_frames():
    """Twirl Monte Carlo factorization equals the explicit M-qubit tree."""
    M=2;N=3;tgt,fs=rs.frame_states(np.pi/3,.45,.01)
    words,wR=rs._seq_words(N);Tw,Tt,Fq=rs._per_qubit_tables(tgt,fs,N,words)
    P=rs._kron_all([tgt]*M);pw=3**np.arange(N)[::-1]
    for fr in itertools.islice(itertools.product(range(3),repeat=M*N),0,729,37):
        fr=np.array(fr).reshape(M,N)
        rhos=[rs._kron_all([fs[fr[m,j]] for m in range(M)]) for j in range(N)]
        J=rhos[0]
        for r in rhos[1:]:J=(J@r+r@J)/2
        tix=fr@pw
        b=Tw[tix].prod(axis=0).sum().real/2**(N-1);t=Tt[tix].prod(axis=0).sum().real/2**(N-1)
        assert abs(b-np.trace(P@J).real)<1e-12 and abs(t-np.trace(J).real)<1e-12


def test_reference_profiles_are_recorded():
    a,b=rs.reference('fast'),rs.reference('slow')
    assert a.t_meas<b.t_meas and a.eps1==a.eps2/10


def test_leading_rmse_undefined_outside_delta_method_regime():
    """Regression: b^2+(v+2bc)/S can turn negative when S t^2 is small; it must
    be reported as undefined, never clamped to zero or used to pick an optimal N."""
    row=dict(bias=-.05,variance_coeff=400.,ratio_bias_coeff=-2e4,normalization=.03)
    assert np.isnan(rs.leading_rmse(row,3000))           # S t^2 = 2.7
    assert np.isfinite(rs.leading_rmse(row,10**6))         # S t^2 = 900
    rows=rs.local_resource_study('quick',3)
    assert all(not np.isfinite(r['mse_lead_C1000']) or r['mse_lead_C1000']>0 for r in rows)


def test_leading_rmse_is_zero_for_an_exact_zero_mse():
    """Regression: a noiseless pure input (b=v=c=0) has leading MSE exactly 0, not
    undefined; only m<0 or a failed S t^2 guard is undefined."""
    assert rs.leading_rmse(dict(bias=0.,variance_coeff=0.,ratio_bias_coeff=0.,normalization=1.),1000)==0.
    assert np.isnan(rs.leading_rmse(dict(bias=-.01,variance_coeff=1.,ratio_bias_coeff=1e3,normalization=1.),1000))
    rows,_=rs.global_depol_study('quick')
    zero=[r for r in rows if r['p']==0]
    assert zero and all(r[f'mse_lead_C{int(C)}']==0 for r in zero for C in rs.BUDGETS)


def test_cycle_budgets_are_undefined_when_bias_exceeds_target():
    """Regression: copy budgets for RMSE eps must not be finite when b^2>=eps^2."""
    rows=rs.repeated_cycles_study('standard')
    bad=[r for r in rows if (1-r['ideal_virtual_fidelity'])**2>=r['rmse_target']**2]
    assert bad and all(np.isnan(r['log10_copies_budget']) and not r['feasible_at_eps'] for r in bad)
    good=[r for r in rows if (1-r['ideal_virtual_fidelity'])**2<r['rmse_target']**2]
    assert all(np.isfinite(r['log10_copies_budget']) for r in good)
    assert {r['noise_convention'] for r in rows}=={'pauli_form','replacement_form'}


def test_channel_physical_records_the_apparatus_actually_used():
    rows=rs.physical_vcp_study('quick',3)
    ideal=[r for r in rows if r['apparatus_case']=='ideal_apparatus']
    better=[r for r in rows if r['apparatus_case']=='apparatus_gates_10x_better']
    assert all(r['app_eps2']==0 and r['app_readout_q']==0 and r['app_idle_phi']==0 for r in ideal)
    assert all(abs(r['app_eps2']-r['eps2']/10)<1e-15 and r['app_readout_q']==r['readout_q'] for r in better)


def test_vd_clock_ledger_charges_the_algorithm_circuit_on_every_copy():
    """Regression: a VD row's clock ledger must include the source circuit on all
    N copies and the downstream block on every register that runs it (as VCP2's
    ledger does), so clock_* means the same across protocols, cuts and studies."""
    M=2;hw=rs.reference('fast',.005)
    idle=lambda Ls:sum(M-len({w for g in L for w in g.wires}) for L in Ls)
    two=lambda Ls:sum(len(g.wires)==2 for L in Ls for g in L)
    one=lambda Ls:sum(len(g.wires)==1 for L in Ls for g in L)
    place=[r for r in rs.placement_study('quick',3) if r['protocol']!='vcp2_layerwise' and r['eps2']==.005]
    for r in place:
        case=circuits.cut_case(M,r['family'],r['f'],hw);N=r['N']
        runs=N if r['protocol']=='derangement' else 1
        _,bare,_=simulate_moments(case['rho_A'],N,r['protocol'],hw,case['target'],case['cut'])
        assert r['clock_data_idle']==bare.data_idle+N*idle(case['pre'])+runs*idle(case['post'])
        assert r['clock_cnot']==r['cnot_per_shot']==bare.cnot+N*two(case['pre'])+runs*two(case['post'])
        assert r['clock_gates1']==r['gates1_per_shot']==bare.gates1+N*one(case['pre'])+runs*one(case['post'])
    der={}
    for r in place:
        if r['protocol']=='derangement':der.setdefault((r['family'],r['N']),set()).add(r['clock_data_idle'])
    assert der and all(len(v)==1 for v in der.values())
    full={(r['protocol'],r['N']):r for r in place if r['family']=='entangling' and r['f']==1}
    phys=[r for r in rs.physical_vcp_study('quick',3) if r['protocol']!='vcp2_layerwise']
    layers=circuits.study_layers(M,'entangling');O=circuits.ideal_output(layers,M)
    rho=circuits.run_layers(circuits.ground(M),layers,M,hw)
    for r in phys:
        N=r['N'];_,bare,_=simulate_moments(rho,N,r['protocol'],hw,O,len(layers))
        assert r['clock_cnot']==r['cnot_per_shot']==bare.cnot+N*two(layers)
        assert r['clock_gates1']==r['gates1_per_shot']==bare.gates1+N*one(layers)
        if r['apparatus_case']=='physical' and r['eps2']==.005:
            ref=full[(r['protocol'],r['N'])]
            assert all(r[k]==ref[k] for k in ('clock_data_idle','clock_cnot','clock_gates1','clock_duration'))
