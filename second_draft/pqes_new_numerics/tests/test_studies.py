"""Checks of the study-level formulas against brute-force matrices."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
import pytest
import run_simulations as rs
from core import simulate,Hardware,metrics,noisy_input


@pytest.mark.parametrize('D',[2,4,8])
@pytest.mark.parametrize('N',[1,2,4,8])
def test_global_closed_forms_match_matrices(D,N):
    for p in (.1,.5,.9):
        P=np.zeros((D,D));P[0,0]=1;rho=(1-p)*P+p*np.eye(D)/D
        W=np.linalg.matrix_power(rho,N);F,T,v,a=rs.global_closed_forms(D,N,p)
        m=metrics(rho.astype(complex),W.astype(complex),P,1.)
        assert abs(F-W[0,0]/np.trace(W))<1e-12 and abs(T-np.trace(W))<1e-12
        assert abs(v/m['variance_coeff']-1)<1e-9


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
