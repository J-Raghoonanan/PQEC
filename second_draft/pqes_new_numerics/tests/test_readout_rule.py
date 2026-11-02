"""The uniform observable-readout rule and the unmitigated baseline (readout.py).

Defaults must reproduce the published numbers; every new mechanism is checked
against a closed form or the independent oracle.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import math
from dataclasses import replace
import numpy as np
import pytest
from core import (Hardware,Ledger,ARCH,READOUT_FIELDS,noisy_input,simulate,simulate_moments,moments_metrics,
                  plan_shots,dag,pure_dm,prep_target)
from readout import target_readout,read_out,BASIS_CHANGE
from run_simulations import (arm_eval,reference,uniform_rule,hw_cols,baseline_plan,phase_baseline_study,
                             placement_baseline_study,channel_physical_baseline_study)
import oracle
from test_validation import _hw,_ohw

# phase.csv, eps2=0.005, fast, q=0.01, idle 5e-4: (bias, v, shots, duration)
PUBLISHED_REFERENCE_CELL={
    'tree_parallel':(-0.0133643150852859,2.947774128160616,3980,15),
    'tree_recycled':(-0.0138248269826823,3.068902468282311,4213,23),
    'tree_sequential':(-0.0150996694153263,2.955408244316184,4269,24),
    'coherent_request':(-0.015135495219259,2.706355431316312,3916,24),
    'wide_cyclic':(-0.0143756791037896,2.60267982427564,3649,16),
    'derangement':(-0.0103686383778969,1.125127439970768,1355,16),
}
PUBLISHED_HW_COLUMNS=['eps1','eps2','eps3','readout_q','ancilla_reset_p','data_reset_p','idle_gamma','idle_phi',
                      't_prep','t_meas','t_reset']


def _ref():return reference('fast',.005,readout_q=.01,idle_gamma=5e-4,idle_phi=5e-4)


def test_defaults_reproduce_published_reference_cell():
    rho,O=noisy_input(2,'bell_tilted',.14);ro=target_readout('bell_tilted',2,O)
    for arch,(b,v,S,T) in PUBLISHED_REFERENCE_CELL.items():
        for kw in ({},dict(ro=ro)):   # a compiled readout is inert under the published rule
            met,led,_=arm_eval(rho,4,arch,_ref(),O,1.,**kw)
            assert abs(met['bias']-b)<1e-13 and abs(met['variance_coeff']-v)<1e-11,arch
            assert plan_shots(met,.03)['shots']==S and led.duration==T,arch


def test_published_hardware_columns_are_unchanged():
    assert list(hw_cols(_ref()))==PUBLISHED_HW_COLUMNS
    assert list(hw_cols(_ref(),readout=True))==PUBLISHED_HW_COLUMNS+list(READOUT_FIELDS)
    assert not _ref().readout_rule_active and uniform_rule(_ref()).readout_rule_active


@pytest.mark.parametrize('kind,M',sorted(BASIS_CHANGE))
def test_basis_change_maps_target_to_all_zeros(kind,M):
    P=pure_dm(prep_target(M,kind));ro=target_readout(kind,M,P)
    want=np.zeros(1<<M);want[0]=1
    np.testing.assert_allclose(ro.unitary()@P@dag(ro.unitary()),np.diag(want),atol=1e-14)
    np.testing.assert_allclose(ro.values(),want,atol=1e-14)


@pytest.mark.parametrize('q',[.01,.05,.2])
def test_observable_flip_matches_analytic_single_register(q):
    """One qubit: a flipped bit reads O_eff=(1-2q)O+qI, so mu -> (1-2q)mu+q and
    t is untouched (unlike a sign-bit flip, which cancels in the ratio)."""
    rho,O=noisy_input(1,'tilted',.2);ro=target_readout('tilted',1,O);hw=Hardware(eps2=.01)
    for arch,N in (('coherent_request',2),('tree_sequential',4),('unmitigated',1)):
        ideal,_,_=arm_eval(rho,N,arch,hw,O,1.)
        met,_,_=arm_eval(rho,N,arch,replace(hw,obs_readout_q=q),O,1.,ro=ro)
        assert abs(met['normalization']-ideal['normalization'])<1e-14
        assert abs(met['estimate']-((1-2*q)*ideal['estimate']+q))<1e-13
    F=float(np.trace(O@rho).real);mu=(1-2*q)*F+q
    assert abs(met['variance_coeff']-mu*(1-mu))<1e-13


def test_two_qubit_flip_weights_and_correction():
    """Two bits: reported value (1-q)^2, q(1-q), q(1-q), q^2 on the rotated basis
    strings; the inverse-confusion correction restores mu exactly."""
    rho,O=noisy_input(2,'bell_tilted',.14);ro=target_readout('bell_tilted',2,O);q=.03;hw=Hardware(eps2=.005)
    R,W=simulate(rho,4,'wide_cyclic',hw);V=ro.unitary()
    w=np.array([(1-q)**2,q*(1-q),q*(1-q),q*q])
    met,_,_=arm_eval(rho,4,'wide_cyclic',replace(hw,obs_readout_q=q),O,1.,ro=ro)
    assert abs(met['estimate']-w@np.real(np.diag(V@W@dag(V)))/np.trace(W).real)<1e-13
    for arch in ('wide_cyclic','derangement'):
        base,_,_=arm_eval(rho,4,arch,hw,O,1.)
        corr,_,_=arm_eval(rho,4,arch,replace(hw,obs_readout_q=q,obs_flip_correction=True),O,1.,ro=ro)
        assert abs(corr['estimate']-base['estimate'])<1e-13 and corr['variance_coeff']>base['variance_coeff']


def test_basis_change_is_charged_on_the_clock_and_ledger():
    rho,O=noisy_input(2,'bell_tilted',.14);ro=target_readout('bell_tilted',2,O);N=4
    hw=_ref();hu=replace(hw,obs_basis_change=True)
    for arch in ARCH+('unmitigated',):
        n=1 if arch=='unmitigated' else N;reads=N if arch=='derangement' else 1
        _,a,_=arm_eval(rho,n,arch,hw,O,1.);met,b,_=arm_eval(rho,n,arch,hu,O,1.,ro=ro)
        assert b.duration==a.duration+2 and b.peak_qubits==a.peak_qubits,arch
        assert (b.cnot-a.cnot,b.gates1-a.gates1,b.data_idle-a.data_idle)==(reads,reads,reads),arch
        ideal,_,_=arm_eval(rho,n,arch,hw,O,1.)
        assert met['bias']<ideal['bias'],arch   # the compiled CNOT and Ry carry eps2, eps1 and idle noise


def test_uniform_rule_needs_a_compiled_readout():
    rho,O=noisy_input(2,'bell_tilted',.14)
    with pytest.raises(ValueError):arm_eval(rho,4,'wide_cyclic',replace(_ref(),obs_readout_q=.01),O,1.)


@pytest.mark.parametrize('correct',[False,True])
def test_end_equals_mid_at_zero_idle(correct):
    rho,O=noisy_input(2,'bell_tilted',.14);ro=target_readout('bell_tilted',2,O)
    hw=reference('slow',.005,idle_gamma=0.,idle_phi=0.)
    mid,lm,(sm,vm,pm)=arm_eval(rho,4,'derangement',uniform_rule(hw,'mid',correct),O,1.,ro=ro)
    end,le,(se,ve,pe)=arm_eval(rho,4,'derangement',uniform_rule(hw,'end',correct),O,1.,ro=ro)
    assert mid==end and plan_shots(mid,.03)==plan_shots(end,.03)
    assert np.array_equal(pm,pe) and np.array_equal(vm,ve)
    assert (le.duration,le.cnot,le.gates1)==(lm.duration,lm.cnot,lm.gates1)
    assert le.data_idle-lm.data_idle==2*(2+1)*4   # idle exposure is still counted: (N-1-k)(M+2) units on M qubits
    idle,_,_=arm_eval(rho,4,'derangement',uniform_rule(reference('slow',.005),'end',correct),O,1.,ro=ro)
    assert idle['bias']<mid['bias']


@pytest.mark.parametrize('N',[2,3])
def test_end_timing_matches_oracle(N):
    """Spent registers idle until the last CSWAP ends: the oracle's tail=M+2."""
    M=1;rho,O=noisy_input(1,'tilted',.2)
    Rb,Wb=oracle.coherent(rho,M,N,_ohw(),0,measure_all=True,tail=M+2)
    mom,led,full=simulate_moments(rho,N,'derangement',replace(_hw(),derangement_readout='end'),O)
    assert max(np.abs(Rb-full['Rfull']).max(),np.abs(Wb-full['Wfull']).max())<1e-13
    _,lm,_=simulate_moments(rho,N,'derangement',_hw(),O)
    assert led.data_idle-lm.data_idle==M*(M+2)*sum(N-1-k for k in range(1,N)) and led.duration==lm.duration


@pytest.mark.parametrize('p',[.02,.14,.3])
def test_unmitigated_baseline_matches_analytic_fidelity(p):
    """bell_tilted under Pauli-form local depolarizing p: of the 16 Pauli pairs
    only II, ZZ (overlap 1), ZI, IZ (1/2), XX, YY (sqrt(3)/2) keep overlap."""
    rho,O=noisy_input(2,'bell_tilted',p)
    F=(1-p)**2+2*(1-p)*(p/3)*.25+(p/3)**2*(1+.75+.75)
    hw=reference('slow',.005)
    met,led,(sg,vals,probs)=arm_eval(rho,1,'unmitigated',hw,O,1.)
    assert abs(met['estimate']-F)<1e-14 and abs(met['bias']-(F-1))<1e-14
    assert abs(met['variance_coeff']-F*(1-F))<1e-14 and met['ratio_bias_coeff']==0 and met['normalization']==1
    assert (led.duration,led.peak_qubits,led.preparations,led.cnot,led.ccx,led.measurements)==(hw.t_prep+hw.t_meas,2,1,0,0,0)
    assert np.all(sg[probs>0]==1)
    plan=baseline_plan(met,.03)
    assert plan['feasible']==((1-F)**2<.03**2)
    if plan['feasible']:
        assert plan['shots']==plan['shots_delta']==max(1,math.ceil(F*(1-F)/(.03**2-(1-F)**2)))


def _exact_baseline_shots(r,eps=.03):
    b,v=r['bias'],r['variance_coeff']
    return max(1,math.ceil(v/(eps*eps-b*b))) if b*b<eps*eps else None


@pytest.mark.parametrize('study',['phase','placement','channel_physical'])
def test_baseline_studies_plan_the_exact_shot_count(study):
    """Regression: the baseline was planned with the S t^2>=50 guard (50 shots,
    or 51 where Tr(rho) rounded below 1) instead of its exact shot count."""
    rows={'phase':lambda:phase_baseline_study('quick',31),'placement':lambda:placement_baseline_study('full'),
          'channel_physical':lambda:channel_physical_baseline_study('full',31)}[study]()
    for r in rows:
        assert r['normalization']==1 and r['ratio_bias_coeff']==0
        S=_exact_baseline_shots(r)
        assert r['feasible']==(S is not None) and not r['guard_binding']
        if S is None:assert np.isnan(r['shots']) and np.isnan(r['shots_delta'])
        else:assert r['shots']==r['shots_delta']==S and r['total_copies']==S


def test_baseline_shots_below_the_guard():
    """Cells that the guard used to set to 50 (or 51) shots."""
    place={(r['family'],r['eps2'],r['latency_profile']):r for r in placement_baseline_study('full')}
    chan={(r['M'],r['family'],r['eps2'],r['latency_profile']):r for r in channel_physical_baseline_study('full',31)}
    assert place[('product',.002,'fast')]['shots']==2 and place[('entangling',.005,'fast')]['shots']==41
    assert chan[(1,'entangling',.01,'fast')]['shots']==4 and chan[(1,'entangling',.01,'slow')]['shots']==4
