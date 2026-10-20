"""Independent physics and regression checks (run: pytest -q tests/).

These tests are NOT plots: a failed exact identity is an implementation error.
The suite deliberately checks control-register coherences and nontrivial
entangled targets, not only the easy |+> state. It does not use the legacy
rho2 source, Qiskit, or any external simulation package.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from core import (Noise,site_unitary,partial_trace,depolarize_subset,
                  phase_flip,measure_swap,simulate,noisy_input,metrics,
                  spectral_power,PLUS_DM,PLUS,X,Y,Z,pure_dm,resources)
from circuits import fixed_circuit_case,apply_downstream,vcp_pauli_proxy

ARCH=('tree_parallel','tree_recycled','tree_sequential','coherent_request')

@pytest.mark.parametrize('M,kind',[(1,'plus'),(1,'tilted'),(2,'bell_tilted'),(2,'entangled')])
@pytest.mark.parametrize('N',[2,4,8])
def test_all_ideal_instruments(M,kind,N):
    rho,_=noisy_input(M,kind,.16)
    power=np.linalg.matrix_power(rho,N)
    for arch in ARCH:
        R,W=simulate(rho,N,arch,Noise())
        np.testing.assert_allclose(R,rho,atol=2e-12)
        np.testing.assert_allclose(W,power,atol=2e-12)
        assert abs(np.trace(R)-1)<1e-12
        for sign in (1,-1):
            tau=(R+sign*W)/2
            assert np.linalg.eigvalsh(tau).min()>-1e-12


def test_swap_branch_qubit_formula():
    # Fully independent check, not referring to the architecture's tree code.
    rho,_=noisy_input(1,'tilted',.12)
    p,m=measure_swap(rho,rho,Noise())
    np.testing.assert_allclose(p,(rho+rho@rho)/2,atol=2e-13)
    np.testing.assert_allclose(m,(rho-rho@rho)/2,atol=2e-13)
    assert abs(np.trace(p)+np.trace(m)-1)<1e-12

@pytest.mark.parametrize('N',[2,4,8])
@pytest.mark.parametrize('q',[0.01,0.04,0.12])
def test_readout_attenuation(N,q):
    rho,O=noisy_input(1,'tilted',.1)
    purity=np.trace(np.linalg.matrix_power(rho,N)).real
    for arch in ARCH:
        R,W=simulate(rho,N,arch,Noise(readout_q=q))
        factor=(1-2*q)**(N-1 if arch!='coherent_request' else 1)
        np.testing.assert_allclose(W,factor*np.linalg.matrix_power(rho,N),atol=1e-12)
        assert abs(np.trace(W).real-factor*purity)<1e-12
        assert abs(metrics(R,W,O,1)['estimate']-float(np.trace(O@np.linalg.matrix_power(rho,N)).real/purity))<1e-12


def test_variance_penalty_readout_only():
    rho,O=noisy_input(1,'plus',.1)
    N=8;q=.06
    a,b=[simulate(rho,N,s,Noise(readout_q=q)) for s in ('tree_sequential','coherent_request')]
    va=metrics(*a,O,1)['variance_coeff'];vb=metrics(*b,O,1)['variance_coeff']
    assert abs(va/vb-(1-2*q)**(-2*(N-2)))<1e-10


def test_depolarize_subset_completely_mixed_preserves_trace():
    rng=np.random.default_rng(8)
    z=rng.normal(size=(32,32))+1j*rng.normal(size=(32,32))
    rho=z@z.conj().T;rho/=np.trace(rho)
    for wires in [(0,),(1,2),(0,2,4)]:
        a=depolarize_subset(rho,wires,5,.3)
        assert abs(np.trace(a)-1)<1e-11
        assert np.linalg.eigvalsh(a).min()>-1e-10
        # Fully depolarizing twice gives same result; CP and trace preserving.
        a1=depolarize_subset(rho,wires,5,1.)
        np.testing.assert_allclose(depolarize_subset(a1,wires,5,1.),a1,atol=1e-10)


def test_partial_trace_product():
    a,_=noisy_input(1,'tilted',.22)
    b,_=noisy_input(1,'plus',.12)
    c=np.kron(a,b)
    np.testing.assert_allclose(partial_trace(c,(0,),2),a,atol=1e-12)
    np.testing.assert_allclose(partial_trace(c,(1,),2),b,atol=1e-12)


def test_nonzero_gate_noise_yields_valid_physical_branches():
    rho,O=noisy_input(2,'bell_tilted',.2)
    model=Noise(gate_p=.04,control_z_p=.07,readout_q=.07,
                reset_p=.02,memory_z_p=.015,control_idle_p=.02)
    for arch in ARCH:
        R,W=simulate(rho,4,arch,model)
        np.testing.assert_allclose(np.trace(R),1,atol=1e-9)
        for sign in (1,-1):
            tau=(R+sign*W)/2
            assert np.linalg.eigvalsh(tau).min()>-1e-10
        stats=metrics(R,W,O,1)
        assert stats['variance_coeff']>=0


def test_sequential_arbitrary_N_not_power_of_two():
    rho,_=noisy_input(1,'plus',.11)
    for N in (1,3,5,7):
        for arch in ('tree_sequential','coherent_request'):
            R,W=simulate(rho,N,arch,Noise())
            np.testing.assert_allclose(R,rho,atol=2e-12)
            np.testing.assert_allclose(W,np.linalg.matrix_power(rho,N),atol=2e-12)


def test_gate_placement_ideal_unitary_equivalence():
    case=fixed_circuit_case(2,b_repeats=2,algorithm_p=0)
    model=Noise()
    for arch in ('tree_sequential','coherent_request'):
        Rmid,Wmid=simulate(case['input_A'],4,arch,model)
        Rmid,Wmid=apply_downstream(Rmid,case),apply_downstream(Wmid,case)
        Rout,Wout=simulate(case['noisy_output'],4,arch,model)
        np.testing.assert_allclose(Wmid,Wout,atol=2e-12)
        np.testing.assert_allclose(Rmid,Rout,atol=2e-12)


def test_noisy_placement_is_not_commutative():
    case=fixed_circuit_case(2,b_repeats=2,algorithm_p=.09)
    R,W=simulate(case['input_A'],4,'coherent_request',Noise())
    Wmid=apply_downstream(W,case)
    _,Wout=simulate(case['noisy_output'],4,'coherent_request',Noise())
    assert np.linalg.norm(Wmid-Wout)>1e-3


def test_product_purity_versus_approximate_copy_counts():
    rho,O=noisy_input(1,'plus',.13)
    N=4
    assert abs(np.trace(np.linalg.matrix_power(rho,N)).real-
               np.sum(np.linalg.eigvalsh(rho)**N))<1e-12
    m=resources(2,N,'tree_sequential',Noise())
    assert (m['peak_qubits'],m['register_swaps'],m['native_fredkins'])==(5,3,6)
    p=resources(2,N,'tree_parallel',Noise())
    assert p['peak_qubits']==10


def test_ideal_channel_power_proxy_no_noise():
    for n in (1,2,4):
        assert abs(vcp_pauli_proxy(2,1,0,n)-1)<1e-12


def test_independent_vs_shared_frame_average_noncommuting_nonlinearity():
    p=.3;project=PLUS_DM
    states=[(1-p)*project+p*U@project@U for U in (X,Y,Z)]
    avg=sum(states)/3
    a=np.linalg.matrix_power(avg,2)
    b=sum(np.linalg.matrix_power(r,2) for r in states)/3
    F=lambda z:np.trace(project@z).real/np.trace(z).real
    assert abs(F(a)-F(b))>1e-3
    np.testing.assert_allclose(avg,(1-4*p/3)*project+(4*p/3)*np.eye(2)/2)


@pytest.mark.parametrize('M',[1,2])
@pytest.mark.parametrize('p',[0,.025,.12])
def test_physical_vcp2_channel_superinstrument(M,p):
    """A physical TWO-CSWAP ancilla circuit must implement Pauli p_i^2 weighting."""
    from vcp_physical import vcp2_branch
    from circuits import apply_ideal_pauli_channel
    rho,_=noisy_input(M,'plus',.11)
    for wires in ((0,),tuple(range(M))):
        plus,minus=vcp2_branch(rho,wires,p,M,Noise())
        assert np.linalg.eigvalsh(plus).min()>-1e-10
        assert np.linalg.eigvalsh(minus).min()>-1e-10
        R=plus+minus
        W=plus-minus
        noisy=apply_ideal_pauli_channel(rho,wires,p,M,power=1)
        purified=apply_ideal_pauli_channel(rho,wires,p,M,power=2)
        q=((1-p)**2+p*p/3)**len(wires)
        np.testing.assert_allclose(R,noisy,atol=1e-10)
        np.testing.assert_allclose(W,q*purified,atol=1e-10)


@pytest.mark.parametrize('M',[1,2])
@pytest.mark.parametrize('reps',[1,2])
def test_layerwise_physical_vcp_matches_ideal_pauli_proxy(M,reps):
    from vcp_physical import simulate_layerwise_vcp2,noisy_algorithm_pauli
    R,W,G=simulate_layerwise_vcp2(M,.035,reps,Noise())
    rho,target,_=noisy_algorithm_pauli(M,.035,reps)
    np.testing.assert_allclose(R,rho,atol=1e-10)
    score=np.trace(target@W).real/np.trace(W).real
    assert abs(score-vcp_pauli_proxy(M,reps,.035,power=2))<2e-10


def test_noisy_physical_vcp2_remains_positive_in_each_reported_branch():
    from vcp_physical import simulate_layerwise_vcp2
    R,W,G=simulate_layerwise_vcp2(2,.04,1,Noise(gate_p=.015,control_z_p=.02,readout_q=.025))
    np.testing.assert_allclose(np.trace(R),1,atol=1e-9)
    for sign in (1,-1):
        assert np.linalg.eigvalsh((R+sign*W)/2).min()>-1e-9
