#!/usr/bin/env python3
"""NUMERICAL STUDIES FOR THE REVISED PQES PAPER — A NARRATED EXECUTABLE METHODS SECTION.
====================================================================================

Run from the parent directory that contains the user's OLD .py files:

    python pqes_new_numerics/run_simulations.py --profile quick
    python pqes_new_numerics/run_simulations.py --profile standard
    python pqes_new_numerics/run_simulations.py --profile full
    python pqes_new_numerics/run_simulations.py --task phase --profile standard

Nothing in this folder imports or modifies the user's old simulation files.
All CSV tables go to ../data/pqes_new_numerics (unless --data-dir overrides).
Figures are NOT created here; use plot_results.py after CSV generation.

====================================================================================
I. WHAT THE ORIGINAL CODE COMPUTED, AND WHAT THIS CODE DOES DIFFERENTLY
====================================================================================

The user's old rho2_purification.py computes the matrix function

            P_N(rho) = rho**N / Tr(rho**N)

and feeds that normalized matrix into a new round of noise. Those are
legitimate *ideal normalized virtual-state dynamics*, but do NOT describe a
physical output state of non-postselected SWAP instrumentation. The old
'P_success = 1.0' is a mathematical flag, not a demonstrated deterministic
physical preparation of rho**N/Tr(rho**N). Similarly, calling copies_used
'(t+1)*2**ell' for t nested cycles misses the N**t physical-tree cost.

Our new simulations explicitly implement the controlled-SWAP instrument,
including the physical quantum control qubit, projected control readout,
discard of the unused register, and outcome-weighted aggregation of branches.
The normalized estimator is formed only when extracting an observable.

For N independently prepared copies of rho, define the unnormalized physical
operators conditioned on total reported parity Omega=+/-:

  tau_+, tau_-.

We save the ordinary and weighted operators:

  R = tau_+ + tau_-,          W = tau_+ - tau_-.

If the apparatus is perfect, regardless of balanced/sequential tree shape:

  R = rho,                   W = rho**N.

For a noisy implementation A, the limiting measured observable is

  mu_A = Tr(O W_A)/t_A,      t_A = Tr(W_A).

This need not equal Tr(O rho**N)/Tr(rho**N). The physical output remains R_A.
The ratio estimator's single-shot delta-method variance coefficient is

  v_A = [Tr(O**2 R_A)-2 mu_A Tr(O R_A)+mu_A**2] / t_A**2,

and its leading finite-sample bias coefficient is

  b_ratio = [mu_A-Tr(O R_A)]/t_A**2.

The mean-squared estimation error relative to the known IDEAL target mu0 is

  MSE ~ (mu_A-mu0)**2 + [v_A +2(mu_A-mu0)*b_ratio]/S

for large shots S (provided the random denominator is stable). We save both
coefficients, plus exact quantum-resource counts and a conservative v-only
shot-target estimate. We NEVER infer advantage from rho**N fidelity alone.

====================================================================================
II. FOUR MATCHED VD INSTRUMENTS
====================================================================================

All implementations consume N input copies and N-1 REGISTER controlled-SWAP
operations, each decomposed into M identical native 3-qubit Fredkin gates.
They all report a signed parity and a surviving M-qubit output. They differ
in coherent width, schedule, and the number/lifetime of ancilla readouts.

1. tree_parallel: balanced SWAP tree; each level measured/discarded before
   proceeding, with enough live registers for an entire layer. No left-side
   waiting while a sibling subtree is prepared.

2. tree_recycled: balanced tree evaluated depth-first, recycling registers
   that have been measured/discarded. The left subtree's surviving quantum
   state must wait while the right subtree is prepared; we dephase it once
   per waiting unit. This is the model for a log-width PQES schedule.

3. tree_sequential: accumulator plus a freshly prepared rho, measured after
   every swap; constant coherent width 2M+1, N-1 ancilla readouts.

4. coherent_request: a REQUEST-like coherent-control VD streaming circuit;
   ONE ancilla persists coherently across the N-1 swaps and is measured ONCE
   at the end. Each spent B data register is discarded before a fresh rho is
   prepared. Same 2M+1 peak width, different control-lifetime/noise exposure.

These are concrete realizations. We do not claim the coherent variant
implements all REQUEST optimizations or compares all possible hardware
compilations. Our native-Fredkin noise model and timestep assumptions must
be stated with any plotted result.

====================================================================================
III. SOURCES OF ERROR (all independently configurable)
====================================================================================

  input_p: local Pauli depolarization of each source rho; target depends on
           plus/product/Bell/entangled state family.
  gate_p: 3-qubit depolarizing noise on (control,A_j,B_j) after each Fredkin.
  control_z_p: stochastic control-Z after every Fredkin. This reduces
               visibility without requiring postselection.
  readout_q: classical parity readout bit flip per reported SWAP measurement.
             There are N-1 such readouts in the tree and one in coherent VD.
  memory_z_p: dephasing of stored data qubits per scheduled waiting unit.
  control_idle_p: extra control-qubit dephasing between coherent-control
                  sequential SWAP modules, not during measured/reinitialized
                  modules; an explicit *hardware-different* model.
  reset_p: phenomenological imperfect data register re-preparation after
           recycling (mixes fresh rho with |0..0><0..0|). Not used by
           fully parallel tree, whose copies are all prepared separately.

Caution: noise strengths correspond to these DEFINITIONS, not to randomized
benchmarking infidelities or device physical error rates unless calibrated.

====================================================================================
IV. NUMERICAL FIGURES AND THE HYPOTHESES THEY TEST
====================================================================================

  validation.csv: Ideal equivalence & measured sign-flip analytics. Any
      nonzero discrepancy is a coding bug; NO plotted novelty claim.

  readout.csv: Compare independent readout errors. Analytically tree
      normalization attenuation c=(1-2q)**(N-1), coherent c=1-2q. Ratio
      cancels the bias but variance grows ~1/c**2. In this model PQES
      must be WORSE for N>2; important counterexample, not inconvenient data.

  gate_noise.csv: Control, Fredkin and memory errors separately; systematic
      bias, effective signed normalization and shot coefficient, four arms.

  phase.csv: A 2D sweep over ancilla readout q and extra coherent-control
      idle error p_idle, using a fixed target and SWAP-gate model. At a FIXED
      RMSE target eps, record whether it can be reached, then estimate total
      native-Fredkin budget S*M*(N-1), plus width/depth. Avoid 'winner'
      comparisons of errors before accounting for variance and bias.

  placement.csv: Divide a known, entangling algorithm U=U_B U_A into blocks.
      Compare end-only: N x (U_A+U_B), VD; versus mid: N x U_A, VD, 1 x U_B.
      Both measured and coherent baselines are allowed to use the SAME
      compression point. Includes gate counts, placement, exact finite-power
      observable, noisy bias and variance. Ensures no PQES-only free lunch.

  shot_check.csv: Sample ACTUAL parity and final observable outcomes (not
      conditional branch means) to verify the expected estimator variance.

  twirling.csv: Test independent versus shared frames across copies, and
      finite empirical K-frame channel approximation. A subset is not the
      same thing as a Monte Carlo implementation of exact channel twirling.

  cycles.csv: Ideal repeated normalized noise + power dynamics, but track
      log10 Z_t and N**t input copies needed for nested signed trajectories.
      This is NOT a circuit-level memory experiment.

  channel_physical.csv: ACTUAL order-two layerwise VCP with auxiliary
      maximally mixed state and two controlled-SWAP passes per Pauli-noise
      location. Compare full signed moments, bias, variance and resources
      to output-only physical state VD2. Does not conflate channel and state
      purification: VCP may perform more controlled gates but fewer ideal
      state preparations. Restricted to stochastic Pauli channels.

  channel_proxy.csv: Ideal layerwise Pauli-channel powering, as a limited
      VCP-inspired theoretical comparator. DOES NOT include physical VCP
      measurements, ancillas, its sampling overhead or native compilation;
      is NEVER ranked in the matched hardware phase diagram.

====================================================================================
V. METHODS, UNCERTAINTY AND VALIDATION
====================================================================================

All quantum operators through M=2 data qubits are simulated exactly using
NumPy density matrices, including the control and both input data registers
(maximum 1+2M qubits per SWAP step). No exponential N-copy density matrix
is allocated. The simulator keeps parity-resolved unnormalized density
operators and contracts them exactly. The coherent cyclic streaming arm
keeps the control--data density matrix across swaps.

These are EXACT probabilities within the stated Markovian noise model, NOT
Monte Carlo noisy-gate trajectories. CSV 'variance_coeff' is the analytical
per-shot CLT coefficient. 'shot_check' separately draws physical outcomes.

Parameters and a reproducible seed are saved in manifest.json. Each task
writes its CSV afresh, never silently appends/repeats an old sweep. This
avoids mixing results with distinct simulator versions.

The absence of real-device gate calibrations means that any regime favoring
PQES is conditional on the stated hypothetical model. No 'hardware advantage'
may be announced until sensitivity and fair compilation are evaluated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from core import (Noise,simulate,noisy_input,pure_dm,metrics,resources,
                  spectral_power,sample_ratio,phase_flip,dag,PLUS_DM,PLUS,X,Y,Z)
from circuits import (fixed_circuit_case,apply_downstream,vcp_pauli_proxy)
from vcp_physical import compare_vcp2_statevd

ARCH=('tree_parallel','tree_recycled','tree_sequential','coherent_request')
ROOT=Path(__file__).resolve().parent.parent
DEFAULT_DATA=ROOT/'data'/'pqes_new_numerics'


def _row(rho,N,arch,model,O,ideal,target=None,extra=None):
    R,W=simulate(rho,N,arch,model)
    m=metrics(R,W,O,ideal)
    res=resources(int(round(np.log2(rho.shape[0]))),N,arch,model)
    record=dict(architecture=arch,M=int(round(np.log2(rho.shape[0]))),N=N,
                input_p=extra.get('input_p',np.nan) if extra else np.nan,
                **asdict(model),**res,**m)
    if target is not None:
        record['input_target_fidelity']=float(np.trace(rho@target).real)
        record['ideal_power_target_fidelity']=float(np.trace(spectral_power(rho,N)@target).real)
    if extra:record.update(extra)
    return record,R,W


def write_csv(data,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(data)
    df.to_csv(output,index=False,float_format='%.16g')
    print(f'  saved {output}: {len(df)} rows, {len(df.columns)} columns')
    return df


def ideal_validation(profile):
    """Algebraic contract tests; fail loudly on any incorrect probability."""
    rows=[]
    cases=[(1,'plus',.11),(1,'tilted',.25),(2,'bell_tilted',.18),(2,'entangled',.09)]
    for M,kind,p in cases:
        rho,target=noisy_input(M,kind,p)
        for N in ((2,4) if profile=='quick' else (2,4,8)):
            for arch in ARCH:
                R,W=simulate(rho,N,arch,Noise())
                r_err=np.linalg.norm(R-rho)
                w_err=np.linalg.norm(W-np.linalg.matrix_power(rho,N))
                if max(r_err,w_err)>1e-9:
                    raise AssertionError(f'IDEAL INSTRUMENT ERROR: {M,kind,N,arch,r_err,w_err}')
                rec,_,_=_row(rho,N,arch,Noise(),target,1,target,
                             dict(target=kind,p_input=p))
                rec.update(r_err=r_err,w_err=w_err,expected_normalization=float(np.trace(np.linalg.matrix_power(rho,N)).real))
                rows.append(rec)
    return rows


def readout_study(profile):
    """Prove a NEGATIVE hardware result: independent sign flips penalize trees."""
    rows=[]
    qvals=[0,.02,.07] if profile=='quick' else [0,.002,.005,.01,.02,.035,.05,.08,.12]
    Mvals=[1] if profile=='quick' else [1,2]
    Nvals=[2,4] if profile=='quick' else [2,4,8]
    for M in Mvals:
        kind='tilted' if M==1 else 'bell_tilted'
        rho,target=noisy_input(M,kind,.18)
        for N in Nvals:
            purity=float(np.trace(np.linalg.matrix_power(rho,N)).real)
            for q in qvals:
                model=Noise(readout_q=q)
                for arch in ARCH:
                    rec,_,_=_row(rho,N,arch,model,target,1,target,dict(target=kind,p_input=.18))
                    attenuation=(1-2*q)**(N-1 if arch!='coherent_request' else 1)
                    rec['predicted_normalization']=attenuation*purity
                    rec['predicted_visibility']=attenuation
                    if abs(rec['normalization']-rec['predicted_normalization'])>2e-8:
                        raise AssertionError('Readout attenuation mismatch')
                    rows.append(rec)
    return rows


def gates_study(profile):
    """Separate noises to identify *why* tree shape or coherent control matters."""
    rows=[]
    M=2;rho,target=noisy_input(M,'bell_tilted',.13)
    Nlist=[4] if profile=='quick' else [2,4,8]
    settings=[('ideal',Noise()),
              ('fredkin_0p01',Noise(gate_p=.01)),
              ('ancilla_z_0p02',Noise(control_z_p=.02)),
              ('readout_0p02',Noise(readout_q=.02)),
              ('memory_0p02',Noise(memory_z_p=.02)),
              ('coherent_idle_0p05',Noise(control_idle_p=.05)),
              ('reset_0p025',Noise(reset_p=.025)),
              ('combined',Noise(gate_p=.008,control_z_p=.008,readout_q=.012,
                                memory_z_p=.012,control_idle_p=.025,reset_p=.01))]
    for N in Nlist:
        for label,model in settings:
            for arch in ARCH:
                rec,_,_=_row(rho,N,arch,model,target,1,target,dict(target='bell_tilted',noise_case=label))
                rows.append(rec)
    return rows


def phase_study(profile,eps,width_cap):
    """Total-cost architecture map. Invalid biased estimators cannot 'win'.

    Hardware hypothesis: shared control requires extra between-module coherent
    idle time; local readout can be noisy. The color of the map changes with
    either parameter. Under a different device gate/noise model it may invert.
    """
    rows=[];M=2;N=4
    rho,target=noisy_input(M,'bell_tilted',.14)
    qvals=[0,.03,.08] if profile=='quick' else np.linspace(0,.10,7 if profile=='standard' else 9)
    idlevals=[0,.025,.08] if profile=='quick' else np.linspace(0,.12,7 if profile=='standard' else 9)
    gates=[.004] if profile=='quick' else ([.006] if profile=='standard' else [.0,.006,.02])
    memoryvals=[.008] if profile=='quick' else ([.01] if profile=='standard' else [0,.015])
    for gate in gates:
        for memory in memoryvals:
            for q in qvals:
                for idle in idlevals:
                    model=Noise(gate_p=float(gate),readout_q=float(q),
                                memory_z_p=float(memory),control_idle_p=float(idle),
                                control_z_p=.003,reset_p=.006)
                    for arch in ARCH:
                        rec,_,_=_row(rho,N,arch,model,target,1,target,dict(target='bell_tilted',rmse_target=eps))
                        b=rec['bias'];v=rec['variance_coeff'];k=rec['ratio_bias_coeff']
                        feasible=bool(np.isfinite(v) and b*b<eps*eps and rec['peak_qubits']<=width_cap)
                        # v-only planning estimate, explicitly *not* full MSE.
                        S=max(1,int(np.ceil(v/(eps*eps-b*b)))) if feasible else None
                        # More accurate leading MSE includes ratio finite-shot bias:
                        v_mse=v+2*b*k if feasible else np.nan
                        S_corrected=(max(1,int(np.ceil(max(v_mse,0)/(eps*eps-b*b))))
                                     if feasible and v_mse>=0 else None)
                        rec.update(width_cap=width_cap,feasible=feasible,shots_asymptotic=S,
                                   shots_mse_leading=S_corrected,
                                   total_fredkins=(S*rec['native_fredkins'] if S is not None else np.nan),
                                   total_preparations=(S*N if S is not None else np.nan),
                                   total_swap_depth=(S*rec['critical_swap_stages'] if S is not None else np.nan))
                        rows.append(rec)
    return rows


def placement_study(profile):
    """Fair mid-vs-end comparisons permit BOTH VD architectures to move.

    Here known U_A prepares noisy rho_A, U_B implements subsequent ideal gates
    with separate per-gate CPTP noise. Output-only distills Lambda_B(rho_A),
    whereas mid-circuit processing distills rho_A first and applies Lambda_B
    ONCE to R and W. No nonlinear operator is fed physically forward.
    """
    rows=[]
    Mvals=[1,2] if profile!='quick' else [2]
    powvals=[2,4] if profile!='quick' else [2,4]
    circuits=[0,1,2,4] if profile!='quick' else [0,2]
    per_gate=[.0,.01,.04] if profile!='quick' else [.025]
    model=Noise(gate_p=.008,control_z_p=.004,readout_q=.01,
                control_idle_p=.018,memory_z_p=.005,reset_p=.004)
    for M in Mvals:
        for repeats in circuits:
            for p in per_gate:
                case=fixed_circuit_case(M,repeats,p)
                O=case['ideal'];target=float(np.trace(O@O).real)
                rhoA=case['input_A'];rhoOut=case['noisy_output']
                for N in powvals:
                    for arch in ARCH:
                        for mode in ('output_only','intermediate'):
                            source=rhoOut if mode=='output_only' else rhoA
                            R,W=simulate(source,N,arch,model)
                            if mode=='intermediate':
                                R=apply_downstream(R,case)
                                W=apply_downstream(W,case)
                            stat=metrics(R,W,O,target)
                            res=resources(M,N,arch,model)
                            record=dict(architecture=arch,M=M,N=N,placement=mode,
                                        b_repeats=repeats,algorithm_gate_p=p,
                                        gates_A=len(case['A']),gates_B=len(case['B']),
                                        algorithm_gates=(N*(len(case['A'])+len(case['B']))
                                                         if mode=='output_only' else N*len(case['A'])+len(case['B'])),
                                        input_fidelity=float(np.trace(source@O).real),
                                        **asdict(model),**res,**stat)
                            rows.append(record)
    return rows


def shots_study(profile,seed):
    """Check actual finite-shot estimator distributions against delta method."""
    rho,O=noisy_input(1,'tilted',.15)
    model=Noise(gate_p=.004,readout_q=.015,control_idle_p=.03)
    rows=[]
    trials=12 if profile=='quick' else 100
    shotlist=[128,512,2048] if profile=='quick' else [256,1024,4096,16384]
    for arch in ('tree_sequential','coherent_request'):
        for N in (2,4):
            R,W=simulate(rho,N,arch,model)
            base=metrics(R,W,O,1.)
            for shots in shotlist:
                for trial in range(trials):
                    draw=sample_ratio(R,W,O,shots,seed=seed+3000*trial+101*N+11*len(arch))
                    rows.append(dict(architecture=arch,N=N,trial=trial,M=1,
                                     **asdict(model),**base,**draw))
    return rows


def twirling_study(profile,seed):
    """Physically distinct random-frame protocols, and the K-frame estimator.

    A fixed |+> input subject to Z-dephasing p becomes, under conjugations
    mapping Z to X/Y/Z, states rho_X, rho_Y, rho_Z. The exact channel twirl
    is bar_rho=(rho_X+rho_Y+rho_Z)/3.

    Independent frame choice on each incoming copy -> (bar_rho)**N.
    ONE frame shared across all copies              -> avg_f(rho_f**N).
    Fixed empirical K-frame channel                 -> (avg_K rho_f)**N.

    These are NOT equal, even though the same three Cliffords are available.
    No fictitious 3**M enumeration cost is counted for randomization.
    """
    rows=[];M=1
    ps=[.1,.3,.45] if profile!='quick' else [.3]
    powers=[2,4,8] if profile!='quick' else [2,4]
    Kvals=[1,2,4,8,16,32,64,128] if profile!='quick' else [1,4,16]
    trials=80 if profile!='quick' else 15
    for p in ps:
        project=PLUS_DM
        options=[]
        for error in (X,Y,Z):
            options.append((1-p)*project+p*error@project@error)
        avg=sum(options)/3
        for N in powers:
            indep_mat=np.linalg.matrix_power(avg,N)
            shared_mat=sum(np.linalg.matrix_power(x,N) for x in options)/3
            for mode,mat in [('independent_exact',indep_mat),('shared_exact',shared_mat)]:
                norm=float(np.trace(mat).real)
                fid=float(np.trace(project@mat).real/norm)
                rows.append(dict(mode=mode,p=p,N=N,M=M,K=0,trial=-1,
                                 fidelity=fid,normalization=norm,
                                 normalization_log10=np.log10(norm),
                                 abs_error_vs_independent=abs(fid-float(np.trace(project@indep_mat).real/np.trace(indep_mat).real))))
            for K in Kvals:
                for trial in range(trials):
                    rng=np.random.default_rng(seed+317*trial+29*K+int(1000*p)+N)
                    samples=rng.integers(0,3,size=K)
                    empirical=sum(options[int(i)] for i in samples)/K
                    mat=np.linalg.matrix_power(empirical,N)
                    norm=float(np.trace(mat).real)
                    fid=float(np.trace(project@mat).real/norm)
                    targetfid=float(np.trace(project@indep_mat).real/np.trace(indep_mat).real)
                    rows.append(dict(mode='fixed_K_empirical',p=p,N=N,M=M,K=K,trial=trial,
                                     fidelity=fid,normalization=norm,
                                     normalization_log10=np.log10(norm),
                                     abs_error_vs_independent=abs(fid-targetfid)))
    return rows


def repeated_cycles_study(profile):
    """Spectral fidelity IS NOT a physical repeated recovery process.

    One ideal normalized noise-filter recurrence:
       s_{t+1}=P_N(E(s_t)); q_t=Tr[E(s_t)**N].
    The signed physical realization must nest all earlier branch signs:
       Z_{t+1}=Z_t**N * q_t ; cost_per_trajectory=N**t copies.
    Track log10 Z_t to avoid numerical underflow. No tomography claimed.
    """
    rows=[]
    powers=[2,4] if profile!='quick' else [2]
    p_list=[.08,.25,.4] if profile!='quick' else [.2]
    for M,kind in [(1,'plus'),(2,'bell_tilted')]:
        target=pure_dm(PLUS.reshape(-1)) if M==1 else None
        if M==2:
            _,target=noisy_input(M,kind,0)
        initial=target
        for p in p_list:
            for N in powers:
                s=initial.copy();log10Z=0.
                for t in range(1,(7 if profile=='quick' else 10)):
                    # Local stochastic Pauli depolarization and power filter.
                    # E_p applied to current normalized virtual state.
                    noisy=s.copy()
                    for q in range(M):
                        from core import embed_pauli
                        update=(1-p)*noisy
                        for pauli in 'xyz':
                            U=embed_pauli(pauli,q,M)
                            update+=p/3*(U@noisy@U)
                        noisy=update
                    evals,evecs=np.linalg.eigh(noisy)
                    powvals=np.clip(evals,0,None)**N
                    q_norm=float(powvals.sum())
                    log10Z=N*log10Z+np.log10(q_norm)
                    s=(evecs*powvals)@dag(evecs)/q_norm
                    fidelity=float(np.trace(s@target).real)
                    rows.append(dict(M=M,target=kind,p=p,N=N,cycle=t,
                                     ideal_virtual_fidelity=fidelity,
                                     log10_signed_normalization=log10Z,
                                     log10_conservative_sampling_factor=-2*log10Z,
                                     trajectory_copies=N**t,trajectory_swaps=N**t-1,
                                     log10_copies=np.log10(N)*t))
    return rows


def vcp_proxy_study(profile):
    """Ideal channel-level VCP diagnostic, deliberately NOT ranked with circuits."""
    rows=[]
    for M in ([2] if profile=='quick' else [1,2]):
        for p in ([.02] if profile=='quick' else [.005,.02,.06,.12]):
            for repeats in ([1,2] if profile=='quick' else [1,2,4]):
                for N in ([1,2,4] if profile=='quick' else [1,2,4,8]):
                    rows.append(dict(M=M,algorithm_gate_p=p,b_repeats=repeats,N=N,
                        ideal_layerwise_pauli_channel_fidelity=
                                    vcp_pauli_proxy(M,repeats,p,power=N),
                        operational_sampling_modeled=False,
                        hardware_implementation='IDEAL_CHANNEL_ONLY_NOT_PHYSICAL_VCP'))
    return rows



def physical_vcp_study(profile,eps):
    """Order-two VCP PHYSICAL SUPERINSTRUMENT versus output-only state VD2.

    This implements the actual two-swap Pauli-channel VCP with a maximally
    mixed auxiliary input, not just the algebraic p_i^2 result. The same
    channel E acts on both registers, once each. Apparatus noise is added to
    every native Fredkin and ancilla readout in BOTH protocol families.

    Fair comparison requirements:
      - same ideal algorithm and stochastic Pauli channel per gate;
      - each state VD2 trajectory calls each channel twice on two complete
        state preparations, while VCP2 consumes two channel copies per gate;
      - VCP2 uses 2M Fredkins PER noise location; stateVD2 M total;
      - VCP2 uses one target-state preparation, stateVD2 uses two;
      - we show all resource counts separately; one cannot collapse them
        into a single scalar 'cost' without device-specific time prices.

    Benchmark projectors are known and measured ideally. At zero apparatus
    noise the VCP2 signed estimator equals ideal layerwise Pauli power.
    """
    rows=[]
    if profile=='quick':
        Ms=[1,2]; ps=[.03]; repetitions=[1,2]
    elif profile=='standard':
        Ms=[1,2]; ps=[.01,.03,.08]; repetitions=[1,2]
    else:
        Ms=[1,2]; ps=[.005,.01,.03,.06,.12]; repetitions=[1,2,4]
    settings=[('ideal_apparatus',Noise()),
              ('native_fredkin_noise',Noise(gate_p=.008)),
              ('combined_physical_noise',Noise(gate_p=.008,control_z_p=.005,readout_q=.012))]
    for M in Ms:
        for p in ps:
            for r in repetitions:
                for label,model in settings:
                    for rec in compare_vcp2_statevd(M,p,r,model):
                        rec.update(apparatus_case=label,rmse_target=eps,**asdict(model))
                        b=rec['bias'];v=rec['variance_coeff']
                        feasible=np.isfinite(v) and b*b<eps*eps
                        nshots=int(np.ceil(v/(eps*eps-b*b))) if feasible else None
                        rec.update(feasible=feasible,shots_asymptotic=nshots,
                                   total_native_fredkins=(nshots*rec['native_fredkins'] if nshots else np.nan),
                                   total_channel_calls=(nshots*rec['noisy_channel_calls'] if nshots else np.nan),
                                   total_target_preps=(nshots*rec['source_state_preparations'] if nshots else np.nan))
                        rows.append(rec)
    return rows


def main():
    parser=argparse.ArgumentParser(description='Self-contained PQES noisy-VD numerics')
    parser.add_argument('--profile',choices=['quick','standard','full'],default='quick')
    parser.add_argument('--task',choices=['all','validation','readout','gate_noise','phase',
                                           'placement','shots','twirling','cycles','channel_proxy','channel_physical'],default='all')
    parser.add_argument('--seed',type=int,default=31)
    parser.add_argument('--rmse',type=float,default=.12,
                        help='Target RMSE used for planning cost (not a rigorous finite-shot CI)')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--width-cap',type=int,default=7,
                        help='Maximum available coherent qubits for phase diagram; M=2,N=4 uses default 7')
    args=parser.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    if not 0<args.rmse<1:raise ValueError('rmse must be in (0,1)')
    tasks={'validation':lambda:ideal_validation(args.profile),
           'readout':lambda:readout_study(args.profile),
           'gate_noise':lambda:gates_study(args.profile),
           'phase':lambda:phase_study(args.profile,args.rmse,args.width_cap),
           'placement':lambda:placement_study(args.profile),
           'shots':lambda:shots_study(args.profile,args.seed),
           'twirling':lambda:twirling_study(args.profile,args.seed),
           'cycles':lambda:repeated_cycles_study(args.profile),
           'channel_proxy':lambda:vcp_proxy_study(args.profile),
           'channel_physical':lambda:physical_vcp_study(args.profile,args.rmse)}
    selected=tasks.keys() if args.task=='all' else [args.task]
    manifest=dict(simulator='pqes_new_numerics_v1',profile=args.profile,seed=args.seed,
                  rmse_target=args.rmse,width_cap=args.width_cap,selected_tasks=list(selected),
                  implementation='actual noisy measured instrument and coherent-control VD; VCP ideal proxy only',
                  specifications={
                     'data_noise':'local stochastic Pauli channel',
                     'fredkin_noise':'native 3-qubit depolarizing AFTER each elementary Fredkin',
                     'control_noise':'stochastic control Z, optional extra idle for coherent ancilla',
                     'readout':'independent classical outcome flips',
                     'memory':'Markovian Z dephasing per idle step',
                     'reset':'mix fresh state with zero-state projector on reused registers',
                     'measurement':'ideal projective O, with finite-shot parity sampling for selected cases',
                     'unit_cost':'one register-level SWAP step; M native 3-qubit Fredkins',
                     'vcp':'physical order-2 Pauli superinstrument separately benchmarked; ideal Pauli power proxy retained'})
    (args.data_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('='*75)
    print('PQES NEW NUMERICS — EXACT PHYSICAL INSTRUMENT COMPARISONS')
    print(f'profile={args.profile} task={args.task} data={args.data_dir}')
    print('='*75)
    for title in selected:
        print(f'Running {title} ...',flush=True)
        write_csv(tasks[title](),args.data_dir/f'{title}.csv')
    print('Done. Plot from CSV with: python pqes_new_numerics/plot_results.py')


if __name__=='__main__':main()
