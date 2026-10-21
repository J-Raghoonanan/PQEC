#!/usr/bin/env python3
"""NUMERICAL STUDIES FOR THE REVISED PQES PAPER — A NARRATED EXECUTABLE METHODS SECTION.
====================================================================================

    python pqes_new_numerics/run_simulations.py --profile quick
    python pqes_new_numerics/run_simulations.py --profile standard
    python pqes_new_numerics/run_simulations.py --profile full
    python pqes_new_numerics/run_simulations.py --task phase --profile standard

All CSV tables go to ../data/pqes_new_numerics (unless --data-dir overrides).
Figures are made by plot_results.py from the CSVs only.

I. THE OBJECT SIMULATED
-----------------------
For N independently prepared copies of rho, every architecture A returns the
unnormalized surviving-register operators grouped by REPORTED parity,
tau_+, tau_-, and R=tau_++tau_-, W=tau_+-tau_-. Ideally R=rho, W=rho**N. The
estimator is the ratio of signed sample means; its limiting value, delta-method
variance coefficient and leading ratio bias are

  mu_A=Tr(O W)/t,  v_A=[m2-2 mu m1+mu^2]/t^2,  c_A=(mu-m1)/t^2,  t=Tr W,

with m1, m2 the unsigned first/second moments of the measured outcome (one
register: Tr(O R), Tr(O^2 R); derangement: the register average). The MSE
relative to the ideal target mu0 is b^2+(v+2bc)/S (draft Eq. full_mse_expansion).

II. SIX ARCHITECTURES ON ONE HARDWARE MODEL AND ONE CLOCK (core.py)
-------------------------------------------------------------------
tree_parallel, tree_recycled, tree_sequential (measured SWAP trees: PQES),
coherent_request (streaming one-control VD with resets), wide_cyclic (one
control, all N registers, no resets), derangement (wide_cyclic circuit with O
measured on all N registers, Koczor's cyclic-derangement estimator; each spent
register is read out mid-circuit right after its CSWAP). Every
controlled SWAP is compiled to CNOT-CCX-CNOT with the draft's two-qubit
depolarizing channel; every waiting qubit gets the same idle channel; resets
are charged per actual register reuse; the clock is shared (core docstring).

III. PARAMETER CONVENTIONS (recorded in every CSV row)
-----------------------------------------------------
  source_pauli_p : local depolarizing of the abstract source state, Pauli form
                   (1-p)rho+p/3 sum P rho P  (manuscript Eq. pauli-kraus).
  eps1, eps2, eps3 : replacement depolarizing D_eps after 1-, 2-, 3-qubit gates
                   (draft Eq. noisy_entangling_channel). One-qubit Pauli
                   probability = 3 eps/4. eps3 defaults to eps2.
  readout_q, ancilla_reset_p, data_reset_p, idle_gamma, idle_phi (per unit),
  t_prep, t_meas, t_reset (units of one gate time).
  noise_convention column where a bare p appears: pauli_form (local
                   depolarizing), replacement_form (global depolarizing), z_flip.
Reference hardware (two latency profiles; every conclusion is stated per
profile, never from one guessed value): see REFERENCE below.

IV. SHOT PLANNING
-----------------
Target RMSE EPS=0.03. S=ceil((v+2bc)/(EPS^2-b^2)), infeasible if b^2>=EPS^2,
and S*t^2>=50 is enforced (core.plan_shots). Monte Carlo of the raw ratio
estimator at the planned S gives the achieved RMSE with a 95% CI.

V. STUDIES -> CSV
-----------------
  validation  ideal equivalence of all six arms (any nonzero residual = bug)
  readout     assignment-error visibility and the (1-2q)^(-2(N-2)) penalty
  shots       direct Monte Carlo of the estimator along the q sweep
  mechanisms  one-at-a-time sweeps from zero of every apparatus mechanism
  phase       architecture cost map at fixed RMSE, several cost metrics, with
              no width constraint (primary) and at width budgets 5, 7, 9;
              labelled slices of eps2 and latency
  placement   fixed total depth L (12 entangling, 8 product), compression after
              exactly f*L layers, f in {0,1/4,1/2,3/4,1},
              for every VD arm and for VCP2 at the same cut
  channel_physical  layerwise VCP2 vs output-only state VD at fixed RMSE
  channel_proxy     ideal channel-purification reference (never ranked)
  twirling / twirl_mc / twirl_budget   K-frame sets, per-copy per-shot frame
              Monte Carlo, noisy Cliffords, fixed-budget error before/after
  cycles      repeated virtual cycles: F_t, Z_t, nested copy budget N^t S
  global_depol / global_boundaries  exact global-depolarizing results with
              p_c(N) and p_SN(D,N)
  local_resource  local depolarization: F_N, T_N, leading and sampled MSE at
              fixed copy budget, optimal N, entangled mismatch floor

All operators are exact density matrices within the stated Markovian model.
No figure is produced here; no 'hardware advantage' is asserted anywhere.
"""
from __future__ import annotations
import argparse
import json
import math
import time
import zlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from core import (Hardware,Ledger,ARCH,MEASURED,simulate,simulate_moments,noisy_input,pure_dm,metrics,
                  moments_metrics,resources,plan_shots,outcome_table,mc_rmse,spectral_power,
                  partial_trace,dag,PLUS_DM,X,Y,Z,H,S_GATE,I2,depolarize_subset,
                  derangement_moments,derangement_table)
import circuits
from circuits import cut_case,run_layers,apply_layer,study_layers,ideal_output,count_gates,vcp_pauli_proxy
import vcp_physical as vp

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_DATA=ROOT/'data'/'pqes_new_numerics'
EPS=0.03
BUDGETS=(1e3,1e4,1e5)
EPS2_GRID=(0.002,0.005,0.01,0.02)


def seed_for(*keys):
    """SeedSequence entropy from an explicit key tuple (study, arm, N, shots, ...)."""
    return [zlib.crc32(repr(k).encode()) for k in keys]


def reference(profile='fast',eps2=.005,**kw):
    """Reference hardware. eps1=eps2/10; latency profiles: fast (t_meas=t_reset=1)
    and slow (t_meas=10, t_reset=5) mid-circuit measurement/reset."""
    lat={'fast':dict(t_meas=1,t_reset=1),'slow':dict(t_meas=10,t_reset=5)}[profile]
    base=dict(eps1=eps2/10,eps2=eps2,readout_q=.01,ancilla_reset_p=.005,data_reset_p=.005,
              idle_gamma=5e-4,idle_phi=5e-4,t_prep=3,**lat)
    base.update(kw)
    return Hardware(**base)


def hw_cols(hw):return {('eps3' if k=='eps3' else k):v for k,v in hw.record().items()}


def cost_record(led,mom_row,plan,extra_2q=0,extra_1q=0,copies=None):
    """Per-shot and total resource vector; totals use the planned shots."""
    copies=led.preparations if copies is None else copies
    per=dict(copies_per_shot=copies,cnot_per_shot=led.cnot+extra_2q,ccx_per_shot=led.ccx,
             gates1_per_shot=led.gates1+extra_1q,duration_per_shot=led.duration,
             width=led.peak_qubits,spacetime_per_shot=led.peak_qubits*led.duration)
    S=plan['shots']
    tot={('total_'+k[:-9] if k.endswith('_per_shot') else 'total_'+k):(S*v if plan['feasible'] else np.nan)
         for k,v in per.items() if k!='width'}
    out=dict(per);out.update(tot)
    for C in BUDGETS:out[f'rmse_lead_C{int(C)}']=leading_rmse(mom_row,math.floor(C/copies))
    return out


def leading_rmse(row,S,guard=50.):
    """sqrt(b^2+(v+2bc)/S), reported only where the delta method holds (S t^2>=guard)."""
    b,v,c,t=row['bias'],row['variance_coeff'],row['ratio_bias_coeff'],row['normalization']
    if S<1 or not np.isfinite(v) or S*t*t<guard:return np.nan
    m=b*b+(v+2*b*c)/S
    return math.sqrt(m) if m>0 else np.nan


def write_csv(data,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(data)
    df.to_csv(output,index=False,float_format='%.16g')
    print(f'  saved {output}: {len(df)} rows, {len(df.columns)} columns')
    return df


def arm_eval(rho,N,arch,hw,O,mu0,t_prep=None,post=None,M=None,alg_hw=None):
    """Run one VD arm (optionally followed by an unpurified downstream block
    `post` on every measured register). Returns (metrics row, ledger, table)."""
    M=int(round(math.log2(rho.shape[0]))) if M is None else M
    alg_hw=hw if alg_hw is None else alg_hw
    post=post or []
    mom,led,obj=simulate_moments(rho,N,arch,hw,O,t_prep)
    if arch=='derangement':
        if post:
            Rf,Wf=obj['Rfull'],obj['Wfull'];n=M*N
            for layer in post:
                for i in range(N):   # count each register's post-layer idle once
                    Rf=apply_layer(Rf,layer,M,alg_hw,offset=M*i,n=n,ledger=led)
                    Wf=apply_layer(Wf,layer,M,alg_hw,offset=M*i,n=n)
            obj=dict(obj,Rfull=Rf,Wfull=Wf)
            mom=derangement_moments(Rf,Wf,O,M,N)
            led.duration+=len(post)
        table=derangement_table(obj['Rfull'],obj['Wfull'],O,M,N)
    else:
        R,W=obj
        if post:
            R=run_layers(R,post,M,alg_hw,ledger=led);W=run_layers(W,post,M,alg_hw);led.duration+=len(post)
            mom=dict(t=float(np.trace(W).real),num=float(np.trace(O@W).real),m1=float(np.trace(O@R).real),
                     m2=float(np.trace(O@O@R).real),trR=float(np.trace(R).real))
        table=outcome_table(R,W,O)
    return moments_metrics(mom,mu0),led,table


def mc_validate(table,mu0,shots,reps,seed):
    sg,vals,probs=table
    return mc_rmse(sg,vals,probs,mu0,shots,reps,seed)


def derangement_ok(M,N):return 1+M*N<=9


# =========================================================================== studies
def ideal_validation(profile):
    rows=[]
    cases=[(1,'plus',.11),(1,'tilted',.25),(2,'bell_tilted',.18),(2,'entangled',.09)]
    for M,kind,p in cases:
        rho,target=noisy_input(M,kind,p)
        for N in ((2,4) if profile=='quick' else (2,4,8)):
            P=np.linalg.matrix_power(rho,N)
            for arch in ARCH:
                if arch=='derangement':
                    if not derangement_ok(M,N):continue
                    mom,led,full=simulate_moments(rho,N,arch,Hardware(),target)
                    r_err=abs(mom['m1']-np.trace(target@rho).real)
                    w_err=max(abs(mom['t']-np.trace(P).real),abs(mom['num']-np.trace(target@P).real))
                    norm=mom['t']
                else:
                    R,W=simulate(rho,N,arch,Hardware())
                    r_err=float(np.linalg.norm(R-rho));w_err=float(np.linalg.norm(W-P));norm=float(np.trace(W).real)
                if max(r_err,w_err)>1e-9:raise AssertionError(f'IDEAL INSTRUMENT ERROR: {M,kind,N,arch,r_err,w_err}')
                rows.append(dict(architecture=arch,M=M,N=N,target=kind,source_pauli_p=p,r_err=r_err,w_err=w_err,
                                 normalization=norm,expected_normalization=float(np.trace(P).real),
                                 **resources(M,N,arch)))
    return rows


def readout_study(profile):
    rows=[]
    qvals=[0,.02,.07] if profile=='quick' else [0,.002,.005,.01,.02,.035,.05,.08,.12]
    for M in ([1] if profile=='quick' else [1,2]):
        kind='tilted' if M==1 else 'bell_tilted';p=.18
        rho,target=noisy_input(M,kind,p)
        for N in ([2,4] if profile=='quick' else [2,4,8]):
            purity=float(np.trace(np.linalg.matrix_power(rho,N)).real)
            for q in qvals:
                hw=Hardware(readout_q=q)
                for arch in ARCH:
                    if arch=='derangement' and not derangement_ok(M,N):continue
                    met,led,_=arm_eval(rho,N,arch,hw,target,1.)
                    att=(1-2*q)**(N-1 if arch in MEASURED else 1)
                    if abs(met['normalization']-att*purity)>2e-8:raise AssertionError('Readout attenuation mismatch')
                    rows.append(dict(architecture=arch,M=M,N=N,target=kind,source_pauli_p=p,**hw_cols(hw),**met,
                                     predicted_normalization=att*purity,predicted_visibility=att,**led.record('clock_')))
    return rows


def shots_study(profile,seed):
    """Direct Monte Carlo of the ratio estimator along the readout sweep."""
    rho,O=noisy_input(1,'tilted',.15);p=.15
    rows=[]
    trials=40 if profile=='quick' else (200 if profile=='standard' else 400)
    shotlist=[256,2048] if profile=='quick' else [256,1024,4096,16384]
    qlist=[0,.05] if profile=='quick' else [0,.02,.05,.1]
    for arch in ('tree_sequential','coherent_request'):
        for N in (2,4):
            for q in qlist:
                hw=Hardware(eps2=.005,eps1=.0005,readout_q=q)
                met,led,(sg,vals,probs)=arm_eval(rho,N,arch,hw,O,1.)
                for S in shotlist:
                    rng=np.random.default_rng(seed_for(seed,'shots',arch,N,q,S))
                    c=rng.multinomial(S,probs,size=trials);num=c@(sg*vals);den=c@sg
                    for trial in range(trials):
                        rows.append(dict(architecture=arch,N=N,M=1,source_pauli_p=p,readout_q=q,eps2=hw.eps2,eps1=hw.eps1,
                                         shots=S,trial=trial,**met,
                                         empirical=(float(num[trial]/den[trial]) if den[trial]!=0 else np.nan),
                                         signed_normalization=float(den[trial]/S),zero_denominator=bool(den[trial]==0)))
    return rows


MECHANISMS={'eps2':[0,.002,.005,.01,.02],'eps1':[0,.001,.002,.005],'readout_q':[0,.01,.02,.05],
            'ancilla_reset_p':[0,.01,.02,.05],'data_reset_p':[0,.01,.02,.05],
            'idle_phi':[0,5e-4,1e-3,2e-3,5e-3],'idle_gamma':[0,5e-4,1e-3,2e-3,5e-3],
            't_meas':[1,2,5,10,20],'t_prep':[1,3,10,30]}


def mechanisms_study(profile):
    """One mechanism at a time from zero (latency/prep sweeps at fixed idle 1e-3)."""
    rows=[];M=2;p=.13
    rho,target=noisy_input(M,'bell_tilted',p)
    Ns=[4] if profile=='quick' else [4,8]
    mechs=MECHANISMS if profile!='quick' else {k:v[:3] for k,v in MECHANISMS.items()}
    for N in Ns:
        for mech,vals in mechs.items():
            for val in vals:
                base=dict(idle_gamma=1e-3,idle_phi=1e-3) if mech in ('t_meas','t_prep') else {}
                hw=Hardware(**{**base,mech:val})
                for arch in ARCH:
                    if arch=='derangement' and not derangement_ok(M,N):continue
                    met,led,_=arm_eval(rho,N,arch,hw,target,1.)
                    rows.append(dict(architecture=arch,M=M,N=N,target='bell_tilted',source_pauli_p=p,
                                     mechanism=mech,value=val,**hw_cols(hw),**met,**led.record('clock_')))
    return rows


def phase_study(profile,seed,eps=EPS):
    """Cost map at fixed RMSE for every arm. Axes: readout q and idle rate r
    (idle_gamma=idle_phi=r); slices: eps2 and the latency profile. Every row
    carries the arm's width; plot_results ranks for four cost metrics with no
    width constraint (the primary result) and at the WIDTH_BUDGETS."""
    rows=[];M=2;N=4;p=.14
    rho,target=noisy_input(M,'bell_tilted',p)
    if profile=='quick':
        qvals=[0,.03];rvals=[0,2e-3];slices=[(.005,'fast')]
    elif profile=='standard':
        qvals=np.linspace(0,.05,4);rvals=[0,5e-4,1e-3,2e-3];slices=[(.005,'fast'),(.005,'slow')]
    else:
        qvals=np.linspace(0,.05,6);rvals=[0,2.5e-4,5e-4,1e-3,2e-3,4e-3]
        slices=[(e,lp) for e in EPS2_GRID for lp in ('fast','slow')]
    mc_reps=200 if profile=='quick' else 2000
    for e2,lp in slices:
        for q in qvals:
            for r in rvals:
                hw=reference(lp,e2,readout_q=float(q),idle_gamma=float(r),idle_phi=float(r))
                for arch in ARCH:
                    met,led,table=arm_eval(rho,N,arch,hw,target,1.)
                    plan=plan_shots(met,eps)
                    rec=dict(architecture=arch,M=M,N=N,target='bell_tilted',source_pauli_p=p,latency_profile=lp,
                             idle_rate=float(r),rmse_target=eps,**hw_cols(hw),**met,
                             feasible=plan['feasible'],shots=plan['shots'],shots_delta=plan['shots_delta'],
                             guard_binding=plan['guard_binding'],
                             **led.record('clock_'),**cost_record(led,met,plan))
                    if plan['feasible']:
                        rec.update(mc_validate(table,1.,plan['shots'],mc_reps,seed_for(seed,'phase',arch,e2,lp,float(q),float(r))))
                    rows.append(rec)
    return rows


def placement_study(profile,seed,eps=EPS):
    """Fixed total depth L of the study circuit (circuits.study_layers: L=12
    entangling, L=8 product), cut after exactly f*L layers. VD arms compress rho_A and the remaining layers act
    once on the surviving register (on every register for derangement); VCP2
    purifies the layers before the cut that contain a two-qubit gate and leaves
    the rest unpurified. f=1 is output-only VD / full layerwise VCP2. f=0 is
    unmitigated for VCP2; the VD arms at f=0 compress the ideal initial state
    with a noisy apparatus and then run the whole circuit."""
    rows=[];M=2
    fams=['entangling','product']
    fs=[0,.5,1] if profile=='quick' else [0,.25,.5,.75,1]
    e2s=[.005] if profile=='quick' else (list(EPS2_GRID) if profile=='full' else [.005,.02])
    profiles=['fast'] if profile=='quick' else ['fast','slow']
    Ns=[2,4]
    for fam in fams:
        for e2 in e2s:
            for lp in profiles:
                hw=reference(lp,e2)
                for f in fs:
                    case=cut_case(M,fam,f,hw);O=case['target'];L=case['L']
                    g1A,g2A=case['gates_pre'];g1B,g2B=case['gates_post']
                    common=dict(M=M,family=fam,f=f,L=L,cut=case['cut'],f_realized=case['cut']/L,latency_profile=lp,
                                rmse_target=eps,**hw_cols(hw))
                    for N in Ns:
                        for arch in ARCH:
                            if arch=='derangement' and not derangement_ok(M,N):continue
                            met,led,table=arm_eval(case['rho_A'],N,arch,hw,O,1.,t_prep=case['cut'],post=case['post'],M=M)
                            plan=plan_shots(met,eps)
                            post_copies=N if arch=='derangement' else 1
                            rec=dict(common,protocol=arch,N=N,**met,feasible=plan['feasible'],shots=plan['shots'],
                                     guard_binding=plan['guard_binding'],
                                     algorithm_gates=N*(g1A+g2A)+post_copies*(g1B+g2B),
                                     **led.record('clock_'),
                                     **cost_record(led,met,plan,extra_2q=N*g2A+post_copies*g2B,extra_1q=N*g1A+post_copies*g1B))
                            rows.append(rec)
                    led=Ledger()
                    R,W=vp.simulate_layerwise_vcp2(M,case['pre'],case['post'],hw,ledger=led)
                    met=metrics(R,W,O,1.);plan=plan_shots(met,eps)
                    acc=vp.vcp_accounting(case['pre'],case['post'])
                    rec=dict(common,protocol='vcp2_layerwise',N=2,**met,feasible=plan['feasible'],shots=plan['shots'],
                             guard_binding=plan['guard_binding'],algorithm_gates=acc['noisy_gate_executions'],
                             vcp_control_pool=vp.vcp_pool_size(case['pre'],hw),
                             **acc,**led.record('clock_'),**cost_record(led,met,plan,copies=1))
                    rows.append(rec)
    return rows


def physical_vcp_study(profile,seed,eps=EPS):
    """Full layerwise VCP2 vs output-only state VD (all arms, N=2,4) at fixed RMSE.

    'ideal_apparatus': algorithm gates noisy, everything else perfect;
    'apparatus_gates_10x_better': gadget/control gate errors eps/10, readout,
    resets and idle unchanged; 'physical': the same hardware for both. Columns
    without a prefix record the algorithm/source hardware; app_* columns record
    the apparatus actually used. Every resource type is reported separately;
    there is no single scalar cost."""
    rows=[]
    cases=[(1,'entangling'),(2,'entangling'),(2,'product')] if profile!='quick' else [(2,'entangling')]
    e2s=list(EPS2_GRID) if profile!='quick' else [.005,.02]
    profiles=['fast','slow'] if profile!='quick' else ['fast']
    mc_reps=200 if profile=='quick' else 2000
    for M,fam in cases:
        layers=study_layers(M,fam);O=ideal_output(layers,M);g1,g2=count_gates(layers)
        for e2 in e2s:
            for lp in profiles:
                hwp=reference(lp,e2)
                for app in ('ideal_apparatus','apparatus_gates_10x_better','physical'):
                    hw={'ideal_apparatus':Hardware(t_prep=hwp.t_prep,t_meas=hwp.t_meas,t_reset=hwp.t_reset),
                        'apparatus_gates_10x_better':replace(hwp,eps1=hwp.eps1/10,eps2=hwp.eps2/10,eps3=hwp.e3/10),
                        'physical':hwp}[app]
                    rho=run_layers(circuits.ground(M),layers,M,hwp)
                    common=dict(M=M,family=fam,L=len(layers),latency_profile=lp,apparatus_case=app,rmse_target=eps,
                                **hw_cols(hwp),**{'app_'+k:v for k,v in hw_cols(hw).items()})
                    for N in (2,4):
                        for arch in ARCH:
                            if arch=='derangement' and not derangement_ok(M,N):continue
                            met,led,table=arm_eval(rho,N,arch,hw,O,1.,t_prep=len(layers),M=M)
                            plan=plan_shots(met,eps)
                            rec=dict(common,protocol=arch,N=N,**met,feasible=plan['feasible'],shots=plan['shots'],
                                     guard_binding=plan['guard_binding'],noisy_gate_executions=N*(g1+g2),
                                     main_preparations=N,aux_preparations=0,**led.record('clock_'),
                                     **cost_record(led,met,plan,extra_2q=N*g2,extra_1q=N*g1))
                            if plan['feasible']:rec.update(mc_validate(table,1.,plan['shots'],mc_reps,seed_for(seed,'vcp',M,fam,e2,lp,app,arch,N)))
                            rows.append(rec)
                    led=Ledger()
                    R,W=vp.simulate_layerwise_vcp2(M,layers,[],hw,ledger=led,alg_hw=hwp)
                    met=metrics(R,W,O,1.);plan=plan_shots(met,eps);acc=vp.vcp_accounting(layers,[])
                    rec=dict(common,protocol='vcp2_layerwise',N=2,**met,feasible=plan['feasible'],shots=plan['shots'],
                             guard_binding=plan['guard_binding'],vcp_control_pool=vp.vcp_pool_size(layers,hw),
                             **acc,**led.record('clock_'),**cost_record(led,met,plan,copies=1))
                    if plan['feasible']:
                        rec.update(mc_validate(outcome_table(R,W,O),1.,plan['shots'],mc_reps,seed_for(seed,'vcp',M,fam,e2,lp,app,'vcp')))
                    rows.append(rec)
    return rows


def vcp_proxy_study(profile):
    rows=[]
    for M,fam in ([(2,'entangling')] if profile=='quick' else [(1,'entangling'),(2,'entangling'),(2,'product')]):
        for e2 in EPS2_GRID:
            for N in (1,2,3,4,8):
                rows.append(dict(M=M,family=fam,eps2=e2,eps1=e2/10,N=N,
                                 ideal_layerwise_channel_fidelity=vcp_pauli_proxy(M,None,e2,N,fam,eps1=e2/10),
                                 operational_sampling_modeled=False,hardware_implementation='IDEAL_CHANNEL_ONLY_NOT_PHYSICAL_VCP'))
    return rows


# --------------------------------------------------------------------------- twirling
def _bloch_dm(theta,phi=np.pi/4):
    n=(np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta))
    return .5*(I2+n[0]*X+n[1]*Y+n[2]*Z)


FRAMES=(I2,H,H@S_GATE)   # U in {I,H,HS}: U^dag Z_p(U . U^dag) U maps the Z error to Z, X, Y


def frame_states(theta,p,eps_c):
    """Per-qubit frame-conditioned noisy states rho_f (f=0,1,2), with noisy
    Clifford gates: each nontrivial U and U^dag is followed by D_eps_c."""
    tgt=_bloch_dm(theta);out=[]
    for k,U in enumerate(FRAMES):
        r=U@tgt@dag(U)
        if k and eps_c:r=(1-eps_c)*r+eps_c*I2/2
        r=(1-p)*r+p*Z@r@Z
        r=dag(U)@r@U
        if k and eps_c:r=(1-eps_c)*r+eps_c*I2/2
        out.append(r)
    return tgt,out


def _kron_all(ms):
    out=np.array([[1.]],complex)
    for m in ms:out=np.kron(out,m)
    return out


def _F(P,W):return float(np.trace(P@W).real/np.trace(W).real)


def twirling_study(profile,seed):
    """Exact independent/shared frames and fixed empirical K-frame sets."""
    rows=[]
    Ms=[1,2] if profile=='quick' else [1,2,5]
    ps=[.45,.7] if profile=='quick' else [.45,.55,.7,.8]
    Ks=[1,3,9,27] if profile=='quick' else [1,3,9,27,81,243]
    trials=10 if profile=='quick' else (40 if profile=='standard' else 100)
    for M in Ms:
        for theta in (np.pi/2,np.pi/3):
            for p in ps:
                for eps_c in (0.,.01):
                    tgt,fs=frame_states(theta,p,eps_c);P=_kron_all([tgt]*M)
                    bar1=sum(fs)/3;bar=_kron_all([bar1]*M)
                    deph=_kron_all([fs[0]]*M)
                    for N in (2,4):
                        ind=np.linalg.matrix_power(bar,N)
                        sh=_kron_all([sum(np.linalg.matrix_power(f,N) for f in fs)/3]*M)
                        dp=np.linalg.matrix_power(deph,N)
                        Find=_F(P,ind)
                        common=dict(M=M,theta=theta,p=p,noise_convention='z_flip',eps_clifford=eps_c,N=N)
                        for mode,mat in (('independent_exact',ind),('shared_exact',sh),('untwirled',dp)):
                            rows.append(dict(common,mode=mode,K=0,trial=-1,fidelity=_F(P,mat),normalization=float(np.trace(mat).real),
                                             fidelity_minus_independent=_F(P,mat)-Find,state_trace_distance=np.nan,power_trace_distance=np.nan))
                        Pind=ind/np.trace(ind)
                        for K in Ks:
                            rng=np.random.default_rng(seed_for(seed,'twirl',M,theta,p,eps_c,N,K))
                            for trial in range(trials):
                                fr=rng.integers(0,3,size=(K,M))
                                rk=sum(_kron_all([fs[i] for i in row]) for row in fr)/K
                                mat=np.linalg.matrix_power(rk,N)
                                rows.append(dict(common,mode='fixed_K_empirical',K=K,trial=trial,fidelity=_F(P,mat),
                                                 normalization=float(np.trace(mat).real),fidelity_minus_independent=_F(P,mat)-Find,
                                                 state_trace_distance=.5*np.abs(np.linalg.eigvalsh(rk-bar)).sum(),
                                                 power_trace_distance=.5*np.abs(np.linalg.eigvalsh(mat/np.trace(mat)-Pind)).sum()))
    return rows


def _seq_words(N):
    """Jordan-tree expansion of the sequential accumulator: list of index words
    (weight 1/2^(N-1) each) and the unsigned leaf weights."""
    words=[(0,)]
    for k in range(1,N):words=[w+(k,) for w in words]+[(k,)+w for w in words]
    wR=[2.**-(N-1),2.**-(N-1)]+[2.**-(N-k) for k in range(2,N)]
    return words,np.array(wR[:N])


def _per_qubit_tables(tgt,fs,N,words):
    """For every frame tuple of one qubit across the N copies: <psi|word|psi>,
    Tr(word) for each word, and <psi|rho_f|psi> per copy."""
    tuples=list(np.ndindex(*(3,)*N))
    Tw=np.zeros((len(tuples),len(words)),complex);Tt=np.zeros_like(Tw)   # complex: words are not Hermitian
    for ti,tu in enumerate(tuples):
        for wi,w in enumerate(words):
            m=np.eye(2,dtype=complex)
            for j in w:m=m@fs[tu[j]]
            Tw[ti,wi]=np.trace(tgt@m);Tt[ti,wi]=np.trace(m)
    Fq=np.array([np.trace(tgt@f).real for f in fs])
    return Tw,Tt,Fq


def twirl_mc_study(profile,seed):
    """Fresh independent frames for EVERY copy and EVERY shot (ideal SWAP tree,
    sequential accumulator). Product targets factorize per qubit, so M=5 is
    exact and cheap. Compares with Tr(P rhobar^N)/Tr(rhobar^N) and its v."""
    rows=[]
    Ms=[1,2] if profile=='quick' else [1,2,5]
    ps=[.45,.7]
    shotlist=[100,1000] if profile=='quick' else ([100,1000,10000] if profile=='standard' else [100,1000,10000,100000])
    reps=30 if profile=='quick' else (100 if profile=='standard' else 200)
    for M in Ms:
        for theta in (np.pi/2,np.pi/3):
            for p in ps:
                for eps_c in (0.,.01):
                    tgt,fs=frame_states(theta,p,eps_c)
                    bar1=sum(fs)/3
                    for N in (2,4):
                        words,wR=_seq_words(N);Tw,Tt,Fq=_per_qubit_tables(tgt,fs,N,words)
                        P=_kron_all([tgt]*M);bar=_kron_all([bar1]*M);Wb=np.linalg.matrix_power(bar,N)
                        met=metrics(bar,Wb,P,1.)
                        pw=3**np.arange(N)[::-1]
                        for S in shotlist:
                            rng=np.random.default_rng(seed_for(seed,'twirl_mc',M,theta,p,eps_c,N,S))
                            est=[];den=[];numr=[]
                            for _ in range(reps):
                                fr=rng.integers(0,3,size=(S,M,N))
                                tix=fr@pw                                       # (S,M) tuple index per qubit
                                b=(Tw[tix].prod(axis=1)).sum(axis=1).real/2**(N-1)   # <psi|W_f|psi>
                                t=(Tt[tix].prod(axis=1)).sum(axis=1).real/2**(N-1)   # Tr W_f
                                a=(np.prod(Fq[fr],axis=1)*wR).sum(axis=1)       # <psi|R_f|psi>
                                pr=np.stack([(a+b)/2,(a-b)/2,((1-a)+(t-b))/2,((1-a)-(t-b))/2],axis=1)
                                pr=np.clip(pr,0,None);pr/=pr.sum(axis=1,keepdims=True)
                                u=rng.random(S)[:,None];k=(u>np.cumsum(pr,axis=1)).sum(axis=1)
                                sg=np.array([1,-1,1,-1])[k];o=np.array([1,1,0,0])[k]
                                D=sg.sum();A=(sg*o).sum();est.append(A/D if D!=0 else np.nan);den.append(D/S);numr.append(A/S)
                            est=np.array(est);ok=np.isfinite(est)
                            rows.append(dict(M=M,theta=theta,p=p,noise_convention='z_flip',eps_clifford=eps_c,N=N,shots=S,reps=reps,
                                             mc_mean=float(np.nanmean(est)),mc_sd=float(np.nanstd(est,ddof=1)),
                                             mc_mean_se=float(np.nanstd(est,ddof=1)/np.sqrt(ok.sum())),
                                             mc_t=float(np.mean(den)),mc_t_se=float(np.std(den,ddof=1)/np.sqrt(reps)),
                                             mc_A=float(np.mean(numr)),mc_A_se=float(np.std(numr,ddof=1)/np.sqrt(reps)),
                                             predicted_A=float(met['estimate']*met['normalization']),St2=float(S*met['normalization']**2),
                                             zero_denominators=int((~ok).sum()),
                                             predicted_mu=met['estimate'],predicted_t=met['normalization'],
                                             predicted_sd=float(np.sqrt(met['variance_coeff']/S)),
                                             predicted_ratio_bias=float(met['ratio_bias_coeff']/S),
                                             shared_mu=_F(P,_kron_all([sum(np.linalg.matrix_power(f,N) for f in fs)/3]*M))))
    return rows


def twirl_budget_study(profile):
    """Fixed-copy-budget MSE before (dephasing) and after (independent frames,
    noisy Cliffords) twirling, from the exact moments; target fidelity estimate."""
    rows=[]
    Ms=[1,2] if profile=='quick' else [1,2,5]
    for M in Ms:
        for theta in (np.pi/2,np.pi/3):
            for p in ([.45,.7] if profile=='quick' else [.3,.45,.55,.7,.8]):
                for eps_c in (0.,.005,.01,.02):
                    tgt,fs=frame_states(theta,p,eps_c);P=_kron_all([tgt]*M)
                    for mode,rho1 in (('untwirled',fs[0]),('twirled_independent',sum(fs)/3)):
                        if mode=='untwirled' and eps_c:continue
                        rho=_kron_all([rho1]*M)
                        for N in (1,2,4,8):
                            met=metrics(rho,np.linalg.matrix_power(rho,N),P,1.)
                            rec=dict(M=M,theta=theta,p=p,noise_convention='z_flip',eps_clifford=eps_c,mode=mode,N=N,**met)
                            for C in BUDGETS:rec[f'rmse_lead_C{int(C)}']=leading_rmse(met,math.floor(C/N))
                            rows.append(rec)
    return rows


# --------------------------------------------------------------------------- cycles and global/local closed forms
def _local_depol(rho,M,p):
    for q in range(M):
        out=(1-p)*rho
        for P in (X,Y,Z):
            from core import site_unitary
            U=site_unitary(P,(q,),M);out=out+p/3*U@rho@U
        rho=out
    return rho


def repeated_cycles_study(profile,eps=EPS):
    """Nested signed-ensemble cycles W_{t+1}=[E(W_t)]^N (local and global
    depolarizing). The unsigned physical ensemble output after t cycles is
    E^t(P0) (tree averages are unbiased), which sets Tr(O R_t) in v_t."""
    rows=[]
    powers=[2,4] if profile=='quick' else [2,4,8]
    plist=[.2] if profile=='quick' else [.05,.15,.3,.45]
    cases=[('local',1,'plus'),('local',2,'bell_tilted'),('global',2,'global_D4'),('global',4,'global_D16')]
    for chan,M,kind in cases:
        D=2**M
        if kind in ('plus','bell_tilted'):_,P0=noisy_input(M,kind,0)
        else:P0=np.zeros((D,D),complex);P0[0,0]=1
        for p in plist:
            E=(lambda r:_local_depol(r,M,p)) if chan=='local' else (lambda r:(1-p)*r+p*np.eye(D)/D*np.trace(r))
            for N in powers:
                s=P0.copy();Rt=P0.copy();log10Z=0.
                for t in range(1,(6 if profile=='quick' else 11)):
                    e=E(s);Rt=E(Rt)
                    ev,U=np.linalg.eigh(e);pv=np.clip(ev,0,None)**N;q=float(pv.sum())
                    log10Z=N*log10Z+np.log10(q);s=(U*pv)@dag(U)/q
                    F=float(np.trace(s@P0).real);a=float(np.trace(P0@Rt).real)
                    num=a-2*F*a+F*F;b=F-1.
                    log10v=np.log10(max(num,1e-300))-2*log10Z
                    # plan_shots' rule in log space: infeasible if b^2>=eps^2, else
                    # S=(v+2bc)/(eps^2-b^2) with c=(F-a)/Z^2, and S*Z^2>=50
                    if b*b>=eps*eps:log10S=np.nan
                    else:
                        n2=num+2*b*(F-a)
                        lead=(np.log10(n2)-2*log10Z-np.log10(eps*eps-b*b)) if n2>0 else -np.inf
                        log10S=max(lead,np.log10(50)-2*log10Z)
                    rows.append(dict(channel=chan,noise_convention=('pauli_form' if chan=='local' else 'replacement_form'),
                                     M=M,D=D,target=kind,p=p,N=N,cycle=t,ideal_virtual_fidelity=F,bias=b,
                                     feasible_at_eps=bool(np.isfinite(log10S)),
                                     one_cycle_infidelity=(1-F) if t==1 else np.nan,
                                     log10_signed_normalization=log10Z,log10_variance_coeff=log10v,
                                     log10_shots_for_eps=log10S,log10_copies_budget=t*np.log10(N)+log10S,
                                     log10_swaps_budget=np.log10(float(N)**t-1)+log10S,
                                     trajectory_copies=float(N)**t,rmse_target=eps))
    return rows


def global_closed_forms(D,N,p):
    a=1-p+p/D;b=p/D
    T=a**N+(D-1)*b**N;F=a**N/T
    v=(a-2*a*F+F*F)/T**2
    return F,T,v,a


def global_map(F,p,D,N):
    f=(1-p)*F+p/D
    return f**N/(f**N+(1-f)**N/(D-1)**(N-1))


def global_fixed_points(p,D,N,grid=20001):
    """Stable nontrivial fixed points of the global repeated map (F>1/D)."""
    Fs=np.linspace(1/D+1e-9,1-1e-12,grid);g=global_map(Fs,p,D,N)-Fs
    roots=[]
    for i in np.nonzero(np.sign(g[:-1])!=np.sign(g[1:]))[0]:
        lo,hi=Fs[i],Fs[i+1]
        for _ in range(60):
            mid=(lo+hi)/2
            if np.sign(global_map(mid,p,D,N)-mid)==np.sign(global_map(lo,p,D,N)-lo):lo=mid
            else:hi=mid
        roots.append((lo+hi)/2)
    return roots


def p_saddle_node(D,N):
    lo,hi=0.,1.
    for _ in range(50):
        mid=(lo+hi)/2
        (lo,hi)=(mid,hi) if global_fixed_points(mid,D,N,4001) else (lo,mid)
    return lo


def global_depol_study(profile):
    rows=[];bnd=[]
    Ds=[2,4] if profile=='quick' else [2,4,8]
    pgrid=np.linspace(0,.98,25 if profile=='quick' else 99)
    for D in Ds:
        for N in (1,2,4,8):
            Fit=np.ones_like(pgrid)            # repeated map iterated from the pure target
            for _ in range(20000):
                nxt=global_map(Fit,pgrid,D,N) if N>1 else (1-pgrid)*Fit+pgrid/D
                if np.max(np.abs(nxt-Fit))<1e-14:Fit=nxt;break
                Fit=nxt
            for p,Fi in zip(pgrid,Fit):
                F,T,v,a=global_closed_forms(D,N,p)
                fps=global_fixed_points(p,D,N,4001) if N>1 else []
                rec=dict(D=D,N=N,p=p,noise_convention='replacement_form',F_N=F,T_N=T,variance_coeff=v,input_fidelity=a,
                         repeated_stable_F=float(Fi),repeated_n_fixed_points_interior=len(fps),
                         mixed_point_stable=bool(N*(1-p)<1))
                for C in BUDGETS:   # draft Eq. global_resource_mse, only where S t^2>=50
                    rec[f'mse_lead_C{int(C)}']=(1-F)**2+N*v/C if math.floor(C/N)*T*T>=50 else np.nan
                rows.append(rec)
            if N>1:
                bnd.append(dict(D=D,N=N,p_c=1-1/N,p_SN_numeric=p_saddle_node(D,N),
                                p_SN_formula=(D/(D+2*np.sqrt(D-1)) if N==2 else np.nan)))
    return rows,bnd


def local_resource_study(profile,seed):
    """Local depolarization, |+>^M (closed forms) and an entangled M=2 target
    cos(t)|00>+sin(t)|11>, t=pi/8 (mismatch floor). Leading MSE at copy budget
    C=N*S and directly sampled MSE of the raw ratio estimator."""
    rows=[]
    Ms=[1,2] if profile=='quick' else [1,2,5]
    Nmax=8 if profile=='quick' else 16
    ps=[.05,.3,.7] if profile=='quick' else [.05,.15,.3,.5,.7]
    reps=300 if profile=='quick' else 2000
    targets=[('plus',M) for M in Ms]+[('entangled_pi8',2)]
    for kind,M in targets:
        for p in ps:
            if kind=='plus':
                a=1-2*p/3;b=2*p/3;Fin=a**M;floor=1.
            else:
                th=np.pi/8;psi=np.array([np.cos(th),0,0,np.sin(th)],complex);P0=np.outer(psi,psi.conj())
                rho=_local_depol(P0,2,p);Fin=float(np.trace(P0@rho).real)
                ev,U=np.linalg.eigh(rho);floor=float(abs(U[:,-1].conj()@psi)**2)
            for N in range(1,Nmax+1):
                if kind=='plus':
                    T=(a**N+b**N)**M;F=(1+(b/a)**N)**(-M)
                else:
                    W=np.linalg.matrix_power(rho,N);T=float(np.trace(W).real);F=float(np.trace(P0@W).real/T)
                v=(Fin-2*Fin*F+F*F)/T**2;c=(F-Fin)/T**2;bias=F-1
                probs=np.array([(Fin+T*F)/2,(Fin-T*F)/2,((1-Fin)+T*(1-F))/2,((1-Fin)-T*(1-F))/2])
                probs=np.clip(probs,0,None);probs/=probs.sum()
                sg=np.array([1,-1,1,-1.]);o=np.array([1,1,0,0.])
                rec=dict(target=kind,M=M,p=p,noise_convention='pauli_form',N=N,F_N=F,T_N=T,input_fidelity=Fin,variance_coeff=v,ratio_bias_coeff=c,
                         **{f'St2_C{int(C)}':math.floor(C/N)*T**2 for C in BUDGETS},
                         bias=bias,mismatch_floor=floor,
                         small_p_floor_prediction=(1-p*p*np.sin(4*np.pi/8)**2/9 if kind!='plus' else 1.))
                for C in BUDGETS:
                    S=math.floor(C/N)
                    r_=leading_rmse(dict(bias=bias,variance_coeff=v,ratio_bias_coeff=c,normalization=T),S)
                    rec[f'mse_lead_C{int(C)}']=r_**2 if np.isfinite(r_) else np.nan
                    rng=np.random.default_rng(seed_for(seed,'local',kind,M,p,N,C))
                    cnt=rng.multinomial(S,probs,size=reps);num=cnt@(sg*o);den=cnt@sg
                    ok=den!=0;e=num[ok]/den[ok]
                    rec[f'mse_sampled_C{int(C)}']=float(np.mean((e-1)**2)) if ok.any() else np.nan
                    rec[f'mse_sampled_se_C{int(C)}']=float(np.std((e-1)**2,ddof=1)/np.sqrt(ok.sum())) if ok.sum()>1 else np.nan
                    rec[f'zero_den_frac_C{int(C)}']=float(1-ok.mean())
                rows.append(rec)
    return rows


# =========================================================================== main
TASKS=['validation','readout','shots','mechanisms','phase','placement','channel_physical','channel_proxy',
       'twirling','twirl_mc','twirl_budget','cycles','global_depol','local_resource']


def main():
    parser=argparse.ArgumentParser(description='Self-contained PQES noisy-VD numerics')
    parser.add_argument('--profile',choices=['quick','standard','full'],default='quick')
    parser.add_argument('--task',choices=['all']+TASKS,default='all')
    parser.add_argument('--seed',type=int,default=31)
    parser.add_argument('--rmse',type=float,default=EPS,help='Target RMSE for shot planning')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    args=parser.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    if not 0<args.rmse<1:raise ValueError('rmse must be in (0,1)')
    P=args.profile
    tasks={'validation':lambda:ideal_validation(P),'readout':lambda:readout_study(P),
           'shots':lambda:shots_study(P,args.seed),'mechanisms':lambda:mechanisms_study(P),
           'phase':lambda:phase_study(P,args.seed,args.rmse),
           'placement':lambda:placement_study(P,args.seed,args.rmse),
           'channel_physical':lambda:physical_vcp_study(P,args.seed,args.rmse),
           'channel_proxy':lambda:vcp_proxy_study(P),'twirling':lambda:twirling_study(P,args.seed),
           'twirl_mc':lambda:twirl_mc_study(P,args.seed),'twirl_budget':lambda:twirl_budget_study(P),
           'cycles':lambda:repeated_cycles_study(P,args.rmse),'global_depol':lambda:global_depol_study(P),
           'local_resource':lambda:local_resource_study(P,args.seed)}
    selected=TASKS if args.task=='all' else [args.task]
    print('='*75);print('PQES NEW NUMERICS — COMPILED-GATE NOISY INSTRUMENTS ON ONE CLOCK')
    print(f'profile={P} task={args.task} data={args.data_dir}');print('='*75)
    runtimes={}
    for title in selected:
        print(f'Running {title} ...',flush=True);t0=time.time()
        out=tasks[title]()
        if title=='global_depol':
            write_csv(out[0],args.data_dir/'global_depol.csv');write_csv(out[1],args.data_dir/'global_boundaries.csv')
        else:write_csv(out,args.data_dir/f'{title}.csv')
        runtimes[title]=round(time.time()-t0,1);print(f'  {title}: {runtimes[title]} s')
    mpath=args.data_dir/'manifest.json'
    old=json.loads(mpath.read_text()) if mpath.exists() and args.task!='all' else {}
    manifest=dict(simulator='pqes_new_numerics_v2',profile=P,seed=args.seed,rmse_target=args.rmse,
                  width_budgets_reported=['5','7','9','unconstrained'],delta_method_guard_St2=50,
                  task_profiles={**old.get('task_profiles',{}),**{t:P for t in selected}},
                  task_runtimes_s={**old.get('task_runtimes_s',{}),**runtimes},
                  reference_hardware={lp:reference(lp).record() for lp in ('fast','slow')},
                  eps2_grid=list(EPS2_GRID),copy_budgets=list(BUDGETS),
                  specifications={
                     'gate_noise':'replacement depolarizing D_eps after every gate (draft Eq. noisy_entangling_channel); eps1=eps2/10 in the reference; eps3=eps2',
                     'cswap':'compiled CNOT(b->a) layer, M Toffolis CCX(c,a->b), CNOT layer (M+2 units)',
                     'idle':'amplitude damping idle_gamma and Z flip idle_phi per unit on every waiting data and control qubit',
                     'clock':'gates 1 unit; t_prep/t_meas/t_reset; just-in-time preparation; no feed-forward; ancilla reuse waits for measure+reset+H',
                     'resets':'first use perfect; every actual mid-circuit reuse pays t_reset and the reset error',
                     'readout':'independent classical flip of each recorded ancilla outcome',
                     'source_noise':'Pauli-form local depolarizing source_pauli_p for abstract sources; algorithm circuits use the gate noise above',
                     'measurement':'ideal projective O on the measured register(s)',
                     'vcp':'order-2 Liu-style VCP on the gate support of every layer that contains a two-qubit gate (one-qubit-only layers unpurified, eps1=eps2/10); a fresh control (and auxiliary I/2^k) per purified layer from the smallest round-robin pool that keeps measurement/reset off the main register critical path',
                     'derangement':'wide controlled cyclic shift, O on all N registers averaged (Koczor cyclic derangement); each spent register read out mid-circuit right after its CSWAP (t_meas; observable readouts ideal as in every arm); dense, 1+MN<=9',
                     'study_circuits':'algorithm_blocks M=2: entangling L=12 (first 12 layers of U_B^2 U_A), product L=8 (U_B^3 U_A); M=1 entangling L=6',
                     'p_conventions':'pauli_form (1-p)rho+p/3 sum P rho P: sources, local depolarizing; replacement_form (1-p)rho+p I/D: global depolarizing, gate noise eps; z_flip (1-p)rho+p Z rho Z: dephasing'})
    mpath.write_text(json.dumps(manifest,indent=2))
    print('Done. Plot from CSV with: python pqes_new_numerics/plot_results.py')


if __name__=='__main__':main()
