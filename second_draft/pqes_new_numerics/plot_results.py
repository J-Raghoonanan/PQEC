#!/usr/bin/env python3
"""PQES publication plotting — LOAD CSVs, NEVER recompute quantum dynamics.

    python pqes_new_numerics/plot_results.py
    python pqes_new_numerics/plot_results.py --figure placement

Every panel states its slice (eps2, latency profile, N, M, ...) on the figure.
Overlapping curves are drawn with distinct markers and line styles, so a hidden
series is never silent; a coincidence of arms is noted in the panel instead.
If a CSV is missing, the figure is skipped and the task to run is printed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from scipy import stats

_installed={f.name for f in font_manager.fontManager.ttflist}
_SERIF=next((f for f in ('Times New Roman','Liberation Serif','DejaVu Serif') if f in _installed),'DejaVu Serif')
try:plt.style.use('seaborn-v0_8-paper')
except OSError:pass
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':10,
                     'ytick.labelsize':10,'legend.fontsize':9,'figure.titlesize':13,'font.family':_SERIF,
                     'mathtext.fontset':'dejavuserif','axes.linewidth':1.,'lines.linewidth':1.8,
                     'lines.markersize':6,'pdf.fonttype':42,'ps.fonttype':42})

ARCH=('tree_parallel','tree_recycled','tree_sequential','coherent_request','wide_cyclic','derangement')
LABEL={'tree_parallel':'Parallel SWAP tree','tree_recycled':'Recycled SWAP tree','tree_sequential':'Measured accumulator',
       'coherent_request':'Coherent streaming VD','wide_cyclic':'Wide cyclic VD','derangement':'Cyclic derangement',
       'vcp2_layerwise':'Layerwise VCP2'}
SHORT={'tree_parallel':'P','tree_recycled':'R','tree_sequential':'S','coherent_request':'C','wide_cyclic':'W',
       'derangement':'D','vcp2_layerwise':'V'}
COLORS={'tree_parallel':'#2E86AB','tree_recycled':'#00845A','tree_sequential':'#F18F01','coherent_request':'#7B3294',
        'wide_cyclic':'#C73E1D','derangement':'#5C5C5C','vcp2_layerwise':'#1B998B'}
MARKER={'tree_parallel':'o','tree_recycled':'s','tree_sequential':'^','coherent_request':'D','wide_cyclic':'v',
        'derangement':'P','vcp2_layerwise':'X'}
LS={'tree_parallel':'-','tree_recycled':'--','tree_sequential':':','coherent_request':'-.','wide_cyclic':(0,(5,1)),
    'derangement':(0,(1,1)),'vcp2_layerwise':'-'}
ROOT=Path(__file__).resolve().parent.parent
DATA_DEFAULT=ROOT/'data'/'pqes_new_numerics'
FIG_DEFAULT=ROOT/'figures'/'pqes_new_numerics'


def _load(data,name):
    path=data/f'{name}.csv'
    if not path.is_file():
        print(f'  Missing {path}. To generate: --task {name}');return None
    return pd.read_csv(path)


def _save(fig,dest,stem):
    dest.mkdir(parents=True,exist_ok=True)
    for typ in ('pdf','png'):
        fname=dest/f'{stem}.{typ}';fig.savefig(fname,bbox_inches='tight',dpi=300);print('  Saved',fname)
    plt.close(fig)


def _panel(ax,letter,title,logx=False,logy=False):
    ax.set_title(f'({letter}) {title}',loc='left')
    if logx:ax.set_xscale('log')
    if logy:ax.set_yscale('log')
    ax.spines[['top','right']].set_visible(False);ax.grid(True,alpha=.25,linewidth=.6)


def _arch_kw(a,**kw):
    return dict(color=COLORS[a],marker=MARKER[a],linestyle=LS[a],label=LABEL[a],**kw)


def _feasible(s):return s.astype(str).str.lower()=='true'


# ------------------------------------------------------------------ Fig A
def fig_readout(data,dest):
    d=_load(data,'readout');s=_load(data,'shots')
    if d is None or s is None:return
    fig,axes=plt.subplots(2,2,figsize=(12,9));a,b,c,e=axes.ravel()
    sub=d[d.M==d.M.max()]
    for N,ls in ((4,'-'),(8,'--')):
        for arch in ('tree_sequential','coherent_request'):
            cur=sub[(sub.N==N)&(sub.architecture==arch)].sort_values('readout_q')
            if cur.empty or not (cur.readout_q==0).any():continue
            t0=float(cur[cur.readout_q==0].normalization.iloc[0])
            a.plot(cur.readout_q,cur.normalization/t0,color=COLORS[arch],marker=MARKER[arch],linestyle=ls,
                   label=f'{LABEL[arch]}, N={N}')
    a.text(.02,.04,'All measured trees coincide with the accumulator;\nwide cyclic and derangement with coherent VD',
           transform=a.transAxes,fontsize=8)
    _panel(a,'a','Signed visibility vs readout error');a.set_xlabel('Readout error $q$');a.set_ylabel('$t(q)/t(0)$');a.legend()
    for N,col in ((4,'#2E86AB'),(8,'#C73E1D')):
        m=sub[(sub.N==N)&(sub.architecture=='tree_sequential')].sort_values('readout_q')
        co=sub[(sub.N==N)&(sub.architecture=='coherent_request')].sort_values('readout_q')
        if m.empty or len(m)!=len(co):continue
        b.plot(m.readout_q,m.variance_coeff.to_numpy()/co.variance_coeff.to_numpy(),'o',color=col,label=f'simulated, N={N}')
        b.plot(m.readout_q,(1-2*m.readout_q)**(-2*(N-2)),'-',color=col,lw=1,label=f'$(1-2q)^{{-2(N-2)}}$, N={N}')
    _panel(b,'b','Measured-tree / coherent variance ratio',logy=True);b.set_xlabel('Readout error $q$')
    b.set_ylabel(r'$v_{\rm tree}/v_{\rm coh}$');b.legend()
    s4=s[s.N==4]
    for (arch,q),g in s4.groupby(['architecture','readout_q']):
        if q not in (0,.05,.1):continue
        agg=g.groupby('shots').agg(sd=('empirical','std'),n=('empirical','count'),v=('variance_coeff','first')).reset_index()
        clt=np.sqrt(agg.v/agg.shots);r=agg.sd/clt
        lo=np.sqrt((agg.n-1)/stats.chi2.ppf(.975,agg.n-1));hi=np.sqrt((agg.n-1)/stats.chi2.ppf(.025,agg.n-1))
        c.errorbar(agg.shots*(1.06 if arch=='coherent_request' else 1),r,yerr=[r-r*lo,r*hi-r],color=COLORS[arch],
                   marker=MARKER[arch],linestyle=':' if q else '-',capsize=3,label=f'{LABEL[arch]}, q={q:g}')
    c.axhline(1,color='k',lw=.8)
    _panel(c,'c','Monte Carlo sd / delta-method sd (95% CI), N=4',logx=True);c.set_xlabel('Shots $S$');c.set_ylabel(r'sd$(\hat\mu)/\sqrt{v/S}$');c.legend(fontsize=7)
    for arch in ('tree_sequential','coherent_request'):
        g=s4[(s4.architecture==arch)&np.isclose(s4.readout_q,.05)]
        agg=g.groupby('shots').agg(m=('empirical','mean'),se=('empirical','sem'),mu=('estimate','first'),cc=('ratio_bias_coeff','first')).reset_index()
        e.errorbar(agg.shots,(agg.m-agg.mu)*agg.shots,yerr=1.96*agg.se*agg.shots,color=COLORS[arch],marker=MARKER[arch],capsize=3,
                   label=f'{LABEL[arch]}: MC')
        e.plot(agg.shots,agg.cc,color=COLORS[arch],lw=1,label=f'{LABEL[arch]}: $c_A$')
    _panel(e,'d','Ratio bias x S at q=0.05, N=4 (95% CI)',logx=True);e.set_xlabel('Shots $S$');e.set_ylabel(r'$S(\bar{\hat\mu}-\mu_A)$');e.legend(fontsize=8)
    fig.suptitle(f'Readout errors and estimator statistics (M={int(sub.M.max())} bell_tilted for a-b; M=1 tilted, eps2=0.005 for c-d)')
    fig.tight_layout();_save(fig,dest,'PQES_FigA_readout_and_shots')


def fig_mechanisms(data,dest):
    d=_load(data,'mechanisms')
    if d is None:return
    d=d[d.N==4];mechs=list(dict.fromkeys(d.mechanism))
    fig,axes=plt.subplots(2,len(mechs),figsize=(2.6*len(mechs),6.2),squeeze=False)
    for j,mech in enumerate(mechs):
        g=d[d.mechanism==mech]
        for arch in ARCH:
            cur=g[g.architecture==arch].sort_values('value')
            if cur.empty:continue
            base=cur.iloc[0]
            axes[0,j].plot(cur.value,cur.bias,**_arch_kw(arch,markersize=4,lw=1.2))
            axes[1,j].plot(cur.value,cur.variance_coeff/base.variance_coeff,**_arch_kw(arch,markersize=4,lw=1.2))
        _panel(axes[0,j],chr(97+j),mech);axes[0,j].set_xlabel(mech);axes[1,j].set_xlabel(mech)
        axes[1,j].spines[['top','right']].set_visible(False);axes[1,j].grid(True,alpha=.25)
    axes[0,0].set_ylabel(r'bias $\mu_A-\mu_0$');axes[1,0].set_ylabel(r'$v/v$ at sweep start')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.5,1.04))
    fig.suptitle('One-at-a-time apparatus mechanisms (M=2 bell_tilted, source p=0.13, N=4; t_meas and t_prep sweeps at idle 1e-3/unit)',y=1.09)
    fig.tight_layout();_save(fig,dest,'PQES_FigA2_apparatus_mechanisms')


# ------------------------------------------------------------------ Fig B
METRICS=(('shots','Shots at RMSE 0.03'),('total_cnot','Two-qubit gates'),('total_duration','Duration (gate units)'),
         ('total_spacetime','Space-time volume'))


def _winner_map(sub,metric,capped):
    qs=np.sort(sub.readout_q.unique());rs=np.sort(sub.idle_rate.unique())
    win=np.full((len(rs),len(qs)),np.nan);txt=np.full(win.shape,'',dtype=object)
    for yi,r in enumerate(rs):
        for xi,q in enumerate(qs):
            rr=sub[np.isclose(sub.readout_q,q)&np.isclose(sub.idle_rate,r)&_feasible(sub.feasible)]
            if capped:rr=rr[rr.within_cap.astype(str).str.lower()=='true']
            if rr.empty:continue
            rr=rr.sort_values([metric,'width','clock_duration'])
            best=rr.iloc[0];win[yi,xi]=ARCH.index(best.architecture);txt[yi,xi]=SHORT[best.architecture]
            if len(rr)>1 and rr.iloc[1][metric]<=1.05*best[metric]:txt[yi,xi]+='*'
    return qs,rs,win,txt


def fig_phase(data,dest):
    d=_load(data,'phase')
    if d is None:return
    e2=.005 if np.isclose(d.eps2,.005).any() else float(d.eps2.iloc[0])
    profiles=[p for p in ('fast','slow') if (d.latency_profile==p).any()]
    cmap=ListedColormap([COLORS[a] for a in ARCH]);cmap.set_bad('#eeeeee')
    for capped in (True,False):
        fig,axes=plt.subplots(len(profiles),len(METRICS),figsize=(3.6*len(METRICS),3.4*len(profiles)),squeeze=False)
        for i,lp in enumerate(profiles):
            sub=d[np.isclose(d.eps2,e2)&(d.latency_profile==lp)]
            for j,(metric,title) in enumerate(METRICS):
                ax=axes[i,j];qs,rs,win,txt=_winner_map(sub,metric,capped)
                ax.imshow(win,origin='lower',aspect='auto',vmin=-.5,vmax=len(ARCH)-.5,cmap=cmap)
                for yi in range(len(rs)):
                    for xi in range(len(qs)):ax.text(xi,yi,txt[yi,xi],ha='center',va='center',fontsize=7,color='white')
                ax.set_xticks(range(len(qs)),[f'{q:.2f}' for q in qs],fontsize=7)
                ax.set_yticks(range(len(rs)),[f'{r*1e3:g}' for r in rs],fontsize=7)
                ax.set_title(f'({chr(97+i*len(METRICS)+j)}) {title}\n{lp} measurement',fontsize=10,loc='left')
                ax.set_xlabel('Readout error $q$');ax.set_ylabel(r'Idle rate ($10^{-3}$/unit)')
        handles=[plt.Line2D([0],[0],marker='s',linestyle='',color=COLORS[a],label=f'{SHORT[a]}: {LABEL[a]}',markersize=8) for a in ARCH]
        fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.5,1.05),fontsize=8)
        cap=f'width <= {int(d.width_cap.iloc[0])} qubits' if capped else 'no width cap'
        fig.suptitle(f'Lowest-cost arm at RMSE 0.03 for each cost metric; M=2, N=4, eps2={e2:g}, eps1=eps2/10, {cap}. '
                     '* = runner-up within 5%',y=1.1,fontsize=11)
        fig.tight_layout();_save(fig,dest,'PQES_FigB_architecture_cost_map'+('' if capped else '_uncapped'))
    f=d[_feasible(d.feasible)&d.rmse.notna()]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for arch in ARCH:
        g=f[f.architecture==arch]
        ax[0].errorbar(np.arange(len(g))+ARCH.index(arch)*len(g),g.rmse,yerr=[g.rmse-g.rmse_lo,g.rmse_hi-g.rmse],fmt='.',
                       color=COLORS[arch],label=LABEL[arch],ms=3,elinewidth=.5)
    ax[0].axhline(float(d.rmse_target.iloc[0]),color='k',lw=1)
    _panel(ax[0],'a','Achieved RMSE at the planned shots (95% CI), all cells');ax[0].set_xlabel('cell index (grouped by arm)')
    ax[0].set_ylabel('Monte Carlo RMSE');ax[0].legend(fontsize=7,ncol=2)
    ax[1].hist((f.rmse/f.rmse_target),bins=40,color='#555555')
    _panel(ax[1],'b','RMSE / target over all feasible cells');ax[1].set_xlabel('achieved / target')
    fig.tight_layout();_save(fig,dest,'PQES_FigB2_planning_validation')


# ------------------------------------------------------------------ Fig C
def fig_placement(data,dest):
    d=_load(data,'placement')
    if d is None:return
    e2=.005 if np.isclose(d.eps2,.005).any() else float(d.eps2.iloc[0])
    fams=list(dict.fromkeys(d.family))
    fig,axes=plt.subplots(len(fams),4,figsize=(16,4.2*len(fams)),squeeze=False)
    for i,fam in enumerate(fams):
        sub=d[(d.family==fam)&np.isclose(d.eps2,e2)&(d.latency_profile=='fast')]
        for prot in list(ARCH)+['vcp2_layerwise']:
            cur=sub[(sub.protocol==prot)&((sub.N==4)|(sub.protocol=='vcp2_layerwise'))].sort_values('f')
            if prot=='derangement' and cur.empty:cur=sub[(sub.protocol==prot)&(sub.N==2)].sort_values('f')
            if cur.empty:continue
            kw=_arch_kw(prot)
            axes[i,0].plot(cur.f,cur.bias.abs(),**kw)
            axes[i,1].plot(cur.f,cur.variance_coeff,**kw)
            ok=cur[_feasible(cur.feasible)]
            axes[i,2].plot(ok.f,ok.total_cnot,**kw)
            axes[i,3].plot(cur.f,cur['rmse_lead_C10000'],**kw)
        axes[i,0].axhline(float(d.rmse_target.iloc[0]),color='k',lw=.8,ls='--')
        axes[i,3].axhline(float(d.rmse_target.iloc[0]),color='k',lw=.8,ls='--')
        L=int(sub.L.iloc[0])
        _panel(axes[i,0],chr(97+4*i),f'{fam}: |bias| (L={L} layers)',logy=True);axes[i,0].set_ylabel(r'$|\mu_A-\mu_0|$')
        _panel(axes[i,1],chr(98+4*i),f'{fam}: variance coefficient',logy=True);axes[i,1].set_ylabel('$v_A$')
        _panel(axes[i,2],chr(99+4*i),f'{fam}: two-qubit gates to RMSE 0.03',logy=True);axes[i,2].set_ylabel('total two-qubit gates')
        _panel(axes[i,3],chr(100+4*i),f'{fam}: RMSE at C=$10^4$ copies',logy=True);axes[i,3].set_ylabel('leading RMSE')
        for ax in axes[i]:ax.set_xlabel('compression point $L_A/L$')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=7,frameon=False,bbox_to_anchor=(.5,1.03))
    fig.suptitle(f'Placement at fixed total depth: M=2, N=4 (derangement N=4), eps2={e2:g}, fast measurement. '
                 'Missing points in (c,g): infeasible (|bias|>=0.03). VCP2: purify layers before the cut.',y=1.06)
    fig.tight_layout();_save(fig,dest,'PQES_FigC_compression_placement')


# ------------------------------------------------------------------ Fig D
def fig_twirling(data,dest):
    tw=_load(data,'twirling');mc=_load(data,'twirl_mc');bud=_load(data,'twirl_budget')
    if tw is None or mc is None or bud is None:return
    fig,axes=plt.subplots(2,2,figsize=(13,9.5));a,b,c,e=axes.ravel()
    p0=.45
    for M,col in zip(sorted(tw.M.unique()),('#2E86AB','#F18F01','#7B3294')):
        g=tw[(tw['mode']=='fixed_K_empirical')&(tw.M==M)&np.isclose(tw.theta,np.pi/2)&np.isclose(tw.p,p0)&(tw.N==4)&(tw.eps_clifford==0)]
        agg=g.groupby('K').agg(m=('fidelity_minus_independent','mean'),se=('fidelity_minus_independent','sem'),
                               td=('power_trace_distance','mean'),tdse=('power_trace_distance','sem')).reset_index()
        a.errorbar(agg.K,agg.m.abs(),yerr=1.96*agg.se,color=col,marker='o',capsize=3,label=f'|mean $F_K-F_{{ind}}$|, M={M}')
        a.errorbar(agg.K,agg.td,yerr=1.96*agg.tdse,color=col,marker='s',ls='--',capsize=3,label=f'power-state trace distance, M={M}')
    _panel(a,'a',f'Fixed K-frame sets vs independent twirl (p={p0}, theta=pi/2, N=4)',logx=True,logy=True)
    a.set_xlabel('frames in the fixed set $K$');a.set_ylabel('deviation (95% CI)');a.legend(fontsize=7)
    for (N,th),ls in zip([(2,np.pi/3),(4,np.pi/3)],('-','--')):
        g=mc[(mc.M==2)&np.isclose(mc.theta,th)&np.isclose(mc.p,p0)&(mc.N==N)&(mc.eps_clifford==.01)].sort_values('shots')
        if g.empty:continue
        b.errorbar(g.shots,g.mc_mean,yerr=g.mc_sd,color='#2E86AB' if N==2 else '#C73E1D',marker='o',ls=ls,capsize=3,
                   label=f'per-copy random frames, N={N} (mean $\\pm$ sd)')
        b.axhline(g.predicted_mu.iloc[0],color='#2E86AB' if N==2 else '#C73E1D',lw=.8)
        b.axhline(g.shared_mu.iloc[0],color='#2E86AB' if N==2 else '#C73E1D',lw=.8,ls=':')
        bad=g[g.St2<50]
        b.scatter(bad.shots,bad.mc_mean,s=90,facecolors='none',edgecolors='k')
    _panel(b,'b','Shot-level frame Monte Carlo, M=2, theta=pi/3, eps_C=0.01',logx=True)
    b.set_xlabel('Shots $S$');b.set_ylabel('estimate');b.legend(fontsize=7)
    b.text(.02,.03,'solid: Tr(P rhobar^N)/Tr(rhobar^N); dotted: shared-frame value;\ncircled: S t^2<50 (delta method not yet valid)',
           transform=b.transAxes,fontsize=7)
    for M,col in zip(sorted(tw.M.unique()),('#2E86AB','#F18F01','#7B3294')):
        for mode,ls in (('independent_exact','-'),('shared_exact','--'),('untwirled',':')):
            g=tw[(tw['mode']==mode)&(tw.M==M)&np.isclose(tw.theta,np.pi/2)&(tw.N==4)&(tw.eps_clifford==0)].sort_values('p')
            c.plot(g.p,g.fidelity,color=col,ls=ls,marker='o',ms=4,label=f'M={M}, {mode.replace("_"," ")}')
    c.axvline(.5,color='k',lw=.6);c.axvline(.75,color='k',lw=.6)
    _panel(c,'c','Exact virtual fidelity, theta=pi/2, N=4');c.set_xlabel('dephasing $p$');c.set_ylabel('$F_N$');c.legend(fontsize=6,ncol=3)
    for (th,p),col in zip([(np.pi/2,.45),(np.pi/3,.45),(np.pi/2,.7),(np.pi/3,.7)],('#2E86AB','#F18F01','#7B3294','#C73E1D')):
        g=bud[(bud.M==2)&np.isclose(bud.theta,th)&np.isclose(bud.p,p)]
        tws=g[g['mode']=='twirled_independent'];un=g[g['mode']=='untwirled']
        best_t=tws.groupby('eps_clifford')['rmse_lead_C10000'].min()
        e.plot(best_t.index,best_t.values,color=col,marker='o',label=f'twirled, theta={th:.2f}, p={p}')
        e.axhline(un['rmse_lead_C10000'].min(),color=col,ls=':',lw=1)
    _panel(e,'d','Best-N RMSE at C=$10^4$ copies, M=2 (dotted: untwirled)',logy=True)
    e.set_xlabel('Clifford gate error $\\epsilon_C$');e.set_ylabel('leading RMSE');e.legend(fontsize=7)
    fig.suptitle('Randomized Clifford frames: K-frame sets, per-copy frame sampling, frame correlations, noisy Cliffords')
    fig.tight_layout();_save(fig,dest,'PQES_FigD_randomized_twirling')


# ------------------------------------------------------------------ Fig E
def fig_cycles(data,dest):
    d=_load(data,'cycles')
    if d is None:return
    p0=.15 if np.isclose(d.p,.15).any() else float(d.p.iloc[0])
    fig,axes=plt.subplots(2,2,figsize=(12,9));a,b,c,e=axes.ravel()
    for (chan,M),ls in zip([('local',2),('global',2)],('-','--')):
        for N,col in zip(sorted(d.N.unique()),('#2E86AB','#F18F01','#7B3294')):
            s=d[(d.channel==chan)&(d.M==M)&(d.N==N)&np.isclose(d.p,p0)].sort_values('cycle')
            if s.empty:continue
            lab=f'{chan}, D={2**M}, N={N}'
            a.plot(s.cycle,s.ideal_virtual_fidelity,ls=ls,color=col,marker='o',ms=4,label=lab)
            b.plot(s.cycle,s.log10_signed_normalization,ls=ls,color=col,marker='o',ms=4,label=lab)
            c.plot(s.cycle,s.log10_copies_budget,ls=ls,color=col,marker='o',ms=4,label=lab)
            e.plot(s.cycle,s.log10_swaps_budget,ls=ls,color=col,marker='o',ms=4,label=lab)
    _panel(a,'a',f'Normalized virtual fidelity, p={p0}');a.set_ylabel('$F_t$')
    _panel(b,'b','Signed normalization');b.set_ylabel(r'$\log_{10}Z_t$')
    _panel(c,'c','Original copies for RMSE 0.03: $N^tS$');c.set_ylabel(r'$\log_{10}(N^t S)$')
    _panel(e,'d','SWAP tests for RMSE 0.03: $(N^t-1)S$');e.set_ylabel(r'$\log_{10}[(N^t-1)S]$')
    for ax in axes.ravel():ax.set_xlabel('nested cycles $t$')
    a.legend(fontsize=7)
    fig.suptitle('Repeated ideal virtual cycles: formal fidelity versus the nested signed-ensemble cost')
    fig.tight_layout();_save(fig,dest,'PQES_FigE_repeated_cycle_cost')


# ------------------------------------------------------------------ Fig F
def fig_channel_physical(data,dest):
    d=_load(data,'channel_physical')
    if d is None:return
    sub=d[(d.M==2)&(d.family=='entangling')&(d.latency_profile=='fast')]
    fig,axes=plt.subplots(2,2,figsize=(13,9.5));a,b,c,e=axes.ravel()
    apps=[('physical','-'),('apparatus_gates_10x_better','--'),('ideal_apparatus',':')]
    for prot in ('tree_parallel','tree_sequential','coherent_request','derangement','vcp2_layerwise'):
        for app,ls in apps:
            cur=sub[(sub.protocol==prot)&(sub.apparatus_case==app)&((sub.N==2)|(sub.protocol=='vcp2_layerwise'))].sort_values('eps2')
            if cur.empty:continue
            kw=dict(color=COLORS[prot],marker=MARKER[prot],ls=ls,label=f'{LABEL[prot]}, {app.replace("_"," ")}')
            a.plot(cur.eps2,cur.bias.abs(),**kw);b.plot(cur.eps2,cur.variance_coeff,**kw)
            ok=cur[_feasible(cur.feasible)];c.plot(ok.eps2,ok.total_cnot,**kw)
    a.axhline(float(d.rmse_target.iloc[0]),color='k',lw=.8)
    _panel(a,'a','|bias| (N=2; output-only state VD vs layerwise VCP2)',logx=True,logy=True);a.set_ylabel(r'$|\mu_A-\mu_0|$')
    _panel(b,'b','variance coefficient',logx=True,logy=True);b.set_ylabel('$v_A$')
    _panel(c,'c','two-qubit gates to reach RMSE 0.03 (feasible only)',logx=True,logy=True);c.set_ylabel('total two-qubit gates')
    for ax in (a,b,c):ax.set_xlabel('$\\epsilon_2$')
    a.legend(fontsize=6,ncol=1,loc='lower right')
    row=sub[np.isclose(sub.eps2,.005)&(sub.apparatus_case=='physical')&((sub.N==2)|(sub.protocol=='vcp2_layerwise'))]
    prots=[p for p in ('tree_sequential','coherent_request','derangement','vcp2_layerwise') if (row.protocol==p).any()]
    keys=[('copies_per_shot','copies'),('cnot_per_shot','2q gates'),('ccx_per_shot','Toffolis'),('duration_per_shot','duration'),('width','width')]
    x=np.arange(len(keys));w=.8/len(prots)
    for k,p in enumerate(prots):
        r=row[row.protocol==p].iloc[0]
        vals=[r[key] for key,_ in keys]
        e.bar(x+k*w,vals,w,color=COLORS[p],label=LABEL[p])
        for xi,v in zip(x+k*w,vals):e.text(xi,v*1.05,f'{v:g}',ha='center',fontsize=6,rotation=90)
    e.set_xticks(x+.4-w/2,[lab for _,lab in keys]);e.set_yscale('log')
    _panel(e,'d','Per-shot resources at eps2=0.005 (physical, N=2)');e.legend(fontsize=7)
    fig.suptitle('Physical order-two channel purification (Liu-style, per layer, gate support) vs state VD; M=2 entangling, L=13 layers, fast measurement')
    fig.tight_layout();_save(fig,dest,'PQES_FigF_vcp2_vs_state_vd')


# ------------------------------------------------------------------ Fig G
def fig_global(data,dest):
    d=_load(data,'global_depol');bnd=_load(data,'global_boundaries')
    if d is None or bnd is None:return
    fig,axes=plt.subplots(2,2,figsize=(12,9));a,b,c,e=axes.ravel()
    cols={2:'#2E86AB',4:'#F18F01',8:'#7B3294'};lss={1:(0,(1,1)),2:'-',4:'--',8:':'}
    for D in sorted(d.D.unique()):
        for N in (2,4,8):
            s=d[(d.D==D)&(d.N==N)].sort_values('p')
            a.plot(s.p,s.repeated_stable_F,color=cols.get(D,'k'),ls=lss[N],label=f'D={D}, N={N}')
            bb=bnd[(bnd.D==D)&(bnd.N==N)]
            if len(bb):a.axvline(bb.p_SN_numeric.iloc[0],color=cols.get(D,'k'),ls=lss[N],lw=.6)
        for N in (1,2,4,8):
            s=d[(d.D==D)&(d.N==N)].sort_values('p')
            b.plot(s.p,s.F_N,color=cols.get(D,'k'),ls=lss[N],label=f'D={D}, N={N}')
            c.plot(s.p,s.T_N,color=cols.get(D,'k'),ls=lss[N],label=f'D={D}, N={N}')
    _panel(a,'a','Repeated cycles: stable fixed point (vertical: $p_{SN}$)');a.set_ylabel('$F_\\infty$')
    _panel(b,'b','One-shot power fidelity $F_N$');b.set_ylabel('$F_N$')
    _panel(c,'c','Signed normalization $T_N$',logy=True);c.set_ylabel('$T_N$')
    s=d[d.D==4]
    for N in (1,2,4,8):
        g=s[s.N==N].sort_values('p');e.plot(g.p,g['mse_lead_C10000'],ls=lss[N],color='#333333',label=f'N={N}')
    best=s.loc[s.groupby('p')['mse_lead_C10000'].idxmin()]
    e.scatter(best.p,best['mse_lead_C10000'],c=[cols.get(int(n),'k') if n in cols else '#999999' for n in best.N],s=12,zorder=3,label='optimal N (coloured)')
    _panel(e,'d','Leading MSE at C=$10^4$ copies, D=4',logy=True);e.set_ylabel('MSE')
    for ax in axes.ravel():ax.set_xlabel('global depolarizing $p$')
    a.legend(fontsize=7,ncol=2);b.legend(fontsize=6,ncol=2);e.legend(fontsize=7)
    txt='; '.join(f'D={r.D} N={r.N}: p_c={r.p_c:.3f}, p_SN={r.p_SN_numeric:.3f}' for r in bnd.itertuples() if r.N in (2,8))
    fig.suptitle('Global depolarizing: exact one-shot and repeated results\n'+txt,fontsize=10)
    fig.tight_layout();_save(fig,dest,'PQES_FigG_global_depolarizing')


# ------------------------------------------------------------------ Fig H
def fig_local(data,dest):
    d=_load(data,'local_resource')
    if d is None:return
    fig,axes=plt.subplots(2,2,figsize=(12,9));a,b,c,e=axes.ravel()
    prod=d[d.target=='plus']
    cols={1:'#2E86AB',2:'#F18F01',5:'#7B3294'};lss={.05:(0,(1,1)),.15:'-',.3:'--',.5:'-.',.7:':'}
    for M in sorted(prod.M.unique()):
        for p in (.15,.5,.7):
            s=prod[(prod.M==M)&np.isclose(prod.p,p)].sort_values('N')
            if s.empty:continue
            a.plot(s.N,s.F_N,color=cols.get(M,'k'),ls=lss.get(p,'-'),label=f'M={M}, p={p}')
            b.plot(s.N,s.T_N,color=cols.get(M,'k'),ls=lss.get(p,'-'),label=f'M={M}, p={p}')
    _panel(a,'a','Virtual fidelity, product $|+\\rangle^{\\otimes M}$');a.set_ylabel('$F_N$')
    _panel(b,'b','Signed normalization',logy=True);b.set_ylabel('$T_N$')
    for M in sorted(prod.M.unique()):
        s=prod[(prod.M==M)&np.isclose(prod.p,.3)].sort_values('N')
        if s.empty:continue
        c.plot(s.N,s['mse_lead_C10000'],color=cols.get(M,'k'),label=f'M={M} leading')
        c.errorbar(s.N,s['mse_sampled_C10000'],yerr=1.96*s['mse_sampled_se_C10000'],color=cols.get(M,'k'),fmt='o',ms=3,capsize=2,label=f'M={M} sampled')
        i=s['mse_lead_C10000'].idxmin();c.scatter([s.N[i]],[s['mse_lead_C10000'][i]],s=120,facecolors='none',edgecolors=cols.get(M,'k'))
    _panel(c,'c','MSE at C=$10^4$ copies, p=0.3 (circle: optimal N)',logy=True);c.set_ylabel('MSE')
    ent=d[d.target!='plus']
    for p in sorted(ent.p.unique()):
        s=ent[np.isclose(ent.p,p)].sort_values('N')
        line,=e.plot(s.N,s.F_N,ls=lss.get(round(p,2),'-'),label=f'p={p}')
        e.axhline(s.mismatch_floor.iloc[0],color=line.get_color(),lw=.7,ls=':')
    _panel(e,'d','Entangled target cos(pi/8)|00>+sin(pi/8)|11> (dotted: mismatch floor)');e.set_ylabel('$F_N$')
    for ax in axes.ravel():ax.set_xlabel('power $N$')
    a.legend(fontsize=7,ncol=2);c.legend(fontsize=7,ncol=2);e.legend(fontsize=7)
    fig.suptitle('Local depolarization: exact fidelity, normalization and fixed-copy-budget error (sampled vs leading)')
    fig.tight_layout();_save(fig,dest,'PQES_FigH_local_resource_tradeoff')


def fig_channel(data,dest):
    d=_load(data,'channel_proxy')
    if d is None:return
    fig,ax=plt.subplots(figsize=(7,4.5))
    for (M,fam,e2),dd in d.groupby(['M','family','eps2']):
        if M!=2 or fam!='entangling':continue
        dd=dd.sort_values('N');ax.plot(dd.N,dd.ideal_layerwise_channel_fidelity,marker='o',label=f'eps2={e2:g}')
    _panel(ax,'a','Ideal layerwise channel-purification proxy (NOT a physical circuit), M=2')
    ax.set_xlabel('channel-purification power $N$');ax.set_ylabel('ideal output fidelity');ax.legend()
    fig.tight_layout();_save(fig,dest,'PQES_App_IdealVCP_channel_proxy_ONLY')


def main():
    parser=argparse.ArgumentParser(description='CSV-only plotting for PQES numerics')
    parser.add_argument('--data-dir',type=Path,default=DATA_DEFAULT)
    parser.add_argument('--figures-dir',type=Path,default=FIG_DEFAULT)
    funcs={'readout':fig_readout,'mechanisms':fig_mechanisms,'phase':fig_phase,'placement':fig_placement,
           'twirling':fig_twirling,'cycles':fig_cycles,'vcp':fig_channel_physical,'global':fig_global,
           'local':fig_local,'channel':fig_channel}
    parser.add_argument('--figure',choices=['all']+list(funcs),default='all')
    args=parser.parse_args()
    for name in (funcs if args.figure=='all' else [args.figure]):
        print('Plotting',name);funcs[name](args.data_dir,args.figures_dir)
    print('Complete. Figures are constructed solely from saved CSV data.')


if __name__=='__main__':main()
