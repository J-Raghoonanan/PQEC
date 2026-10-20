#!/usr/bin/env python3
"""PQES publication plotting — LOAD CSVs, NEVER recompute quantum dynamics.

This is intentionally a separate file from run_simulations.py, replicating the
user's earlier `rho2_plots.py` workflow. The plotting conventions follow that
script: seaborn-v0_8-paper, Times New Roman preference, Matplotlib mathtext,
large APS-ready labels, heavy curves, conspicuous marker shapes, multi-panel
figures, vector PDF + 300dpi PNG, and generous panel dimensions.

The chosen data are NOT retroactively filtered to make one architecture win.
The exact error/noise setting is written on each figure. If a CSV has not yet
been generated, we SKIP the corresponding figure and print which task to run.

Use:
    python pqes_new_numerics/plot_results.py
    python pqes_new_numerics/plot_results.py --data-dir data/pqes_new_numerics
    python pqes_new_numerics/plot_results.py --figure readout

Figure map:
 vcp       -> PHYSICAL order-2 layerwise Pauli VCP vs two state-VD instruments;
 readout   -> noisy ancilla readout disadvantage and shot-level checks;
 phase     -> resource-correct architecture-selection heat maps;
 placement -> mid-circuit vs end-only, matched apparatus/midpoint rights;
 twirling  -> independent versus shared frame averaging, K-frame statistics;
 cycles    -> ideal virtual fidelity versus exponential nested resource cost;
 channel   -> IDEAL VCP Pauli-channel theoretical proxy; never hardware ranks.

Every panel is a visualization of a named CSV column or computed ratio of
CSV columns; quantum instrument code is NEVER imported here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib import font_manager

# The user's rho2_plots.py settings are reproduced as the baseline.
# Use the EXACT requested font when available (e.g. on the user's Mac),
# otherwise an available metrically compatible serif fallback.
_installed_fonts={f.name for f in font_manager.fontManager.ttflist}
_SERIF_FONT=next((f for f in ('Times New Roman','Liberation Serif','DejaVu Serif')
                  if f in _installed_fonts),'DejaVu Serif')
try:plt.style.use('seaborn-v0_8-paper')
except OSError:pass
plt.rcParams.update({
    'font.size':20,
    'axes.titlesize':40,
    'axes.labelsize':40,
    'xtick.labelsize':28,
    'ytick.labelsize':28,
    'legend.fontsize':20,
    'figure.titlesize':40,
    'font.family':_SERIF_FONT,
    'mathtext.fontset':'dejavusans',
    'axes.linewidth':1.2,
    'grid.linewidth':0.8,
    'lines.linewidth':3,
    'lines.markersize':12,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})

ARCH=('tree_parallel','tree_recycled','tree_sequential','coherent_request')
LABEL={'tree_parallel':'Parallel SWAP tree',
       'tree_recycled':'Recycled SWAP tree',
       'tree_sequential':'Measured accumulator',
       'coherent_request':'Coherent-control VD'}
COLORS={'tree_parallel':'#2E86AB',
        'tree_recycled':'#00845A',
        'tree_sequential':'#F18F01',
        'coherent_request':'#7B3294'}
MARKER={'tree_parallel':'o','tree_recycled':'s','tree_sequential':'^',
        'coherent_request':'D'}
ROOT=Path(__file__).resolve().parent.parent
DATA_DEFAULT=ROOT/'data'/'pqes_new_numerics'
FIG_DEFAULT=ROOT/'figures'/'pqes_new_numerics'


def _load(data,name):
    path=data/f'{name}.csv'
    if not path.is_file():
        print(f'  Missing {path}. To generate: --task {name}')
        return None
    return pd.read_csv(path)


def _save(fig,dest,stem):
    dest.mkdir(parents=True,exist_ok=True)
    for typ in ('pdf','png'):
        fname=dest/f'{stem}.{typ}'
        fig.savefig(fname,bbox_inches='tight',dpi=300)
        print('  Saved',fname)
    plt.close(fig)


def _panel(ax,letter,title,logx=False,logy=False):
    """Same uncluttered aesthetic as old figures; smaller local titles avoid overlap."""
    ax.set_title(title,fontsize=24,pad=12)
    ax.text(.025,.975,letter,transform=ax.transAxes,fontsize=25,va='top',
            ha='left',fontweight='bold',color='#444444')
    if logx:ax.set_xscale('log')
    if logy:ax.set_yscale('log')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True,alpha=.27,linewidth=.8)
    ax.tick_params(labelsize=17)


def _axis(ax,x=None,y=None):
    if x:ax.set_xlabel(x,fontsize=23)
    if y:ax.set_ylabel(y,fontsize=23)


def _legend(ax,**kw):ax.legend(frameon=False,fontsize=13,**kw)


def fig_readout(data,dest):
    """Key falsification/validation plot.

    a. Relative signed normalization t(q)/t(0), analytic (1-2q)**(#measurements).
    b. Ratio v_tree/v_coherent, N=4+; shows systematic readout penalty.
    c. Gate mechanism sweep, normalized ratio v/v_ideal (includes 2q entanglers).
    d. Empirical shot estimator standard deviation vs leading v/S prediction.
    """
    d=_load(data,'readout');g=_load(data,'gate_noise');s=_load(data,'shots')
    if d is None:return
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    a,b,c,e=axes.ravel()
    sub=d[(d.M==d.M.max())].copy()
    for N in sorted(sub.N.unique()):
        for arch in ('tree_sequential','coherent_request'):
            cur=sub[(sub.N==N)&(sub.architecture==arch)].sort_values('readout_q')
            if len(cur)==0:continue
            norm0=float(cur[cur.readout_q==0].iloc[0].normalization)
            a.plot(cur.readout_q,cur.normalization/norm0,
                   color=COLORS[arch],marker=MARKER[arch],linestyle='-' if N==4 else ':',
                   linewidth=2,label=f'{LABEL[arch]}, N={N}')
    _panel(a,'a','Signed visibility vs readout error');_axis(a,r'Readout probability, $q$',r'$t(q)/t(0)$')
    _legend(a,loc='best')
    for N in sorted(sub.N.unique()):
        if N==2:continue
        measured=sub[(sub.N==N)&(sub.architecture=='tree_sequential')].sort_values('readout_q')
        coherent=sub[(sub.N==N)&(sub.architecture=='coherent_request')].sort_values('readout_q')
        if len(measured)!=len(coherent):continue
        ratio=measured.variance_coeff.to_numpy()/coherent.variance_coeff.to_numpy()
        b.plot(measured.readout_q,ratio,'o-',label=rf'$N={N}$',markersize=7)
        b.plot(measured.readout_q,(1-2*measured.readout_q)**(-2*(N-2)),
               'k:',linewidth=1.5,label='Analytical prediction' if N==4 else None)
    _panel(b,'b','Tree-to-coherent shot-variance ratio');_axis(b,r'Readout probability, $q$',r'$v_{\rm tree}/v_{\rm coh}$')
    _legend(b)
    if g is not None:
        for arch in ARCH:
            cur=g[(g.N==g.N.max())&(g.architecture==arch)]
            baseline=cur[cur.noise_case=='ideal']
            if baseline.empty:continue
            v0=float(baseline.iloc[0].variance_coeff)
            cur=cur[~cur.noise_case.isin(['ideal'])]
            c.plot(range(len(cur)),cur.variance_coeff/v0,marker=MARKER[arch],
                   color=COLORS[arch],linewidth=1.7,markersize=7,label=LABEL[arch])
            labels=cur.noise_case.to_list()
        pretty={'fredkin_0p01':'Fredkin\n1%','ancilla_z_0p02':'Ancilla Z\n2%',
                    'readout_0p02':'Readout\n2%','memory_0p02':'Memory\n2%',
                    'coherent_idle_0p05':'Control idle\n5%',
                    'reset_0p025':'Reset\n2.5%','combined':'Combined'}
        c.set_xticks(range(len(labels)),[pretty.get(x,x) for x in labels],rotation=35,ha='right',fontsize=12)
        _legend(c)
    _panel(c,'c','Apparatus error mechanisms')
    _axis(c,y=r'Relative variance, $v/v_{\rm ideal}$')
    if s is not None:
        for arch in ('tree_sequential','coherent_request'):
            subshot=s[(s.N==s.N.max())&(s.architecture==arch)]
            pairs=subshot.groupby('shots').agg(empirical_std=('empirical','std'),
                                               variance=('variance_coeff','first')).reset_index()
            e.plot(pairs.shots,pairs.empirical_std,marker=MARKER[arch],color=COLORS[arch],
                   label=LABEL[arch]+' sampled')
            e.plot(pairs.shots,np.sqrt(pairs.variance/pairs.shots),
                   linestyle=':',color=COLORS[arch],label=LABEL[arch]+' CLT')
    _panel(e,'d','Measured shot fluctuations',logx=True,logy=True)
    _axis(e,r'Experimental shots, $S$',r'$\operatorname{sd}(\widehat\mu)$')
    _legend(e)
    fig.tight_layout(pad=3.2)
    _save(fig,dest,'PQES_FigA_readout_and_instrument_noise')


def fig_phase(data,dest):
    """Do not infer a 'winner' from virtual fidelity alone.

    We rank each architecture by input-copy/SWAP-gate cost using the same
    target RMSE and reject cases with b^2>=eps^2. Hardware width remains a
    separate axis: for M=2,N=4 a parallel tree and a sequential one cannot
    be compared simply by declaring qubits free. This is a conditional model.
    """
    d=_load(data,'phase')
    if d is None:return
    gate=d.gate_p.unique()[0];mem=d.memory_z_p.unique()[0]
    sub=d[(d.gate_p==gate)&(d.memory_z_p==mem)].copy()
    qs=np.sort(sub.readout_q.unique());idle=np.sort(sub.control_idle_p.unique())
    winners=np.full((len(idle),len(qs)),np.nan)
    ratio=np.full_like(winners,np.nan)
    costs={}
    for yi,ci in enumerate(idle):
        for xi,q in enumerate(qs):
            rr=sub[np.isclose(sub.control_idle_p,ci)&np.isclose(sub.readout_q,q)]
            rr=rr[rr.feasible.astype(str).str.lower()=='true']
            if rr.empty:continue
            # Lexicographic optimization: fewer SWAP gates, then narrower width,
            # then shorter swap critical path. This avoids fake winners on ties.
            rr=rr.sort_values(['total_fredkins','peak_qubits','critical_swap_stages'])
            best=rr.iloc[0]
            winners[yi,xi]=ARCH.index(best.architecture)
            info=rr.set_index('architecture')
            if {'tree_sequential','coherent_request'}.issubset(info.index):
                ratio[yi,xi]=info.loc['tree_sequential','total_fredkins']/info.loc['coherent_request','total_fredkins']
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    a,b,c,e=axes.ravel()
    cmap=ListedColormap([COLORS[x] for x in ARCH]);cmap.set_bad('#eeeeee')
    im=a.imshow(winners,origin='lower',aspect='auto',vmin=-.5,vmax=3.5,cmap=cmap)
    a.set_xticks(range(len(qs)),[f'{q:.3f}' for q in qs]);a.set_yticks(range(len(idle)),[f'{i:.3f}' for i in idle]);
    _panel(a,'a','Lowest-cost architecture');_axis(a,r'Readout error, $q$',r'Control idle error')
    handles=[plt.Line2D([0],[0],marker='s',linestyle='',color=COLORS[x],label=LABEL[x],markersize=9) for x in ARCH]
    # Place the architecture legend in a dedicated figure band above panels.
    z=b.imshow(ratio,origin='lower',aspect='auto',cmap='coolwarm',vmin=.5,vmax=2.)
    cb=fig.colorbar(z,ax=b,shrink=.8)
    cb.set_label('Cost ratio',fontsize=15)
    cb.ax.tick_params(labelsize=12)
    b.set_xticks(range(len(qs)),[f'{q:.3f}' for q in qs]);b.set_yticks(range(len(idle)),[f'{i:.3f}' for i in idle]);
    _panel(b,'b','Measured/coherent cost ratio');_axis(b,r'Readout error, $q$',r'Control idle error')
    central=q=qs[len(qs)//2]
    for arch in ARCH:
        dd=sub[(sub.architecture==arch)&np.isclose(sub.readout_q,q)].sort_values('control_idle_p')
        c.plot(dd.control_idle_p,dd.variance_coeff,marker=MARKER[arch],color=COLORS[arch],
               markersize=7,label=LABEL[arch])
    _panel(c,'c',f'Variance at $q={q:.3f}$');_axis(c,r'Coherent-control idle error',r'Variance coefficient, $v$')
    _legend(c)
    for arch in ARCH:
        dd=sub[(sub.architecture==arch)&np.isclose(sub.readout_q,q)].sort_values('control_idle_p')
        e.plot(dd.control_idle_p,dd.total_fredkins,marker=MARKER[arch],color=COLORS[arch],
               markersize=7,label=LABEL[arch])
    _panel(e,'d',f'Total Fredkins at RMSE={float(sub.rmse_target.iloc[0]):.2f}')
    _axis(e,r'Coherent-control idle error',r'Estimated total Fredkins')
    _legend(e)
    fig.suptitle(f'Matched $M=2$, $N=4$, width cap={int(sub.width_cap.iloc[0])}: Fredkin noise={gate:g}',fontsize=20,y=.995)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=4,fontsize=11,frameon=False)
    fig.tight_layout(rect=(.01,.02,.99,.88),pad=2.5)
    _save(fig,dest,'PQES_FigB_architecture_phase_diagram')


def fig_placement(data,dest):
    """Fair comparison allows coherent cyclic VD and PQES the SAME midpoint."""
    d=_load(data,'placement')
    if d is None:return
    M=int(d.M.max());N=int(d.N.max());p=float(sorted(d.algorithm_gate_p.unique())[len(d.algorithm_gate_p.unique())//2])
    sub=d[(d.M==M)&(d.N==N)&np.isclose(d.algorithm_gate_p,p)].copy()
    fig,axes=plt.subplots(2,2,figsize=(13,10));a,b,c,e=axes.ravel()
    marks={'output_only':'s','intermediate':'o'}
    lines={'output_only':':','intermediate':'-'}
    for arch in ('tree_sequential','coherent_request'):
        for place in ('output_only','intermediate'):
            dd=sub[(sub.architecture==arch)&(sub.placement==place)].sort_values('b_repeats')
            if dd.empty:continue
            kw=dict(color=COLORS[arch],marker=marks[place],linestyle=lines[place],markersize=8,
                    label=f'{LABEL[arch]} / {place}')
            a.plot(dd.b_repeats,dd.bias.abs(),**kw)
            c.plot(dd.b_repeats,dd.variance_coeff,**kw)
            e.plot(dd.b_repeats,dd.normalization.abs(),**kw)
    _panel(a,'a','Systematic estimator bias');_axis(a,r'Downstream block repetitions',r'$|\mu-\mu_{\rm ideal}|$');_legend(a)
    for place in ('output_only','intermediate'):
        dd=sub[(sub.architecture=='coherent_request')&(sub.placement==place)].sort_values('b_repeats')
        b.plot(dd.b_repeats,dd.algorithm_gates,marker=marks[place],linestyle=lines[place],
               label=place,markersize=8)
    _panel(b,'b','Algorithmic gate calls');_axis(b,r'Downstream block repetitions',r'Algorithmic gates / shot');_legend(b)
    _panel(c,'c','Sampling variance');_axis(c,r'Downstream block repetitions',r'$v$ per shot')
    _panel(e,'d','Signed normalization');_axis(e,r'Downstream block repetitions',r'$|\mathrm{Tr}(W)|$')
    fig.suptitle(f'Matched placement, $M={M}$, $N={N}$, algorithm noise={p:.3f}',fontsize=22,y=1.01)
    fig.tight_layout(pad=3.)
    _save(fig,dest,'PQES_FigC_midcircuit_placement')


def fig_twirling(data,dest):
    """Difference between nonlinear averaging orders, not a 3^M counting plot."""
    df=_load(data,'twirling')
    if df is None:return
    p=float(sorted(df.p.unique())[len(df.p.unique())//2]); d=df[np.isclose(df.p,p)]
    fig,axes=plt.subplots(2,2,figsize=(13,10));a,b,c,e=axes.ravel()
    exact=d[d['mode'].isin(['independent_exact','shared_exact'])]
    for mode,style in [('independent_exact','o-'),('shared_exact','s--')]:
        r=exact[exact['mode']==mode].sort_values('N')
        a.plot(r.N,r.fidelity,style,markersize=8,label=mode.replace('_',' '))
        e.plot(r.N,r.normalization,style,markersize=8,label=mode.replace('_',' '))
    _panel(a,'a','Frame correlations alter power-state fidelity')
    _axis(a,r'Power, $N$',r'Normalized fidelity');_legend(a)
    fixed=d[d['mode']=='fixed_K_empirical'].copy()
    for N in sorted(fixed.N.unique()):
        g=fixed[fixed.N==N].groupby('K').abs_error_vs_independent.agg(['mean','std']).reset_index()
        b.errorbar(g.K,g['mean'],yerr=g['std']/np.sqrt(fixed.trial.nunique()),
                   marker='o',capsize=4,linewidth=2,label=f'$N={N}$')
    _panel(b,'b','Empirical frame-set error',logx=True)
    _axis(b,r'Empirical frame count, $K$',r'Mean $|F_K-F_{\rm tw}|$');_legend(b)
    N=int(max(fixed.N.unique()))
    Ks=sorted(fixed.K.unique())
    selected=[Ks[0],Ks[-1]]
    for K in selected:
        u=fixed[(fixed.N==N)&(fixed.K==K)]
        c.hist(u.fidelity,bins=15,density=True,alpha=.5,label=f'$K={K}$')
    ref=float(exact[(exact.N==N)&(exact['mode']=='independent_exact')].fidelity.iloc[0])
    c.axvline(ref,color='black',linestyle=':',label='Exact twirl')
    _panel(c,'c',f'Finite frame-set dispersion ($N={N}$)');_axis(c,r'Virtual target fidelity',r'Density');_legend(c)
    _panel(e,'d','Signed normalizations',logy=True)
    _axis(e,r'Power, $N$',r'$\mathrm{Tr}(W_N)$');_legend(e)
    fig.suptitle(rf'Z dephasing $p={p:.2f}$, $|+\rangle$ target; frame sampling',fontsize=22,y=1.02)
    fig.tight_layout(pad=3.)
    _save(fig,dest,'PQES_FigD_randomized_twirling')


def fig_cycles(data,dest):
    """Fidelity saturation vs rapidly vanishing signed-normalization signal."""
    d=_load(data,'cycles')
    if d is None:return
    M=d.M.max();N=d.N.max();p=d.p.unique()[0]
    s=d[(d.M==M)&(d.N==N)&np.isclose(d.p,p)].sort_values('cycle')
    if s.empty:return
    fig,axes=plt.subplots(2,2,figsize=(12,10));a,b,c,e=axes.ravel()
    a.plot(s.cycle,s.ideal_virtual_fidelity,'o-',color='#2E86AB')
    _panel(a,'a','Normalized virtual-state fidelity');_axis(a,r'Nested cycles, $t$',r'$F_t$')
    b.plot(s.cycle,s.log10_signed_normalization,'s-',color='#F18F01')
    _panel(b,'b','Vanishing signed signal');_axis(b,r'Nested cycles, $t$',r'$\log_{10}Z_t$')
    c.plot(s.cycle,s.log10_copies,'^-',color='#00845A')
    _panel(c,'c','Input copies per signed trajectory');_axis(c,r'Nested cycles, $t$',r'$\log_{10}(N^t)$')
    e.plot(s.cycle,s.log10_conservative_sampling_factor,'D-',color='#7B3294')
    _panel(e,'d','Inverse-signal sampling factor');_axis(e,r'Nested cycles, $t$',r'$\log_{10}(Z_t^{-2})$')
    fig.suptitle(f'Ideal repeated normalized map ONLY: $M={M}$, $N={N}$, $p={p:.2f}$',fontsize=20,y=1.01)
    fig.tight_layout(pad=3.)
    _save(fig,dest,'PQES_FigE_repeated_cycle_operational_overhead')


def fig_channel(data,dest):
    """Ideal VCP theoretical proxy shown SEPARATELY, never in cost ranking."""
    d=_load(data,'channel_proxy')
    if d is None:return
    d=d[(d.M==d.M.max())]
    fig,ax=plt.subplots(figsize=(9.2,5.8))
    for (p,reps),dd in d.groupby(['algorithm_gate_p','b_repeats']):
        dd=dd.sort_values('N')
        ax.plot(dd.N,dd.ideal_layerwise_pauli_channel_fidelity,marker='o',
                label=f'$p={p:g}$, downstream repeats={reps}')
    _panel(ax,'a','Ideal layerwise Pauli-channel proxy (not a physical circuit)')
    _axis(ax,r'Channel-purification power, $N$',r'Ideal output fidelity')
    _legend(ax)
    fig.tight_layout(pad=2.0)
    _save(fig,dest,'PQES_App_IdealVCP_channel_proxy_ONLY')



def fig_channel_physical(data,dest):
    """ACTUAL order-two VCP instrument, not merely theoretical power weighting.

    Physical VCP2 requires an aux I/d register and two controlled-SWAP passes
    per Pauli-noise channel location. State VD2 needs one controlled-SWAP on
    two completed noisy circuit outputs. They both use two channel instances
    per noise location but different algorithm preparations/ancilla lifetimes.
    The figure shows both observable performance and raw physical resources;
    it does not pretend they are identical state/channel mitigation tasks.
    """
    d=_load(data,'channel_physical')
    if d is None:return
    M=int(d.M.max());reps=int(sorted(d.b_repeats.unique())[len(d.b_repeats.unique())//2])
    sub=d[(d.M==M)&(d.b_repeats==reps)&(d.apparatus_case=='combined_physical_noise')]
    protos=['stateVD2_tree_sequential','stateVD2_coherent_request','layerwise_physical_VCP2']
    names={'stateVD2_tree_sequential':'State VD: measured accumulator',
           'stateVD2_coherent_request':'State VD: coherent control',
           'layerwise_physical_VCP2':'Layerwise VCP2 (physical)'}
    col={'stateVD2_tree_sequential':'#F18F01','stateVD2_coherent_request':'#7B3294',
         'layerwise_physical_VCP2':'#2E86AB'}
    mark=['o','s','D']
    fig,axes=plt.subplots(2,2,figsize=(13,10));a,b,c,e=axes.ravel()
    for k,prot in enumerate(protos):
        dd=sub[sub.protocol==prot].sort_values('p_channel')
        if dd.empty:continue
        a.plot(dd.p_channel,dd.bias.abs(),marker=mark[k],color=col[prot],label=names[prot])
        b.plot(dd.p_channel,dd.variance_coeff,marker=mark[k],color=col[prot],label=names[prot])
        c.plot(dd.p_channel,dd.normalization.abs(),marker=mark[k],color=col[prot],label=names[prot])
    _panel(a,'a','Physical circuit estimator bias');_axis(a,r'Pauli noise probability, $p$',r'$|\mu-\mu_{\rm ideal}|$')
    _legend(a)
    _panel(b,'b','Physical circuit sampling variance');_axis(b,r'Pauli noise probability, $p$',r'Variance coefficient, $v$')
    _panel(c,'c','Physical signed normalization');_axis(c,r'Pauli noise probability, $p$',r'$|\mathrm{Tr}W|$')
    q=float(sorted(sub.p_channel.unique())[len(sub.p_channel.unique())//2])
    bars=sub[np.isclose(sub.p_channel,q)].set_index('protocol')
    x=np.arange(len(protos))
    vals=[bars.loc[prot,'native_fredkins'] for prot in protos]
    e.bar(x,vals,color=[col[prot] for prot in protos],width=.62)
    e.set_xticks(x,['Measured\nstate VD','Coherent\nstate VD','Layerwise\nVCP2'],fontsize=15)
    _panel(e,'d',f'Native Fredkins per shot ($p={q:g}$)')
    _axis(e,y=r'Controlled Fredkins / trajectory')
    for xi,v in zip(x,vals):e.text(xi,v+max(vals)*.025,str(int(v)),fontsize=14,ha='center')
    fig.suptitle(f'Physical order-2 Pauli-channel VCP vs state VD: $M={M}$, downstream repetitions={reps}',
                 fontsize=20,y=1.005)
    fig.tight_layout(rect=(.01,.01,.99,.97),pad=3.)
    _save(fig,dest,'PQES_FigF_physical_layerwise_VCP2_vs_stateVD2')


def main():
    parser=argparse.ArgumentParser(description='CSV-only plotting for PQES numerics')
    parser.add_argument('--data-dir',type=Path,default=DATA_DEFAULT)
    parser.add_argument('--figures-dir',type=Path,default=FIG_DEFAULT)
    parser.add_argument('--figure',choices=['all','readout','phase','placement','twirling','cycles','channel','vcp'],default='all')
    args=parser.parse_args()
    funcs={'readout':fig_readout,'phase':fig_phase,'placement':fig_placement,
           'twirling':fig_twirling,'cycles':fig_cycles,'channel':fig_channel,
           'vcp':fig_channel_physical}
    for name in (funcs.keys() if args.figure=='all' else [args.figure]):
        print('Plotting',name)
        funcs[name](args.data_dir,args.figures_dir)
    print('Complete. Figures are constructed solely from saved CSV data.')


if __name__=='__main__':main()
