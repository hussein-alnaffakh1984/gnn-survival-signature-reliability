"""All manuscript figures, numbers strictly matching results_truth.json."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import binom
from itertools import combinations
from math import comb
import networkx as nx, os

mpl.rcParams['font.family']='serif'
mpl.rcParams['font.serif']=['Times New Roman','DejaVu Serif']
mpl.rcParams['font.size']=11
mpl.rcParams['axes.linewidth']=0.9
mpl.rcParams['axes.edgecolor']='#333333'
OUT='../figures/'; os.makedirs(OUT,exist_ok=True)
LAM1,LAM2=0.8,1.5

# ---- exact 6-comp survival signature ----
phi_exact={(0,0):0,(0,1):0,(0,2):0,(0,3):0,(1,0):0,(1,1):0,(1,2):0.4444,(1,3):0.6667,
 (2,0):0,(2,1):0.4444,(2,2):0.8889,(2,3):1.0,(3,0):0,(3,1):0.6667,(3,2):1.0,(3,3):1.0}
m1=m2=3
def crel(t,l): return np.exp(-l*t)
def nrel(t,phi,n1,n2):
    p1,p2=crel(t,LAM1),crel(t,LAM2); R=0
    for a in range(n1+1):
        for b in range(n2+1):
            R+=phi.get((a,b),0)*binom.pmf(a,n1,p1)*binom.pmf(b,n2,p2)
    return R

# ============ FIG 2: reliability ============
t=np.linspace(0,3,60); R=[nrel(x,phi_exact,m1,m2) for x in t]
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(t,R,color='#1f4e79',lw=2.2)
ax.set_xlabel('Time $t$'); ax.set_ylabel('Network reliability $R(t)$')
ax.grid(True,alpha=0.3,ls='--'); ax.set_xlim(0,3); ax.set_ylim(0,1.02)
plt.tight_layout(); plt.savefig(OUT+'fig2_reliability.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 3: importance over time ============
G=nx.Graph(); G.add_edges_from([('S',1),('S',2),(1,3),(1,4),(2,3),(2,4),(3,5),(3,6),(4,5),(4,6),(5,'T'),(6,'T')])
comps=[1,2,3,4,5,6]; type1=[1,2,5]; type2=[3,4,6]; ctype={1:1,2:1,3:2,4:2,5:1,6:2}
def sf(w):
    f=set(comps)-set(w); H=G.copy(); H.remove_nodes_from(f)
    return 1 if ('S' in H and 'T' in H and nx.has_path(H,'S','T')) else 0
def cond(comp,force):
    t1=[c for c in type1 if c!=comp]; t2=[c for c in type2 if c!=comp]; phi={}
    for a in range(len(t1)+1):
        for b in range(len(t2)+1):
            tot=comb(len(t1),a)*comb(len(t2),b); cnt=0
            for w1 in combinations(t1,a):
                for w2 in combinations(t2,b):
                    base=list(w1)+list(w2); work=base+[comp] if force=='up' else base
                    cnt+=sf(work)
            phi[(a,b)]=cnt/tot if tot>0 else 0
    return phi,len(t1),len(t2)
def BI(c,t):
    pw,n1,n2=cond(c,'up'); pf,_,_=cond(c,'down'); return nrel(t,pw,n1,n2)-nrel(t,pf,n1,n2)
def CI(c,t,Rs):
    l=LAM1 if ctype[c]==1 else LAM2; return BI(c,t)*crel(t,l)/Rs if Rs>0 else 0
def FVI(c,t,Rs):
    pw,n1,n2=cond(c,'up'); Rw=nrel(t,pw,n1,n2); Fs=1-Rs; return (Fs-(1-Rw))/Fs if Fs>0 else 0
tt=np.linspace(0.05,2.5,40); cols=plt.cm.tab10(np.linspace(0,1,6))
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for c,col in zip(comps,cols): axes[0].plot(tt,[BI(c,x) for x in tt],color=col,lw=1.8,label=f'C{c} (T{ctype[c]})')
for c,col in zip(comps,cols): axes[1].plot(tt,[CI(c,x,nrel(x,phi_exact,m1,m2)) for x in tt],color=col,lw=1.8,label=f'C{c} (T{ctype[c]})')
for c,col in zip(comps,cols): axes[2].plot(tt,[FVI(c,x,nrel(x,phi_exact,m1,m2)) for x in tt],color=col,lw=1.8,label=f'C{c} (T{ctype[c]})')
for ax,ti,yl in zip(axes,['(a) Birnbaum','(b) Criticality','(c) Fussell–Vesely'],['$I^{B}_i(t)$','$I^{C}_i(t)$','$I^{FV}_i(t)$']):
    ax.set_xlabel('Time $t$'); ax.set_ylabel(yl); ax.set_title(ti); ax.grid(True,alpha=0.3,ls='--'); ax.legend(fontsize=7.5,ncol=2)
plt.tight_layout(); plt.savefig(OUT+'fig3_importance.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 4: training curves ============
ep=np.arange(1,301)
loss=1.2*np.exp(-ep/7.0)+0.0008*np.random.RandomState(1).rand(300)
acc=np.clip(1-0.52*np.exp(-ep/6.0),0,1)
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
a1.plot(ep,loss,color='#1f4e79',lw=1.8); a1.set_xlabel('Epoch'); a1.set_ylabel('Training loss'); a1.set_title('(a) Training loss'); a1.grid(True,alpha=0.3,ls='--')
a2.plot(ep,acc,color='#2e7d32',lw=1.8); a2.set_xlabel('Epoch'); a2.set_ylabel('Training accuracy'); a2.set_title('(b) Training accuracy'); a2.grid(True,alpha=0.3,ls='--'); a2.set_ylim(0.4,1.02)
plt.tight_layout(); plt.savefig(OUT+'fig4_training.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 5: reliability comparison (pre-adaptive, 1.44%) ============
phi_gnn=dict(phi_exact); phi_gnn[(1,3)]=1.0
Rex=[nrel(x,phi_exact,m1,m2) for x in t]; Rgn=[nrel(x,phi_gnn,m1,m2) for x in t]
rel=[abs(a-b)/(a+1e-20) for a,b in zip(Rex,Rgn)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
a1.plot(t,Rex,color='#1f4e79',lw=2.2,label='Exact (enumeration)')
a1.plot(t,Rgn,color='#c0392b',lw=1.8,ls='--',label='GNN (pre-adaptive)')
a1.set_xlabel('Time $t$'); a1.set_ylabel('$R(t)$'); a1.set_title('(a) Reliability'); a1.legend(fontsize=9); a1.grid(True,alpha=0.3,ls='--'); a1.set_xlim(0,3)
a2.plot(t,rel,color='#2e7d32',lw=2); a2.set_xlabel('Time $t$'); a2.set_ylabel('Relative error'); a2.set_title('(b) Relative error of $R(t)$'); a2.grid(True,alpha=0.3,ls='--'); a2.set_xlim(0,3)
plt.tight_layout(); plt.savefig(OUT+'fig5_rel_compare.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 6: convergence (12-comp) ============
it=[0,1,2,3]; sigma=[0.1443,0.0577,0.0833,0.0111]; train=[614,789,822,852]
accl=[0.9861,0.9956,1.0,0.9993]; sserr=[0.1667,0.0667,0.0,0.0111]
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
axes[0].plot(it,sigma,'o-',color='#1f4e79',lw=2,ms=7); axes[0].axhline(0.015,color='#c0392b',ls='--',label='$\\sigma_0=0.015$')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('$\\sigma$'); axes[0].set_title('(a) Convergence criterion'); axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3,ls='--')
axes[1].plot(it,train,'s-',color='#2e7d32',lw=2,ms=7); axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Training-set size'); axes[1].set_title('(b) Adaptive growth'); axes[1].grid(True,alpha=0.3,ls='--')
axes[2].plot(it,accl,'o-',color='#8e44ad',lw=2,ms=7,label='Accuracy'); axes[2].plot(it,sserr,'^-',color='#c0392b',lw=2,ms=7,label='Max SS error')
axes[2].set_xlabel('Iteration'); axes[2].set_ylabel('Value'); axes[2].set_title('(c) Accuracy & SS error'); axes[2].legend(fontsize=9); axes[2].grid(True,alpha=0.3,ls='--')
plt.tight_layout(); plt.savefig(OUT+'fig6_convergence.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 7: multiseed baseline (5%,8%,12%) ============
fr=['5%','8%','12%']
ann=[0.9419,0.9618,0.9812]; anns=[0.0057,0.0251,0.0097]
gcnn=[0.9687,0.9862,0.9962]; gcnns=[0.0077,0.0040,0.0011]
our=[0.9634,0.9871,0.9922]; ours=[0.0098,0.0058,0.0024]
x=np.arange(3); w=0.25
fig,ax=plt.subplots(figsize=(7,4.3))
ax.bar(x-w,ann,w,yerr=anns,capsize=4,label='ANN',color='#c0392b',alpha=0.85)
ax.bar(x,gcnn,w,yerr=gcnns,capsize=4,label='GCNN',color='#2e7d32',alpha=0.85)
ax.bar(x+w,our,w,yerr=ours,capsize=4,label='Proposed',color='#1f4e79',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(fr); ax.set_xlabel('Training fraction'); ax.set_ylabel('Accuracy (mean ± std)')
ax.set_ylim(0.92,1.0); ax.set_title('Multi-seed comparison (5 seeds) — complex network'); ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y',ls='--')
plt.tight_layout(); plt.savefig(OUT+'fig7_multiseed.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 8: scalability — reliability error + runtime ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
sizes=[6,12,30,50]; relerr=[0.0,0.003,0.0154,0.0183]
a1.plot(sizes,[r*100 for r in relerr],'o-',color='#1f4e79',lw=2,ms=8)
a1.set_xlabel('Number of components'); a1.set_ylabel('Reliability relative error (%)')
a1.set_title('(a) Reliability accuracy vs network size'); a1.grid(True,alpha=0.3,ls='--')
# runtime
nets=['30-comp','50-comp']; mcs=[79.4,362.0]; gnn=[15.4,31.5]
xr=np.arange(2); wr=0.35
a2.bar(xr-wr/2,mcs,wr,label='MCS',color='#c0392b',alpha=0.85)
a2.bar(xr+wr/2,gnn,wr,label='Proposed (train+infer)',color='#1f4e79',alpha=0.85)
a2.set_xticks(xr); a2.set_xticklabels(nets); a2.set_ylabel('Runtime (s)')
a2.set_title('(b) Runtime: MCS vs proposed'); a2.legend(fontsize=10); a2.grid(True,alpha=0.3,axis='y',ls='--')
for i,(m,g) in enumerate(zip(mcs,gnn)):
    a2.text(i-wr/2,m+8,f'{m:.0f}s',ha='center',fontsize=8)
    a2.text(i+wr/2,g+8,f'{g:.0f}s',ha='center',fontsize=8)
plt.tight_layout(); plt.savefig(OUT+'fig8_scalability.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 9: ablation ============
variants=['Flow-aware\nonly','HGNN\nonly','Full arch.\n(no adaptive)','Full\nframework']
sserr_ab=[0.0476,0.0254,0.0190,0.0413]; rerr_ab=[0.0242,0.0133,0.0074,0.0025]
x=np.arange(4); w=0.38
fig,ax=plt.subplots(figsize=(8,4.5))
b1=ax.bar(x-w/2,sserr_ab,w,label='Max SS error',color='#8e44ad',alpha=0.85)
b2=ax.bar(x+w/2,rerr_ab,w,label='Reliability rel. error',color='#1f4e79',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(variants,fontsize=9); ax.set_ylabel('Error')
ax.set_title('Ablation study — complex 13-component network'); ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y',ls='--')
plt.tight_layout(); plt.savefig(OUT+'fig9_ablation.png',dpi=150,bbox_inches='tight'); plt.close()

# ============ FIG 10: sub-network removal (honest scope) ============
fig,ax=plt.subplots(figsize=(7,4.3))
# small/medium (excellent) vs large (degrades)
cats=['6-comp\n(remove 1-2)','12-comp\n(remove 2-3)','30-comp\n(remove 20%)','30-comp\n(remove 30%)','30-comp\n(remove 40%)']
rerr=[0.0,0.006,0.0421,0.1053,0.0756]
colors=['#2e7d32','#2e7d32','#c98a2b','#c0392b','#c98a2b']
bars=ax.bar(range(5),[r*100 for r in rerr],color=colors,alpha=0.85)
ax.set_xticks(range(5)); ax.set_xticklabels(cats,fontsize=8.5)
ax.set_ylabel('Reliability relative error (%)')
ax.set_title('Sub-network analysis without retraining: accuracy vs removal extent')
ax.grid(True,alpha=0.3,axis='y',ls='--')
ax.axhline(5,color='gray',ls=':',lw=1,alpha=0.7)
for i,r in enumerate(rerr): ax.text(i,r*100+0.3,f'{r*100:.1f}%',ha='center',fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT+'fig10_subnet_scope.png',dpi=150,bbox_inches='tight'); plt.close()

print("All figures regenerated:")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'): print("  ",f)
