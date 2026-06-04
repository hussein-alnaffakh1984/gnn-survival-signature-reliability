import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis('off')

# Muted professional palette
c1='#34699a'  # input - blue
c2='#b5482f'  # gnn - brick
c3='#c98a2b'  # adaptive - ochre
c4='#5b8a5a'  # output - green
c_ss='#6a4c93' # signature - purple

def box(x,y,w,h,text,color,fs=9.5,tc='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",
        ec='#2c3e50',fc=color,lw=1.2,alpha=0.95))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,
            color=tc,fontweight='bold')

def arr(x1,y1,x2,y2,color='#2c3e50',lw=1.6,style='-|>'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,
        mutation_scale=16,color=color,lw=lw))

# Top row: pipeline
box(0.3,4.7,2.3,1.1,'Network input\n$G(N,L)$, types,\nfailure models',c1,9)
box(3.0,4.7,2.3,1.1,'Flow-aware GNN\n+ HGNN layers',c2,9.5)
box(5.7,4.7,2.3,1.1,'Adaptive\nrefinement loop',c3,9.5)
box(8.4,4.7,2.3,1.1,'Structure function\n$\\hat{\\phi}(X)$\n(trained once)',c2,9)

arr(2.6,5.25,3.0,5.25); arr(5.3,5.25,5.7,5.25); arr(8.0,5.25,8.4,5.25)

# feedback loop
arr(6.85,4.7,6.85,4.2,color=c3,lw=1.3)
arr(4.15,4.2,4.15,4.7,color=c3,lw=1.3)
ax.plot([4.15,6.85],[4.2,4.2],color=c3,lw=1.3)
ax.text(5.5,3.95,'most-informative samples (convergence: $\\sigma \\leq \\sigma_0$)',
        ha='center',fontsize=8,color=c3,style='italic')

# Survival signature center
box(4.0,2.7,3.0,0.95,'Survival signature  $\\Phi(l_1,\\dots,l_K)$',c_ss,10.5)
arr(9.55,4.7,7.0,3.65,color='#888888',lw=1.4)

# Outputs row
box(0.3,0.9,2.3,1.1,'Reliability\n$R(t)$',c4,9.5)
box(3.0,0.9,2.3,1.1,'Importance\nmeasures\nBI / CI / FVI',c4,9.5)
box(5.7,0.9,2.3,1.1,'Sub-network\nanalysis\n(no retraining)',c4,9.5)
box(8.4,0.9,2.3,1.1,'Component\nranking &\ndecision support',c4,9.5)

for sx,tx in [(4.6,1.45),(5.2,4.1),(6.4,6.85),(6.9,9.5)]:
    arr(sx,2.7,tx,2.0,color='#888888',lw=1.2)

# novelty annotation
ax.text(5.5,2.35,'Novel: importance measures + sub-network analysis from GNN-estimated $\\Phi$',
        ha='center',fontsize=8.5,color='#b5482f',fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig1_framework.png',dpi=160,bbox_inches='tight',facecolor='white')
plt.close()
print("Clean framework diagram saved.")
