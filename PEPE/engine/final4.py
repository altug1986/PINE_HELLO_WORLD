import pickle, numpy as np, pandas as pd, numba, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eng import *
exec(open('arge2.py').read().split("a30,di30=adx_htf(30)")[0].split("S=pickle.load")[1].join(["S=pickle.load",""]) if False else "")
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
S=pickle.load(open('S.pkl','rb'))
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    p[r[1][-1]:]=r[3][-1]; return p
pA,pV=pos(S['A=V2']),pos(S['V3']); t=np.sign(pA+pV)
V4=run_t(C,t,TICK,COMM,CASH); V2,V3=S['A=V2'],S['V3']
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:],a,b)
print(pd.DataFrame({'V2':M(V2),'V3':M(V3),'V4':M(V4),'V2 OOS':M(V2,I_SPLIT),'V3 OOS':M(V3,I_SPLIT),'V4 OOS':M(V4,I_SPLIT)}).T[K].round(2).to_string())
idx=B5.index;cuts=[pd.Timestamp(x,tz='UTC') for x in ['2023-05-01','2024-01-01','2025-01-01','2026-01-01','2026-11-20']];Y={}
for a,b in zip(cuts[:-1],cuts[1:]):
    i0,i1=max(np.searchsorted(idx,a),1),np.searchsorted(idx,b)
    Y[str(a.year)]={nm:f"{M(r,i0,i1)['net']:.0f}/{M(r,i0,i1)['pf']:.2f}/{M(r,i0,i1)['n']}" for nm,r in [('V2',V2),('V3',V3),('V4',V4)]}
i0=np.searchsorted(idx,pd.Timestamp('2026-05-19',tz='UTC'));Y['son6ay']={nm:f"{M(r,i0)['net']:.0f}/{M(r,i0)['pf']:.2f}/{M(r,i0)['n']}" for nm,r in [('V2',V2),('V3',V3),('V4',V4)]}
print('net/PF/işlem\n',pd.DataFrame(Y).T.to_string())
# monthly & weekly evenness
for nm,r in [('V2',V2),('V3',V3),('V4',V4)]:
    d=pd.Series(r[0][DAY_I],index=DAY_T); w=d.resample('W').last().diff().dropna(); mo=d.resample('ME').last().diff().dropna()
    print(nm,'poz hafta',round((w>0).mean(),2),'poz ay',round((mo>0).mean(),2),'en kötü ay',round(mo.min()),'en kötü hafta',round(w.min()),'ay CV',round(mo.std()/mo.mean(),2))
eq,te,tx,td,tp,tc=V4
mae=np.array([100*((C[e]-L_[e+1:x+1].min())/C[e] if d>0 else (H[e+1:x+1].max()-C[e])/C[e]) if x>e else 0 for e,x,d in zip(te,tx,td)])
hh=(tx-te)*5/60;print('V4 MAE% P50/P95/P99/max',np.percentile(mae,[50,95,99]).round(2),mae.max().round(2),'| hold h med/P95',np.median(hh).round(1),np.percentile(hh,95).round(1),'| flat%',round(100*(t==0).mean(),1))
rng=np.random.default_rng(1);q=rng.choice(tp,(5000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5/P50/P95',np.percentile(e[:,-1],[5,50,95]).round(0),'DD P50/P95/P99',np.percentile(dd,[50,95,99]).round(0),'p_loss',(e[:,-1]<=0).mean())
rs=run_t(C,t,3e-7,0.0008,CASH);ms=M(rs);print('STRESS',{k:round(float(ms[k]),2) for k in ['net','pf','calmar','mdd','comm']})
f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
for r,cl,lb,lw in [(V2,'#bbb','V2',.8),(V3,'#f0a35e','V3',.8),(V4,'#1f6feb','V4 ÇARPIŞMA',1.4)]:
    ax[0].plot(idx,r[0],color=cl,lw=lw,label=f'{lb} ({len(r[4])} işlem)')
    d=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-d,color=cl,lw=lw*.7)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].text(SPLIT,eq.max()*.92,' OOS →');ax[0].legend(loc='upper left');ax[0].grid(alpha=.3)
ax[0].set_title('PEPE 5M — V4 ÇARPIŞMA (V2 ∧ V3 aynı yön → pozisyon, ayrışma → FLAT) | fixed 500$, MTM');ax[1].set_ylabel('DD $');ax[1].grid(alpha=.3)
f.tight_layout();f.savefig('pepe_swing_v4.png',dpi=110)
