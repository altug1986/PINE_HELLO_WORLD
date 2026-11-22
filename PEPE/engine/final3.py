import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30);_,_,a5=adx5()
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));mA=mom(4);LA,SA=gA&(mA>=3),gA&(mA<=-3)
def v3(A30=25,n=48,m=2.5,rl=6,hold=24,tick=TICK,comm=COMM):
    g=(a30>A30)&(a5>lag(a5,rl));mo=mom(n);LB,SB=g&(mo>=m),g&(mo<=-m)
    r=run(C,LA|LB,SA|SB,hold,tick,comm,CASH);return r,LB,SB
print('n komşu:',[(n,round(metrics(r[0],*r[2:])['net']),round(metrics(r[0],*r[2:])['pf'],2)) for n in [42,48,54,60] for r in [v3(n=n)[0]]])
r2=run(C,LA,SA,24,TICK,COMM,CASH); r3,LB,SB=v3()
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:],a,b)
print(pd.DataFrame({'V2 FULL':M(r2),'V3 FULL':M(r3),'V2 OOS':M(r2,I_SPLIT),'V3 OOS':M(r3,I_SPLIT)}).T[K].round(2).to_string())
idx=B5.index;cuts=[pd.Timestamp(x,tz='UTC') for x in ['2023-05-01','2024-01-01','2025-01-01','2026-01-01','2026-11-20']]
Y={}
for a,b in zip(cuts[:-1],cuts[1:]):
    i0,i1=max(np.searchsorted(idx,a),1),np.searchsorted(idx,b)
    for nm,r in [('V2',r2),('V3',r3)]: m=M(r,i0,i1); Y[(str(a.year),nm)]=dict(net=m['net'],pf=m['pf'],n=m['n'],p98=m['p98'],posm=m['posm'])
i0=np.searchsorted(idx,pd.Timestamp('2026-05-19',tz='UTC'))
for nm,r in [('V2',r2),('V3',r3)]: m=M(r,i0); Y[('son6ay',nm)]=dict(net=m['net'],pf=m['pf'],n=m['n'],p98=m['p98'],posm=m['posm'])
print(pd.DataFrame(Y).T.round(2).to_string())
eq,te,tx,td,tp,tc=r3
src=np.where(np.where(td>0,LA[te],SA[te]),'A','B'); 
for s in 'AB': print('engine',s,'trades',(src==s).sum(),'net',tp[src==s].sum().round(0),'PF',(tp[(src==s)&(tp>0)].sum()/-tp[(src==s)&(tp<=0)].sum()).round(2))
mae=np.array([100*((C[e]-L_[e+1:x+1].min())/C[e] if d>0 else (H[e+1:x+1].max()-C[e])/C[e]) if x>e else 0 for e,x,d in zip(te,tx,td)])
hh=(tx-te)*5/60; print('MAE% P50/P95/P99/max',np.percentile(mae,[50,95,99]).round(2),mae.max().round(2),'| hold h med/P95',np.median(hh).round(1),np.percentile(hh,95).round(1))
rng=np.random.default_rng(1);q=rng.choice(tp,(5000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5/P50/P95',np.percentile(e[:,-1],[5,50,95]).round(0),'DD P50/P95/P99',np.percentile(dd,[50,95,99]).round(0),'p_loss',(e[:,-1]<=0).mean())
rs,_,_=v3(tick=3e-7,comm=0.0008);ms=metrics(rs[0],*rs[2:]);print('STRESS',{k:round(float(ms[k]),2) for k in ['net','pf','calmar','mdd','comm']})
f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
ax[0].plot(idx,r2[0],color='#999',lw=.9,label='V2 (209 işlem)');ax[0].plot(idx,eq,color='#1f6feb',lw=1.2,label=f'V3 2-MOTOR ({len(tp)} işlem)')
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].text(SPLIT,eq.max()*.92,' OOS →');ax[0].legend(loc='upper left');ax[0].grid(alpha=.3)
ax[0].set_title('PEPE 5M SWING V3 — fixed 500$, MTM equity (araştırma motoru)')
for r,cl in [(r2,'#999'),(r3,'#d1242f')]:
    d=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-d,color=cl,lw=.7)
ax[1].grid(alpha=.3);ax[1].set_ylabel('DD $');f.tight_layout();f.savefig('pepe_swing_v3.png',dpi=110)
