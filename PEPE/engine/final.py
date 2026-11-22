import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30); _,_,a5=adx5()
def sysv(A30=28,A5=20,n=4,m=3,rl=12,hold=24):
    g=(a30>A30)&(a5>A5)&(a5>lag(a5,rl)); mo=mom(n); return evaluate(g&(mo>=m),g&(mo<=-m),hold)
def base(): mo=mom(4); g=(a30>20)&(a5>25); return evaluate(g&(mo>=1),g&(mo<=-1),0)
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm']
print('-- m / n duyarlılık (FULL net, PF | OOS PF)')
for m in [2.5,2.75,3,3.25,3.5]:
    print('m',m,*[f"n{n}: {f['net']:.0f}/{f['pf']:.2f}|{o['pf']:.2f}" for n in [3,4,5] for _,f,i,o in [sysv(m=m,n=n)]])
rb,fb,ib,ob=base(); rw,fw,iw,ow=sysv()
T=pd.DataFrame({'BASE V1 FULL':fb,'NEW FULL':fw,'NEW IS':iw,'NEW OOS':ow}).T[K]; print(T.round(2).to_string())
# yearly + last 6m
idx=B5.index; eq,te,tx,td,tp,tc=rw
cuts=[pd.Timestamp(x,tz='UTC') for x in ['2023-05-01','2024-01-01','2025-01-01','2026-01-01','2026-11-20']]
rows={}
for a,b in zip(cuts[:-1],cuts[1:]):
    i0,i1=np.searchsorted(idx,a),np.searchsorted(idx,b); rows[str(a.year)]=metrics(eq,tx,td,tp,tc,max(i0,1),i1,5)
i0=np.searchsorted(idx,pd.Timestamp('2026-05-19',tz='UTC')); rows['son6ay']=metrics(eq,tx,td,tp,tc,i0,N,5)
print(pd.DataFrame(rows).T[['net','pf','n','p98','mdd','posm','sharpe']].round(2).to_string())
# MAE per trade (intrabar, %) & hold time
mae=[]; 
for e,x,d in zip(te,tx,td):
    ep=C[e]; mae.append(100*((ep-L_[e+1:x+1].min())/ep if d>0 else (H[e+1:x+1].max()-ep)/ep) if x>e else 0)
mae=np.array(mae); hold_h=(tx-te)*5/60
print('MAE% P50/P95/P99/max', np.percentile(mae,[50,95,99]).round(2), mae.max().round(2), '| hold h med/P95', np.median(hold_h).round(1), np.percentile(hold_h,95).round(1))
# MC bootstrap on trade pnl (skip 5%, +1 tick extra cost)
rng=np.random.default_rng(1); q=rng.choice(tp,(5000,len(tp))); q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1); dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5/P50/P95',np.percentile(e[:,-1],[5,50,95]).round(0),'DD P50/P95/P99',np.percentile(dd,[50,95,99]).round(0),'p_loss',(e[:,-1]<=0).mean())
# cost stress: comm 0.08, slip 3 tick
r2_=run(C,*[x for x in [None]] if False else (lambda g,mo:(g&(mo>=3),g&(mo<=-3)))((a30>28)&(a5>20)&(a5>lag(a5,12)),mom(4)),24,3e-7,0.0008,CASH)
print('STRESS comm.08 slip3', {k:round(float(v),2) for k,v in metrics(r2_[0],*r2_[2:]).items() if k in ['net','pf','calmar','mdd']})
# chart
f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
ax[0].plot(idx,rb[0],color='#999',lw=.8,label='BASE V1'); ax[0].plot(idx,eq,color='#1f6feb',lw=1.2,label='SWING V2')
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8); ax[0].text(SPLIT,eq.max()*.9,' OOS →'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title('PEPE 5M ADX-MOMENTUM SWING V2 — fixed 500$ notional, MTM equity (araştırma motoru)')
ax2=ax[0].twinx(); ax2.plot(idx,C,color='#f0a',lw=.4,alpha=.35); ax2.set_yscale('log'); ax2.set_yticks([])
dd=np.maximum.accumulate(np.maximum(eq,0))-eq; ax[1].fill_between(idx,-dd,color='#d1242f',alpha=.6)
dday=dd[DAY_I]
for qq,cc in zip([95,98,99],['#bf8700','#d1242f','#6e1b1b']): ax[1].axhline(-np.percentile(dday,qq),ls='--',lw=.8,color=cc,label=f'P{qq}')
ax[1].legend(fontsize=8,loc='lower left'); ax[1].grid(alpha=.3); f.tight_layout(); f.savefig('/home/claude/work/pepe_swing_v2.png',dpi=110)
