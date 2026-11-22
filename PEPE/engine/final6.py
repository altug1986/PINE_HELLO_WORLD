import pickle, numpy as np, pandas as pd, numba, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    p[r[1][-1]:]=r[3][-1]; return p
S=pickle.load(open('S.pkl','rb'));V4=pickle.load(open('V4.pkl','rb'));PR=pickle.load(open('/home/claude/data/PR.pkl','rb'))
t4=np.sign(pos(S['A=V2'])+pos(S['V3']));pR=PR[(14,10,50,0.08,144)];t6=np.where(pR!=0,pR,t4)
V6=run_t(C,t6,TICK,COMM,CASH);pickle.dump(V6,open('V6.pkl','wb'))
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:],a,b)
T=pd.DataFrame({'V4':M(V4),'V6':M(V6),'V4 OOS':M(V4,I_SPLIT),'V6 OOS':M(V6,I_SPLIT)}).T[K];print(T.round(2).to_string())
yrs=N/(288*365.25)
for nm,r in [('V4',V4),('V6',V6)]: m=M(r);print(nm,'yıllık %/birim',round(100*m['net']/yrs/500,1),'MaxDD %/birim',round(100*m['mdd']/500,1),'net/MaxDD',round(m['net']/m['mdd'],2),'net/P98',round(m['net']/m['p98'],2))
idx=B5.index;cuts=[pd.Timestamp(x,tz='UTC') for x in ['2023-05-01','2024-01-01','2025-01-01','2026-01-01','2026-11-20']];Y={}
for a,b in zip(cuts[:-1],cuts[1:]):
    i0,i1=max(np.searchsorted(idx,a),1),np.searchsorted(idx,b);Y[str(a.year)]={nm:f"{M(r,i0,i1)['net']:.0f}/{M(r,i0,i1)['pf']:.2f}/{M(r,i0,i1)['calmar']:.2f}" for nm,r in [('V4',V4),('V6',V6)]}
print('net/PF/Calmar\n',pd.DataFrame(Y).T.to_string())
d=pd.Series(V6[0][DAY_I],index=DAY_T);w=d.resample('W').last().diff().dropna();mo=d.resample('ME').last().diff().dropna()
print('V6 poz hafta',round((w>0).mean(),2),'poz ay',round((mo>0).mean(),2),'en kötü ay',round(mo.min()),'en kötü hafta',round(w.min()),'RSI işlemi',(np.diff(np.concatenate([[0],(pR!=0).astype(int)]))==1).sum())
tp=V6[4];rng=np.random.default_rng(1);q=rng.choice(tp,(5000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1);print('MC net P5',np.percentile(e[:,-1],5).round(0),'DD P95/P99',np.percentile(dd,[95,99]).round(0))
rs=run_t(C,t6,3e-7,0.0008,CASH);print('STRESS',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
for r,cl,lb,lw in [(V4,'#bbb','V4',.9),(V6,'#1f6feb','V6 = V4 + RSI UÇ',1.3)]:
    ax[0].plot(idx,r[0],color=cl,lw=lw,label=lb);dd_=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-dd_,color=cl,lw=lw*.7)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');ax[0].grid(alpha=.3);ax[1].grid(alpha=.3);ax[1].set_ylabel('DD $')
ax[0].set_title('PEPE 5M V6 — 500$ sabit birim, basit getiri, MTM (araştırma motoru)');f.tight_layout();f.savefig('pepe_swing_v6.png',dpi=110)
