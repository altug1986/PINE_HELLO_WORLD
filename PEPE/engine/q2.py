import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    if len(r[1]): p[r[1][-1]:]=r[3][-1]
    return p
a30,_=adx_htf(30);_,_,a5=adx5();V=B5.volume.values.astype(float)
VR={L:V/pd.Series(V).rolling(L).mean().values for L in [96,288]}
def er(n): ch=np.abs(C-lag(C,n)); vol=pd.Series(np.abs(np.diff(C,prepend=C[0]))).rolling(n).sum().values; return ch/vol
ER={n:er(n) for n in [24,48]}
P=lambda L,S_:pos(run(C,L,S_,24,TICK,COMM,CASH))
K=['net','pf','n','calmar','p98','mdd','sharpe','sortino','r2','posm','cv']
rows=[]
def rec(nm,prm,t):
    r=run_t(C,t.astype(np.int64),TICK,COMM,CASH);f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    rows.append(dict(fam=nm,**prm,**{k:f[k] for k in K},nDD=f['net']/max(f['mdd'],1),Ical=i['calmar'],Ocal=o['calmar'],Onet=o['net'],flat=100*(t==0).mean()))
mA=mom(4);mB=mom(48)
for vk,vl,ek,en in itertools.product([0,1.5,2,3],[96,288],[0,.2,.3,.4],[24,48]):
    if (vk==0 and vl==288) or (ek==0 and en==48): continue
    fv=(VR[vl]>=vk) if vk else np.ones(N,bool); fe=(ER[en]>=ek) if ek else np.ones(N,bool)
    gA=(a30>28)&(a5>20)&(a5>lag(a5,12));gB=(a30>25)&(a5>lag(a5,6))
    for tgt in ['AB','A','B']:
        fa=fv if 'A' in tgt else True; fb=(fv&fe) if 'B' in tgt else True
        LA,SA=gA&(mA>=3)&fa,gA&(mA<=-3)&fa;LB,SB=gB&(mB>=2.5)&fb,gB&(mB<=-2.5)&fb
        rec('FILT',dict(vk=vk,vl=vl,ek=ek,en=en,on=tgt),np.sign(P(LA,SA)+P(LA|LB,SA|SB)))
        if vk==0 and ek==0: break
D=pd.DataFrame(rows);D.to_pickle('q2.pkl');pd.set_option('display.width',250)
print(D.iloc[:1].round(2).to_string(index=False))
print(D[D.Onet>0].sort_values('calmar',ascending=False).head(12).round(2).to_string(index=False))
print('calmar>V4',(D.calmar>D.calmar.iloc[0]).mean().round(2),'| calmar>1.5xV4',(D.calmar>1.5*D.calmar.iloc[0]).mean().round(3))
