import pickle, numpy as np, pandas as pd, numba, time
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
LIB=pickle.load(open('/home/claude/data/LIB.pkl','rb'));W50=pickle.load(open('/home/claude/data/W50.pkl','rb'))
names=list(LIB);Pm=np.stack([LIB[k].astype(np.int64) for k in names])
WI=[(int(np.searchsorted(B5.index,a.tz_localize('UTC'))),int(np.searchsorted(B5.index,b.tz_localize('UTC'))),b<pd.Timestamp('2025-05-20')) for a,b in W50]
def target(w,form):
    w=np.array(w)
    if form==0: return np.sign((w[:,None]*Pm).sum(0)).astype(np.int64)
    order=[i for i in np.argsort(-w,kind='stable') if w[i]>0];t=np.zeros(N,np.int64)
    for i in order[::-1]: t=np.where(Pm[i]!=0,Pm[i],t)
    return t
def evalw(w,form,full=False):
    r=run_t(C,target(w,form),TICK,COMM,CASH);i=metrics(r[0],*r[2:],0,I_SPLIT)
    wk=[metrics(r[0],*r[2:],a,b)['net'] for a,b,isw in WI if isw]
    obj=(i['score'] if np.isfinite(i['score']) else -5)+0.5*np.clip(np.mean(wk)/100,-2,2)
    if not full: return obj
    f=metrics(r[0],*r[2:]);o=metrics(r[0],*r[2:],I_SPLIT,N);wo=[metrics(r[0],*r[2:],a,b)['net'] for a,b,isw in WI if not isw]
    return dict(obj=obj,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],p98=f['p98'],mdd=f['mdd'],q=f['net']/f['mdd'],sharpe=f['sharpe'],posm=f['posm'],Inet=i['net'],Onet=o['net'],Ical=i['calmar'],Ocal=o['calmar'],wIS=sum(wk),wOOS=sum(wo)),r
# baz V6.2 / V6.3 genomları
b62=[3 if k=='RSIX' else 2 if k=='COL' else 1 if k=='MID' else 0 for k in names]
b63=[4 if k=='RSIX' else 3 if k=='COL' else 2 if k=='B1D' else 1 if k=='MID' else 0 for k in names]
rng=np.random.default_rng(11);sc={};t0=time.time()
def E(g):
    k=tuple(g)
    if k not in sc: sc[k]=evalw(g[:-1],g[-1])
    return sc[k]
E(b62+[1]);E(b63+[1])
for i in range(80): E(list(rng.integers(0,4,len(names)))+[int(rng.integers(0,2))])
for gen in range(25):
    el=sorted(sc,key=sc.get,reverse=True)[:8]
    for _ in range(24):
        a,b=(list(el[j]) for j in rng.choice(len(el),2));c=[a[j] if rng.random()<.5 else b[j] for j in range(len(a))]
        for j in range(len(c)-1):
            if rng.random()<.15: c[j]=int(np.clip(c[j]+rng.choice([-1,1]),0,3))
        if rng.random()<.1: c[-1]=1-c[-1]
        E(c)
    if gen%5==4: bk=max(sc,key=sc.get);print(f'gen{gen} best obj {sc[bk]:.3f} evals {len(sc)} {time.time()-t0:.0f}s',flush=True)
pickle.dump(sc,open('/home/claude/data/ga2_sc.pkl','wb'))
rows=[]
for k in sorted(sc,key=sc.get,reverse=True)[:10]+[tuple(b62+[1]),tuple(b63+[1])]:
    m,_=evalw(list(k[:-1]),k[-1],True);rows.append(dict(genom=''.join(map(str,k[:-1])),form='ÖNC' if k[-1] else 'OY',**m))
pd.set_option('display.width',280);print('komponentler:',names);print(pd.DataFrame(rows).round(2).to_string(index=False))
