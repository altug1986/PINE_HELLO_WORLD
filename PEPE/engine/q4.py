import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
exec(open('arge5.py').read().split("@numba.njit\ndef vstate")[1].split("F=np.zeros")[0].join(["@numba.njit\ndef vstate",""]))
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
V4=pickle.load(open('V4.pkl','rb'));a30,_=adx_htf(30);_,_,a5=adx5()
d=np.diff(C,prepend=np.nan);up=rma(np.where(d>0,d,0.),14);dn=rma(np.where(d<0,-d,0.),14);RSI=100-100/(1+up/dn)
F=np.zeros(N,bool);dv4=np.diff(V4[0][DAY_I])
K=['net','pf','n','calmar','p98','mdd','sharpe','r2','posm']
rows=[];EQ={}
def rec(fam,prm,p):
    r=run_t(C,p.astype(np.int64),TICK,COMM,CASH);f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    key=(fam,)+tuple(prm.values());EQ[key]=r[0][DAY_I].astype(np.float32)
    rows.append(dict(fam=fam,**prm,**{k:f[k] for k in K},Inet=i['net'],Onet=o['net'],Ipf=i['pf'],Opf=o['pf'],corr=np.corrcoef(np.diff(r[0][DAY_I]),dv4)[0,1],expo=100*(p!=0).mean()))
# E2 FADE: low-trend regime, fade bursts, close-based TP/SL/time
for T,n,m,tp,sl,mb in itertools.product([20,25,99],[3,4,6,12],[2,3,4,5],[.01,.02,.03],[.02,.03,.05],[12,24,48]):
    g=a30<T;mo=mom(n);rec('FADE',dict(T=T,n_=n,m=m,tp=tp,sl=sl,mb=mb),vstate(C,g&(mo<=-m),g&(mo>=m),tp,sl,mb,F,F,0))
print('fade done',len(rows))
# E3 RSI MR with SL
for lo,sl,mb,T in itertools.product([10,15,20,25],[.03,.05,.08],[48,144],[0,20,25]):
    g=(a30<T) if T else np.ones(N,bool)
    rec('RSI',dict(lo=lo,sl=sl,mb=mb,T=T),vstate(C,g&(RSI<lo),g&(RSI>100-lo),0.,sl,mb,RSI>50,RSI<50,0))
D=pd.DataFrame(rows);pickle.dump(EQ,open('EQ4.pkl','wb'));D.to_pickle('q4.pkl');pd.set_option('display.width',250)
for fam,g in D.groupby('fam'):
    s=g[(g.Inet>0)&(g.Onet>0)];print(fam,'IS+&OOS+',round(len(s)/len(g),2))
    print(s.sort_values('calmar',ascending=False).head(8).round(2).to_string(index=False))
