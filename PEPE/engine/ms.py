import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
exec(open('arge5.py').read().split("@numba.njit\ndef vstate")[1].split("F=np.zeros")[0].join(["@numba.njit\ndef vstate",""]))
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
F=np.zeros(N,bool);a30,_=adx_htf(30);_,_,a5=adx5();A60,D60=adx_htf(60);A240,D240=adx_htf(240)
d=np.diff(C,prepend=np.nan)
def rsi(n): u=rma(np.where(d>0,d,0.),n);w=rma(np.where(d<0,-d,0.),n);return 100-100/(1+u/w)
R7=rsi(7);R3=rsi(3);V7=pickle.load(open('V7.pkl','rb'));dv7=np.diff(V7[0][DAY_I])
K=['net','pf','n','calmar','p98','mdd','sharpe','r2','posm'];rows=[];EQ={}
def rec(fam,prm,p):
    r=run_t(C,p.astype(np.int64),TICK,COMM,CASH);f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    if i['net']>0 and o['net']>0: EQ[(fam,)+tuple(prm.values())]=r
    rows.append(dict(fam=fam,**prm,**{k:f[k] for k in K},Ipf=i['pf'],Opf=o['pf'],Onet=o['net'],corr=np.corrcoef(np.diff(r[0][DAY_I]),dv7)[0,1]))
# MIKRO momentum (always-in virtual), gates: none / 5m ADX rising / 1h ADX>x
for n,m,h,gt in itertools.product([1,2,3,6],[0.8,1.0,1.5,2.0,2.5],[6,12,24],['none','rise','a60']):
    g=np.ones(N,bool) if gt=='none' else (a5>lag(a5,3)) if gt=='rise' else (A60>25);mo=mom(n)
    rec('MIK-MOM',dict(n_=n,m=m,h=h,g=gt),vstate(C,g&(mo>=m),g&(mo<=-m),0.,0.,0,F,F,h))
# MIKRO kırılım SAR
for bn,gt in itertools.product([12,24,48,96,144],['none','a60']):
    hh,ll=donch(bn);g=np.ones(N,bool) if gt=='none' else (A60>25);rec('MIK-BRK',dict(n_=bn,g=gt),vstate(C,g&(C>hh),g&(C<ll),0.,0.,0,F,F,0))
# MIKRO ortalamaya dönüş (TP/SL)
for rl,lo,tp,sl,mb in itertools.product([3,7],[5,10,15,20],[.01,.02,.03],[.02,.04],[24,72]):
    R_=R3 if rl==3 else R7;rec('MIK-MR',dict(n_=rl,m=lo,tp=tp,sl=sl,h=mb),vstate(C,R_<lo,R_>100-lo,tp,sl,mb,F,F,0))
# MAKRO: 4h/1h ADX-DI always-in, 1d-ölçek Donchian
for tf,A in [(240,0),(240,15),(240,20),(60,20),(60,25)]:
    a,di=(A240,D240) if tf==240 else (A60,D60);rec('MAK-DI',dict(n_=tf,m=A),vstate(C,(a>A)&(di>0),(a>A)&(di<0),0.,0.,0,F,F,0))
for bn in [576,864,1152,1728]:
    hh,ll=donch(bn);rec('MAK-DON',dict(n_=bn),vstate(C,C>hh,C<ll,0.,0.,0,F,F,0))
D=pd.DataFrame(rows);D.to_pickle('ms.pkl');pickle.dump(EQ,open('/home/claude/data/msEQ.pkl','wb'));pd.set_option('display.width',250)
for fam,g in D.groupby('fam'):
    s=g[(g.Onet>0)&(g.net>0)];print(fam,len(s),'/',len(g),'IS+OOS pozitif');print(s.sort_values('calmar',ascending=False).head(4).dropna(axis=1,how='all').round(2).to_string(index=False))
