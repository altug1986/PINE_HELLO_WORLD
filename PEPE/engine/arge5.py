import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
S=pickle.load(open('S.pkl','rb'))
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    p[r[1][-1]:]=r[3][-1]; return p
t4=np.sign(pos(S['A=V2'])+pos(S['V3']))
a30,_=adx_htf(30);_,_,a5=adx5()
d=np.diff(C,prepend=np.nan);up=rma(np.where(d>0,d,0.),14);dn=rma(np.where(d<0,-d,0.),14);RSI=100-100/(1+up/dn)
@numba.njit
def vstate(c,L,S,tp,sl,mb,xl,xs,hold):
    n=c.size;p=np.zeros(n,np.int64);s=0;ep=0.;ei=0
    for i in range(n):
        if s!=0:
            r=(c[i]/ep-1)*s; bars=i-ei
            if (tp>0 and r>=tp) or (sl>0 and r<=-sl) or (mb>0 and bars>=mb) or (s>0 and xl[i]) or (s<0 and xs[i]): s=0
            elif s>0 and S[i] and bars>=hold: s=-1;ep=c[i];ei=i
            elif s<0 and L[i] and bars>=hold: s=1;ep=c[i];ei=i
        elif L[i]: s=1;ep=c[i];ei=i
        elif S[i]: s=-1;ep=c[i];ei=i
        p[i]=s
    return p
F=np.zeros(N,bool);K=['net','pf','n','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv','score']
rows=[]
def ev(name,prm,t):
    r=run_t(C,t,TICK,COMM,CASH);f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    rows.append(dict(fam=name,**prm,**{k:f[k] for k in K},flat=100*(t==0).mean(),Inet=i['net'],Onet=o['net'],Opf=o['pf'],mn=min(i['score'],o['score'])))
ev('V4',{},t4)
# C1 momentum filler (always-in virtual) — narrow moves
for n,m,rl,h in itertools.product([3,6,12,24],[0.75,1,1.5,2],[0,6],[0,12,24]):
    g=(a5>lag(a5,rl)) if rl else np.ones(N,bool);mo=mom(n)
    pc=vstate(C,g&(mo>=m),g&(mo<=-m),0.,0.,0,F,F,h); ev('MOMFILL',dict(N_=n,m=m,rl=rl,h=h),np.where(t4!=0,t4,pc))
# C2 RSI mean-reversion filler (exit RSI 50 / SL / max bars)
for lo,sl,mb in itertools.product([15,20,25,30],[0,0.02,0.04,0.08],[0,48,144]):
    pc=vstate(C,RSI<lo,RSI>100-lo,0.,sl,mb,RSI>50,RSI<50,0); ev('RSIFILL',dict(lo=lo,sl=sl,mb=mb),np.where(t4!=0,t4,pc))
D=pd.DataFrame(rows);D.to_pickle('fill.pkl');pd.set_option('display.width',250)
for fam,g in D.groupby('fam'):
    print(fam,'IS+&OOS+',((g.Inet>0)&(g.Onet>0)).mean().round(2),'| net>V4',(g.net>D.net.iloc[0]).mean().round(2))
    print(g.sort_values('mn',ascending=False).head(6).dropna(axis=1,how='all').round(2).to_string(index=False))
