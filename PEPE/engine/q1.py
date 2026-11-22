import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
S=pickle.load(open('S.pkl','rb')); V4=pickle.load(open('V4.pkl','rb'))
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    p[r[1][-1]:]=r[3][-1]; return p
t4=np.sign(pos(S['A=V2'])+pos(S['V3']))
eq,te,tx,td,tp,tc=V4
mfe=np.array([100*((C[e:x+1].max()/C[e]-1) if d>0 else (1-C[e:x+1].min()/C[e])) for e,x,d in zip(te,tx,td)])
fin=100*tp/500
print('V4 işlem:',len(tp),'| MFE(close) P25/50/75/90',np.percentile(mfe,[25,50,75,90]).round(1))
L=tp<=0;print('kaybeden',L.sum(),'— MFE>=2%:',(mfe[L]>=2).sum(),'MFE>=4%:',(mfe[L]>=4).sum(),'| kazanan geri verme medyan %',np.median((mfe-fin)[~L]).round(1))
a30,_=adx_htf(30);_,_,a5=adx5()
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));mA=mom(4);gB=(a30>25)&(a5>lag(a5,6));mB=mom(48)
EL=(gA&(mA>=3))|(gB&(mB>=2.5));ES=(gA&(mA<=-3))|(gB&(mB<=-2.5))
FL=EL&~np.concatenate([[False],EL[:-1]]);FS=ES&~np.concatenate([[False],ES[:-1]])
@numba.njit
def run_ov(c,t,fl,fs,act,gb,sl,re,tick,comm,cash):
    n=c.size;pos=0;lock=0;ep=0.;q=0.;mf=0.;real=0.;eq=np.empty(n);k=0;ei=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    for i in range(n):
        d=t[i]
        if lock!=0 and d!=lock: lock=0
        if lock!=0 and re==1 and ((lock>0 and fl[i]) or (lock<0 and fs[i])): lock=0
        eff=d if lock==0 else 0
        if pos!=0 and eff==pos:
            r=(c[i]/ep-1)*pos
            if r>mf: mf=r
            if (sl>0 and r<=-sl) or (act>=0 and gb>0 and mf>=act and mf-r>=gb): lock=pos;eff=0
        if eff!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
            if eff!=0:
                px=c[i]+tick if eff>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i;mf=0.
            pos=eff
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[k]=ei;tx[k]=n-1;td[k]=pos;tp[k]=(px-ep)*q*pos-fee;tc[k]=fee;k+=1
    return eq,te[:k],tx[:k],td[:k],tp[:k],tc[:k]
K=['net','pf','n','calmar','p98','mdd','sharpe','sortino','r2','posm','cv']
rows=[]
def rec(prm,r):
    f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    rows.append(dict(**prm,**{k:f[k] for k in K},nDD=f['net']/f['mdd'],nP98=f['net']/f['p98'],Ical=i['calmar'],Ocal=o['calmar'],Onet=o['net']))
rec(dict(act=-1,gb=0,sl=0,re=0),run_ov(C,t4,FL,FS,-1.,0.,0.,0,TICK,COMM,CASH))
for act,gb,sl,re in itertools.product([0.,.02,.03,.05,.08,.12,.2],[.01,.02,.03,.05,.08,.12],[0.,.04,.08,.15],[0,1]):
    rec(dict(act=act,gb=gb,sl=sl,re=re),run_ov(C,t4,FL,FS,act,gb,sl,re,TICK,COMM,CASH))
for sl,re in itertools.product([.03,.04,.06,.08,.1,.15],[0,1]): rec(dict(act=-1,gb=0,sl=sl,re=re),run_ov(C,t4,FL,FS,-1.,0.,sl,re,TICK,COMM,CASH))
D=pd.DataFrame(rows);D.to_pickle('q1.pkl');pd.set_option('display.width',250)
print(D.iloc[:1].round(2).to_string(index=False))
print(D[(D.Onet>0)].sort_values('calmar',ascending=False).head(12).round(2).to_string(index=False))
print('calmar>V4:',(D.calmar>D.calmar.iloc[0]).mean().round(2))
