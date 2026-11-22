import pickle, itertools, numpy as np, pandas as pd
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
S=pickle.load(open('S.pkl','rb'))
def pos(r):
    p=np.zeros(N,np.int8); te,tx,td=r[1],r[2],r[3]
    for e,x,d in zip(te,tx,td): p[e:x]=d
    if len(te): p[te[-1]:]=td[-1]
    return p
R=lambda L,S_,h=0:run(C,L,S_,h,TICK,COMM,CASH)
pA,pB,pV=pos(S['A=V2']),pos(S['B']),pos(S['V3'])
a30,di30=adx_htf(30)
X={}
for tf,A in [(240,15),(240,0),(60,20)]: a,di=adx_htf(tf);g=a>A;X[f'DI{tf}/{A}']=R(g&(di>0),g&(di<0))
for n in [576,1152]: hh,ll=donch(n);X[f'DON{n}']=R(C>hh,C<ll)
for tf,ap,f in [(240,14,3),(240,22,2.5),(60,22,4)]: d=st_htf(tf,f,ap);X[f'ST{tf}/{ap}/{f}']=R(d==-1,d==1)
for tf,fa,sl in [(60,12,100),(30,20,200),(240,5,21)]: e1,e2=ema_htf(tf,fa),ema_htf(tf,sl);X[f'EMA{tf}/{fa}/{sl}']=R(e1>e2,e1<e2)
_,_,a5=adx5()
for n,m in [(24,2),(96,4),(144,5)]:
    g=(a30>25)&(a5>lag(a5,6));mo=mom(n);X[f'MOM{n}/{m}']=R(g&(mo>=m),g&(mo<=-m),24)
K=['net','pf','n','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv','score']
rows={}
def rec(name,p):
    r=R(p>0,p<0); f=metrics(r[0],*r[2:]); o=metrics(r[0],*r[2:],I_SPLIT,N); i=metrics(r[0],*r[2:],0,I_SPLIT)
    rows[name]={**{k:f[k] for k in K},'flat%':100*(p==0).mean(),'I_sc':i['score'],'O_net':o['net'],'O_pf':o['pf'],'O_sc':o['score']}
    return r
# 2-engine netted (flat when disagree) — single bot, 500 when agree
r_net=R(((pA+pV)>0),((pA+pV)<0)); # long when both long, short when both short; flat otherwise needs explicit exit:
# exact netted engine: target = sign agreement, 0 otherwise
@__import__('numba').njit
def run_t(c,t,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;real=0.;eq=np.empty(n);k=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ei=0
    for i in range(n):
        if t[i]!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
            if t[i]!=0:
                px=c[i]+tick if t[i]>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=t[i]
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[k]=ei;tx[k]=n-1;td[k]=pos;tp[k]=(px-ep)*q*pos-fee;tc[k]=fee;k+=1
    return eq,te[:k],tx[:k],td[:k],tp[:k],tc[:k]
def rec_t(name,t):
    r=run_t(C,t.astype(np.int64),TICK,COMM,CASH); f=metrics(r[0],*r[2:]); o=metrics(r[0],*r[2:],I_SPLIT,N); i=metrics(r[0],*r[2:],0,I_SPLIT)
    rows[name]={**{k:f[k] for k in K},'flat%':100*(t==0).mean(),'I_sc':i['score'],'O_net':o['net'],'O_pf':o['pf'],'O_sc':o['score']}; return r
for k in ['A=V2','V3']: rec(k,pos(S[k]))
r_AV=rec_t('NET A+V3 (flat ayrılıkta)',np.sign(pA+pV))
rec_t('NET A+B (flat ayrılıkta)',np.sign(pA+pB))
for nm,r in X.items():
    px=pos(r); rec_t(f'MAJ A+B+{nm}',np.sign(pA.astype(int)+pB+px))
    rows[nm]={**{k:metrics(r[0],*r[2:])[k] for k in K},'O_net':metrics(r[0],*r[2:],I_SPLIT,N)['net']}
pd.set_option('display.width',260);D=pd.DataFrame(rows).T
print(D.sort_values('score',ascending=False).round(2).to_string())
pickle.dump((X,r_AV),open('X.pkl','wb'))
