import itertools, numpy as np, pandas as pd
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30);_,_,a5=adx5()
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));mA=mom(4);LA,SA=gA&(mA>=3),gA&(mA<=-3)
def Bsig(A30=25,n=48,m=2.5,rl=6):
    g=(a30>A30)&((a5>lag(a5,rl)) if rl else True);mo=mom(n);return g&(mo>=m),g&(mo<=-m)
R=lambda L,S,h=24:run(C,L,S,h,TICK,COMM,CASH)
LB,SB=Bsig()
S={'A=V2':R(LA,SA),'B':R(LB,SB),'V3':R(LA|LB,SA|SB)}
def port(parts):   # parts: [(res, w)]  w = fraction of 500 notional
    eq=sum(w*r[0] for r,w in parts); cat=lambda j:np.concatenate([r[j]*(w if j in(4,5) else 1) for r,w in parts])
    o=np.argsort(cat(2)); return metrics(eq,cat(2)[o],cat(3)[o],cat(4)[o],cat(5)[o],0,N), metrics(eq,cat(2)[o],cat(3)[o],cat(4)[o],cat(5)[o],0,I_SPLIT), metrics(eq,cat(2)[o],cat(3)[o],cat(4)[o],cat(5)[o],I_SPLIT,N)
def wk(eq):  d=eq[DAY_I]; w=pd.Series(d,index=DAY_T).resample('W').last().diff().dropna(); return (w>0).mean()
# daily pnl correlation
dp=pd.DataFrame({k:np.diff(r[0][DAY_I]) for k,r in S.items()}); print('daily pnl corr\n',dp.corr().round(2))
K=['net','pf','n','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv','score']
rows={}
for k,r in S.items(): f,i,o=port([(r,1)]); rows[k]={**{x:f[x] for x in K},'posw':wk(r[0]),'O_net':o['net'],'O_r2':o['r2']}
for w in [0.3,0.4,0.5,0.6,0.7]:
    for a,b in [('A=V2','B'),('A=V2','V3')]:
        f,i,o=port([(S[a],w),(S[b],1-w)]); eq=w*S[a][0]+(1-w)*S[b][0]
        rows[f'{a[0]}{w:.1f}+{b}{1-w:.1f}']={**{x:f[x] for x in K},'posw':wk(eq),'O_net':o['net'],'O_r2':o['r2']}
pd.set_option('display.width',250);print(pd.DataFrame(rows).T.round(2).to_string())
import pickle; pickle.dump(S,open('S.pkl','wb'))
