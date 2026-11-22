import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec(open('arge2.py').read().split("for k in ['A=V2','V3']")[0].split("a30,di30=adx_htf(30)")[0])  # helpers: pos, R
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30);_,_,a5=adx5()
def sysA(h=24):
    g=(a30>28)&(a5>20)&(a5>lag(a5,12));m=mom(4);return g&(m>=3),g&(m<=-3)
def sysB(A30=25,n=48,m=2.5,rl=6):
    g=(a30>A30)&(a5>lag(a5,rl));mo=mom(n);return g&(mo>=m),g&(mo<=-m)
LA,SA=sysA()
rows=[]
for A30,n,m,rl,h in itertools.product([22,25,28],[36,42,48,54,60],[2,2.5,3],[3,6,9],[12,24,48]):
    LB,SB=sysB(A30,n,m,rl); pa=pos(run(C,LA,SA,h,TICK,COMM,CASH)); pv=pos(run(C,LA|LB,SA|SB,h,TICK,COMM,CASH))
    r=run_t(C,np.sign(pa.astype(np.int64)+pv),TICK,COMM,CASH); f=metrics(r[0],*r[2:]); o=metrics(r[0],*r[2:],I_SPLIT,N); i=metrics(r[0],*r[2:],0,I_SPLIT)
    rows.append(dict(A30=A30,n=n,m=m,rl=rl,h=h,net=f['net'],pf=f['pf'],tr=f['n'],cal=f['calmar'],p98=f['p98'],mdd=f['mdd'],r2=f['r2'],Inet=i['net'],Onet=o['net'],Opf=o['pf'],sc=f['score'],mn=min(i['score'],o['score'])))
D=pd.DataFrame(rows);D.to_pickle('agree_grid.pkl');pd.set_option('display.width',220)
print('n=',len(D),'IS+&OOS+',((D.Inet>0)&(D.Onet>0)).mean().round(2),'PF>1.4',(D.pf>1.4).mean().round(2),'median: net',D.net.median().round(0),'pf',D.pf.median().round(2),'cal',D.cal.median().round(2),'mdd',D.mdd.median().round(0))
for a in ['A30','n','m','rl','h']: print(a,D.groupby(a)[['net','pf','cal','mdd']].median().round(2).to_dict('index'))
print(D.sort_values('mn',ascending=False).head(8).round(2).to_string(index=False))
print(D[(D.A30==25)&(D.n==48)&(D.m==2.5)&(D.rl==6)].round(2).to_string(index=False))
