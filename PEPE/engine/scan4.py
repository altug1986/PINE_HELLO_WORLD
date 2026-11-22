import itertools, numpy as np, pandas as pd
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30);_,_,a5=adx5()
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));mA=mom(4);LA,SA=gA&(mA>=3),gA&(mA<=-3)
rows=[]
for A30,n,m,rl,hold in itertools.product([22,25,28],[36,48,60,72],[2.5,3,3.5],[3,6,9],[12,24,36,48]):
    g=(a30>A30)&(a5>lag(a5,rl));mo=mom(n)
    _,f,i,o=evaluate(LA|(g&(mo>=m)),SA|(g&(mo<=-m)),hold)
    rows.append(dict(A30=A30,n=n,m=m,rl=rl,hold=hold,net=f['net'],pf=f['pf'],tr=f['n'],cal=f['calmar'],p98=f['p98'],mdd=f['mdd'],Ipf=i['pf'],Opf=o['pf'],Onet=o['net'],mn=min(i['score'],o['score'])))
R=pd.DataFrame(rows);R.to_pickle('scan4.pkl');pd.set_option('display.width',200)
print('pos IS&OOS',((R.Onet>0)).mean().round(2),'PF>1.2',(R.pf>1.2).mean().round(2),'median PF',R.pf.median().round(2),'median tr',R.tr.median())
print(R.sort_values('mn',ascending=False).head(10).round(2).to_string(index=False))
for a in ['A30','n','m','rl','hold']: print(a,R.groupby(a)[['pf','net','tr','p98']].median().round(2).to_dict('index'))
