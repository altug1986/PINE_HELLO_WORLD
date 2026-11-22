import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
exec(open('arge5.py').read().split("@numba.njit\ndef vstate")[1].split("F=np.zeros")[0].join(["@numba.njit\ndef vstate",""]))
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    if len(r[1]): p[r[1][-1]:]=r[3][-1]
    return p
b=pd.read_csv('/home/claude/data/BTC_1m.csv.gz');b.columns=[c.strip().lower() for c in b.columns];print(b.columns.tolist()[:6],len(b))
tc=[c for c in b.columns if 'time' in c][0];b.index=pd.to_datetime(b[tc],unit='ms',utc=True)
B=b[['open','high','low','close']].resample('5min').agg(dict(open='first',high='max',low='min',close='last')).dropna()
BC=B.close.reindex(B5.index).ffill().values;print('BTC coverage',np.isfinite(BC).mean().round(4),B.index[-1])
def bmom(n): m=np.full(N,np.nan);m[n:]=(BC[n:]/BC[:-n]-1)*100;return m
S=pickle.load(open('S.pkl','rb'));V4=pickle.load(open('V4.pkl','rb'));pA,pV=pos(S['A=V2']),pos(S['V3']);t4=np.sign(pA+pV)
a30,_=adx_htf(30);_,_,a5=adx5();F=np.zeros(N,bool)
d=np.diff(C,prepend=np.nan);up=rma(np.where(d>0,d,0.),14);dn=rma(np.where(d<0,-d,0.),14);RSI=100-100/(1+up/dn)
K=['net','pf','n','calmar','p98','mdd','sharpe','sortino','r2','posm']
rows=[];dv4=np.diff(V4[0][DAY_I])
def rec(fam,prm,t=None,r=None):
    if r is None: r=run_t(C,t.astype(np.int64),TICK,COMM,CASH)
    f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    rows.append(dict(fam=fam,**prm,**{k:f[k] for k in K},nDD=f['net']/max(f['mdd'],1),Ical=i['calmar'],Ocal=o['calmar'],Onet=o['net'],corr=np.corrcoef(np.diff(r[0][DAY_I]),dv4)[0,1]))
    return r
rec('V4',{},t4)
# 1) BTC lead-lag: BTC burst -> PEPE same direction (always-in virtual, hold)
for n,m,h in itertools.product([1,2,3,4,6],[0.4,0.6,0.8,1.0,1.5],[12,24]):
    bm=bmom(n);rec('BTCLEAD',dict(n_=n,m=m,h=h),vstate(C,bm>=m,bm<=-m,0.,0.,0,F,F,h))
# 2) BTC agreement filter on V4 engines: engine signal valid only if BTC n-bar mom same sign
mA=mom(4);mB=mom(48);gA=(a30>28)&(a5>20)&(a5>lag(a5,12));gB=(a30>25)&(a5>lag(a5,6))
P=lambda L,S_:pos(run(C,L,S_,24,TICK,COMM,CASH))
for n,th,mode in itertools.product([4,12,48],[0,0.5,1.0],['same','idio']):
    bm=bmom(n)
    if mode=='same': fl,fs=bm>th,bm<-th
    else: fl,fs=bm<th,bm>-th           # idiosyncratic: PEPE moves, BTC does not confirm
    LA,SA=gA&(mA>=3)&fl,gA&(mA<=-3)&fs;LB,SB=gB&(mB>=2.5)&fl,gB&(mB<=-2.5)&fs
    rec('BTCFILT',dict(n_=n,th=th,mode=mode),np.sign(P(LA,SA)+P(LA|LB,SA|SB)))
D=pd.DataFrame(rows);pd.set_option('display.width',250);D.to_pickle('q5.pkl')
print(D.iloc[:1].round(2).to_string(index=False))
for fam in ['BTCLEAD','BTCFILT']:
    g=D[D.fam==fam];print(fam,'IS+&OOS+',round(((g.Ical>0)&(g.Ocal>0)).mean(),2));print(g.sort_values('calmar',ascending=False).head(6).dropna(axis=1,how='all').round(2).to_string(index=False))
pickle.dump(BC,open('/home/claude/data/BC.pkl','wb'))
