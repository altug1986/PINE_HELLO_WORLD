import numpy as np, pandas as pd, pickle
from eng import *
b=pd.read_csv('/home/claude/data/BTC_1m.csv.gz');b.index=pd.to_datetime(b.time,unit='ms',utc=True)
for c in ['open','high','low','close']: b[c]=b[c].astype(float)
agg=dict(open='first',high='max',low='min',close='last')
b5=b.resample('5min').agg(agg).dropna();b60=b.resample('60min').agg(agg).dropna()
END=b.index[-1]
def tr(h,l,c): pc=np.r_[np.nan,c[:-1]];t=np.fmax(np.fmax(h-l,np.abs(h-pc)),np.abs(l-pc));t[0]=h[0]-l[0];return t
# 1h ATR% and 30-day percentile rank (last completed 1h bar)
h,l,c=b60.high.values,b60.low.values,b60.close.values
atrp=rma(tr(h,l,c),14)/c*100;rank=pd.Series(atrp).rolling(720).rank(pct=True).values
th=b60.index.as_unit('s').asi8
F={'bATRp':htf_map(atrp,th,3600),'bRank':htf_map(rank,th,3600)}
# PEPE 1h ATR% for relative vol
ph,pl,pc_,pth,_=htf(60);patr=rma(tr(ph,pl,pc_),14)/pc_*100;F['pATRp']=htf_map(patr,pth,3600)
# BTC 5m close & ATR aligned to PEPE 5m bars (same bar, no lookahead: same timestamp close)
x=b5.reindex(B5.index);ok=B5.index<=END
bc=x.close.values;ba=rma(tr(b5.high.values,b5.low.values,b5.close.values),14);ba=pd.Series(ba,index=b5.index).reindex(B5.index).values
F['bC']=np.where(ok,bc,np.nan);F['bATR5']=np.where(ok,ba,np.nan)
for k in ['bATRp','bRank','pATRp']: F[k]=np.where(B5.index<=END+pd.Timedelta(hours=1),F[k],np.nan)
pickle.dump(F,open('/home/claude/data/btcF.pkl','wb'))
print({k:(np.isfinite(v).mean().round(3),np.nanpercentile(v,[10,50,90]).round(3)) for k,v in F.items()})
