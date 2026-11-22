import numpy as np, pandas as pd, sys
from eng import *
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
d=pd.read_excel(sys.argv[1],'İşlemler'); d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası'); x=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'ep':e['Fiyat USDT'].astype(float),
                 'xt':pd.to_datetime(x['Tarih ve saat']),'pnl':x['Net K/Z USDT'].astype(float)})
a30,_=adx_htf(30);_,_,a5=adx5();g=(a30>28)&(a5>20)&(a5>lag(a5,12));mo=mom(4)
(eq,te,tx,td,tp,tc),*_=evaluate(g&(mo>=3),g&(mo<=-3),24)
for off in [0,3]:
    en=pd.DataFrame({'et':B5.index[te].tz_localize(None)+pd.Timedelta(hours=off),'dir':td,'ep':C[te],'xt':B5.index[tx].tz_localize(None)+pd.Timedelta(hours=off),'pnl':tp})
    m=tv.merge(en,on=['et','dir'],how='outer',suffixes=('_tv','_en'),indicator=True)
    print('offset',off,m._merge.value_counts().to_dict())
    if (m._merge=='both').mean()>.5:
        b=m[m._merge=='both']; print('exit match',(b.xt_tv==b.xt_en).sum(),'/',len(b),'sum|dpnl|',round((b.pnl_tv-b.pnl_en).abs().sum(),2),'TV net',round(tv.pnl.sum(),2),'EN net',round(en.pnl.sum(),2))
        print(m[m._merge!='both'].sort_values('et').head(6).to_string())
