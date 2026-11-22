import pickle, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
S=pickle.load(open('S.pkl','rb'))
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    p[r[1][-1]:]=r[3][-1]; return p
V4=run_t(C,np.sign(pos(S['A=V2'])+pos(S['V3'])),TICK,COMM,CASH); pickle.dump(V4,open('V4.pkl','wb'))
x=pd.ExcelFile('tv_v4.xlsx'); p=pd.read_excel(x,'Performans'); print(p.iloc[[1,2,3,4,6,7]].to_string())
d=pd.read_excel(x,'İşlemler'); d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası'); xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float)})
eq,te,tx,td,tp,tc=V4
en=pd.DataFrame({'et':B5.index[te].tz_localize(None)+pd.Timedelta(hours=3),'dir':td,'xt':B5.index[tx].tz_localize(None)+pd.Timedelta(hours=3),'pnl':tp})
m=tv.merge(en,on=['et','dir'],how='outer',suffixes=('_tv','_en'),indicator=True); b=m[m._merge=='both']
print(m._merge.value_counts().to_dict(),'exit match',(b.xt_tv==b.xt_en).sum(),'/',len(b),'TV net',round(tv.pnl.sum(),1),'EN net',round(en.pnl.sum(),1),'TV trades',len(tv))
print(m[m._merge!='both'].sort_values('et').head(4).to_string())
