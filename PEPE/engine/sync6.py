import pickle, numpy as np, pandas as pd, sys
from eng import *
V6=pickle.load(open('V6.pkl','rb'))
for fn in sys.argv[1:]:
    x=pd.ExcelFile(fn);o=pd.read_excel(x,'Özellikler');p=pd.read_excel(x,'Performans')
    d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
    e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
    tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float)})
    eq,te,tx,td,tp,tc=V6
    en=pd.DataFrame({'et':B5.index[te].tz_localize(None)+pd.Timedelta(hours=3),'dir':td,'xt':B5.index[tx].tz_localize(None)+pd.Timedelta(hours=3),'pnl':tp})
    en=en[en.et>=tv.et.min()-pd.Timedelta(minutes=1)]
    m=tv.merge(en,on=['et','dir'],how='outer',indicator=True);b=m[m._merge=='both']
    print(fn,'|',o.iloc[0,1],'| TV net',p.iloc[2,1],'trades',len(tv),'|',m._merge.value_counts().to_dict(),'exit',(b.xt_x==b.xt_y).sum(),'/',len(b))
    print(m[m._merge!='both'].sort_values('et').head(3)[['et','dir','_merge']].to_string())
