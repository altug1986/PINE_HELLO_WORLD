import pickle, numpy as np, pandas as pd
from eng import *
x=pd.ExcelFile('tv12.xlsx');p=pd.read_excel(x,'Performans');o=pd.read_excel(x,'Özellikler')
print('TV V12 net',p.iloc[2,1],'%',p.iloc[2,2],'| aralık',o.iloc[0,1]);print(p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,:2].to_string(index=False))
print(o[o.iloc[:,0].astype(str).str.contains('OI|Ateş|BTC|z|Takip|Kapanış',regex=True)].to_string(index=False))
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float)})
print('TV işlem',len(tv),'PF',round(tv.pnl[tv.pnl>0].sum()/-tv.pnl[tv.pnl<=0].sum(),3),'| yıl %',(tv.groupby(tv.xt.dt.year).pnl.sum()/5).round(0).to_dict())
for nm in ['V12','V111']:
    r=pickle.load(open(f'{nm}.pkl','rb'))
    en=pd.DataFrame({'et':B5.index[r[1]].tz_localize(None)+pd.Timedelta(hours=3),'dir':r[3],'pnl':r[4],'xt':B5.index[r[2]].tz_localize(None)+pd.Timedelta(hours=3)})
    m=tv.merge(en,on=['et','dir'],how='outer',suffixes=('_tv','_en'),indicator=True)
    print(nm,'motor net',round(en.pnl.sum()),'işlem',len(en),'| eşleşme',m._merge.value_counts().to_dict())
    if nm=='V12':
        mm=m[m._merge!='both'].sort_values('et');print('ilk farklar:');print(mm.head(8)[['et','dir','_merge','pnl_tv','pnl_en']].to_string(index=False))
        # aylık fark
        a=tv.groupby(tv.xt.dt.to_period('Q')).pnl.sum();b=en.groupby(en.xt.dt.to_period('Q')).pnl.sum()
        print('çeyreklik TV - motor:');print((a-b).round(0).to_string())
