import pickle, numpy as np, pandas as pd
from eng import *
x=pd.ExcelFile('tv11.xlsx');p=pd.read_excel(x,'Performans');print('TV V11 net',p.iloc[2,1],'%',p.iloc[2,2],'| MaxDD',p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,1].tolist())
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()];xx=d[d.Tip.str.contains('çıkış')];e_=d[d.Tip.str.contains('giriş')]
pn=xx['Net K/Z USDT'].astype(float);print('TV işlem',len(xx),'PF',round(pn[pn>0].sum()/-pn[pn<=0].sum(),3))
V11=pickle.load(open('V11.pkl','rb'))
tv=pd.DataFrame({'et':pd.to_datetime(e_['Tarih ve saat']).values,'dir':np.where(e_.Tip.str.contains('Kısa'),-1,1)})
en=pd.DataFrame({'et':B5.index[V11[1]].tz_localize(None)+pd.Timedelta(hours=3),'dir':V11[3]})
m=tv.merge(en,on=['et','dir'],how='outer',indicator=True);print('SYNC',m._merge.value_counts().to_dict(),'| ilk fark',m[m._merge!='both'].sort_values('et').head(2)[['et','dir','_merge']].values.tolist())
def ulcer(eq):
    e=pd.Series(eq[DAY_I],index=DAY_T);dd=e.cummax()-e;U=np.sqrt((dd**2).mean());yrs=len(e)/365.25
    return U,e.iloc[-1]/yrs/U
U,upi=ulcer(V11[0]);print(f'MOTOR V11 Ulcer$ {U:.1f} | UPI (yıllık net/Ulcer) {upi:.1f}')
# akış verileri
o=pd.read_csv('/mnt/user-data/uploads/1000PEPEUSDT_OI_BINANCE_USDM_5m_2023-05-05_2026-11-19.csv',usecols=['timestamp_utc','open_interest_qty','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio'])
o.index=pd.to_datetime(o.timestamp_utc,utc=True);o=o.drop(columns='timestamp_utc').replace(0,np.nan);o=o[~o.index.duplicated()].reindex(B5.index,method='ffill').ffill()
F={'OI':o.open_interest_qty.values,'TAKER':o.sum_taker_long_short_vol_ratio.values,'TOPPOS':o.sum_toptrader_long_short_ratio.values,'TOPACC':o.count_toptrader_long_short_ratio.values,'RETAIL':o.count_long_short_ratio.values}
pickle.dump(F,open('/home/claude/data/flowF.pkl','wb'))
# tek değişkenli öngörü: özellik z-skoru (1 gün) → ileri getiri (1s, 4s) ; IS keşif / OOS teyit
fw={h:np.r_[(C[h:]/C[:-h]-1),np.full(h,np.nan)]*100 for h in [12,48]}
def z(x,w=288): s=pd.Series(np.log(x));return ((s-s.rolling(w).mean())/s.rolling(w).std()).values
feat={'TAKER z':z(F['TAKER']),'TAKER 12bar ort z':pd.Series(z(F['TAKER'])).rolling(12).mean().values,'OI Δ6 %':np.r_[np.full(6,np.nan),(F['OI'][6:]/F['OI'][:-6]-1)*100],
      'TOPPOS Δ12 %':np.r_[np.full(12,np.nan),(F['TOPPOS'][12:]/F['TOPPOS'][:-12]-1)*100],'TOPPOS z':z(F['TOPPOS']),'RETAIL z':z(F['RETAIL']),'TOPACC-RETAIL z':z(F['TOPACC']/F['RETAIL'])}
mom12=np.r_[np.full(12,np.nan),(C[12:]/C[:-12]-1)*100]
rows=[]
for k,v in feat.items():
    for h in [12,48]:
        for seg,sl in [('IS',slice(300,I_SPLIT)),('OOS',slice(I_SPLIT,N-60))]:
            a,b=v[sl],fw[h][sl];ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok]
            q=pd.qcut(a,10,labels=False,duplicates='drop');g=pd.Series(b).groupby(q).mean()
            rows.append(dict(özellik=k,ufuk=f'{h*5}dk',seg=seg,ic=np.corrcoef(a,b)[0,1],ust_dilim=g.iloc[-1],alt_dilim=g.iloc[0],fark=g.iloc[-1]-g.iloc[0]))
R=pd.DataFrame(rows);pd.set_option('display.width',200)
print(R.pivot_table(index=['özellik','ufuk'],columns='seg',values=['ic','fark']).round(3).to_string())
