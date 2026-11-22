exec(open('cbrk.py').read().split("\nB=row(SH5)")[0])
x=pd.ExcelFile('tv116.xlsx');p=pd.read_excel(x,'Performans');print('TV net',p.iloc[2,1],'%',p.iloc[2,2]);print(p[p.iloc[:,0].astype(str).str.contains('Maks düşüş|Kâr faktörü|Toplam işlem|Kar faktörü')].iloc[:,:2].to_string(index=False))
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float),'sig':xx['Sinyal'] if 'Sinyal' in xx else ''})
print('TV işlem',len(tv),'net',round(tv.pnl.sum(),1),'PF',round(tv.pnl[tv.pnl>0].sum()/-tv.pnl[tv.pnl<=0].sum(),3),'| yıl',tv.groupby(tv.xt.dt.year).pnl.sum().round(0).to_dict())
T=B5.index
def todf(r): return pd.DataFrame({'et':T[r[1]].tz_localize(None)+pd.Timedelta(hours=3),'dir':r[3],'xt':T[r[2]].tz_localize(None)+pd.Timedelta(hours=3),'pnl':r[4]})
r6=cbrk(C,P5,2,100.,0,288,0,TICK,COMM,CASH);en=todf(r6);e5=todf(SH5)
print('motor V11.6 işlem',len(en),'net',round(en.pnl.sum(),1),'tetik',r6[6],'| V11.5 gölge',len(e5),round(e5.pnl.sum(),1))
m=tv.merge(en,on=['et','dir'],how='outer',indicator=True,suffixes=('_tv','_en'));print('V11.6 eşleşme',m._merge.value_counts().to_dict())
m5=tv.merge(e5,on=['et','dir'],how='outer',indicator=True);print('TV vs V11.5 gölge eşleşme',m5._merge.value_counts().to_dict())
mm=m.sort_values('et');f=mm[mm._merge!='both'].head(12);print(f[['et','dir','xt_tv','pnl_tv','xt_en','pnl_en','_merge']].to_string(index=False))
s=tv.groupby(tv.xt.dt.floor('D')).pnl.sum();eq=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq='D'),fill_value=0).cumsum();dd=eq.cummax()-eq
print('TV gün sonu MaxDD',round(dd.max(),1),'| motor 2024-05 Δ',round(en[(en.xt>='2024-05-01')&(en.xt<'2024-06-01')].pnl.sum(),1),'TV 2024-05',round(tv[(tv.xt>='2024-05-01')&(tv.xt<'2024-06-01')].pnl.sum(),1))
b=m[m._merge=='both'];print(f"\nFARK DÖKÜMÜ: eşleşen işlemlerde Δ {(b.pnl_tv-b.pnl_en).sum():+.1f} | sadece TV {m[m._merge=='left_only'].pnl_tv.sum():+.1f} ({(m._merge=='left_only').sum()} işlem) | sadece motor {m[m._merge=='right_only'].pnl_en.sum():+.1f} ({(m._merge=='right_only').sum()} işlem)")
b2=b.copy();b2['xd']=(b2.xt_tv!=b2.xt_en);print('eşleşen ama çıkış zamanı farklı:',int(b2.xd.sum()),'Δ',round((b2[b2.xd].pnl_tv-b2[b2.xd].pnl_en).sum(),1))
mm['yil']=mm.et.dt.year;print('yıla göre uyumsuz işlem sayısı:',mm[mm._merge!='both'].groupby('yil').size().to_dict())
for nm,t in [('TV',tv)]:
    pq=np.percentile(dd,[95,98,99]);ul=np.sqrt((dd**2).mean());yrs=len(eq)/365.25
    print(f"{nm}: DD P95/P98/P99 {pq.round(0)} | Ulcer {ul:.1f} UPI {eq.iloc[-1]/yrs/ul:.1f} | 2025-05-20 sonrası net {t[t.xt>='2025-05-20'].pnl.sum():.0f} | yıl % {(t.groupby(t.xt.dt.year).pnl.sum()/5).round(0).astype(int).tolist()}")
e5['tag']=SH5[6];b3=b2[b2.xd].merge(e5[['et','dir','tag','xt']].rename(columns={'xt':'xt_sh'}),on=['et','dir'],how='left')
b3['d']=b3.pnl_tv-b3.pnl_en;print(b3.reindex(b3.d.abs().sort_values(ascending=False).index).head(8)[['et','dir','xt_tv','xt_en','xt_sh','tag','pnl_tv','pnl_en','d']].round(2).to_string(index=False))
