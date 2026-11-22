exec(open('joint2.py').read().split("\nJ4=dict(")[0])
J4=dict(k='J4',n=6,x=3,y=3,zb=1.5);J2=dict(k='J2',n=12,x=4,y=4,zb=1)
x=pd.ExcelFile('/mnt/user-data/uploads/b0e4c899-CYBORG___PEPE_5M_SWING_V11.7_BINANCE_1000PEPEUSDT.P_2026-11-21_95259.xlsx');p=pd.read_excel(x,'Performans')
print('TV net',p.iloc[2,1],'%',p.iloc[2,2]);print(p[p.iloc[:,0].astype(str).str.contains('Maks düşüş|Kâr faktörü|Kar faktörü')].iloc[:,:2].to_string(index=False))
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float)}).dropna()
T=B5.index;todf=lambda r:pd.DataFrame({'et':T[r[1]].tz_localize(None)+pd.Timedelta(hours=3),'dir':r[3],'xt':T[r[2]].tz_localize(None)+pd.Timedelta(hours=3),'pnl':r[4]})
_,r7,_=runj(0,[J4,J2]);en=todf(r7);tv=tv[tv.et<=en.et.max()]
print('TV işlem',len(tv),'net',round(tv.pnl.sum(),1),'PF',round(tv.pnl[tv.pnl>0].sum()/-tv.pnl[tv.pnl<=0].sum(),3),'| motor',len(en),round(en.pnl.sum(),1))
m=tv.merge(en,on=['et','dir'],how='outer',indicator=True,suffixes=('_tv','_en'));print('eşleşme',m._merge.value_counts().to_dict())
b=m[m._merge=='both'];xd=b.xt_tv!=b.xt_en
print(f"fark: aynı giriş farklı çıkış {int(xd.sum())} işlem Δ {(b[xd].pnl_tv-b[xd].pnl_en).sum():+.0f} | sadece TV {m[m._merge=='left_only'].pnl_tv.sum():+.0f} | sadece motor {m[m._merge=='right_only'].pnl_en.sum():+.0f}")
