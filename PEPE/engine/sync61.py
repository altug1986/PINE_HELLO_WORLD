exec(open('q8.py').read().split("K=['net','pf','n','calmar'")[0])
base=np.where(pR!=0,pR,np.where(t4!=0,t4,mrB)).astype(np.int64)
V61=run_t(C,base,TICK,COMM,CASH);pickle.dump(V61,open('V61.pkl','wb'))
x=pd.ExcelFile('tv61.xlsx');p=pd.read_excel(x,'Performans');print(p.iloc[[0,1,2,3,4,6]].to_string())
for sh in x.sheet_names[1:3]:
    q=pd.read_excel(x,sh);print(q.head(14).to_string())
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
tv=pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(xx['Tarih ve saat']),'pnl':xx['Net K/Z USDT'].astype(float)})
eq,te,tx,td,tp,tc=V61
en=pd.DataFrame({'et':B5.index[te].tz_localize(None)+pd.Timedelta(hours=3),'dir':td,'xt':B5.index[tx].tz_localize(None)+pd.Timedelta(hours=3),'pnl':tp})
m=tv.merge(en,on=['et','dir'],how='outer',suffixes=('_tv','_en'),indicator=True);b=m[m._merge=='both']
print('SYNC',m._merge.value_counts().to_dict(),'exit',(b.xt_tv==b.xt_en).sum(),'/',len(b),'TV net',round(tv.pnl.sum(),1),'EN net',round(en.pnl.sum(),1),'|Δpnl| top',round((b.pnl_tv-b.pnl_en).abs().sum(),1))
print(m[m._merge!='both'].sort_values('et').head(6)[['et','dir','_merge','pnl_tv','pnl_en']].to_string())
y=tv.groupby(tv.xt.dt.year).pnl.sum();mo=tv.groupby(tv.xt.dt.to_period('M')).pnl.sum()
print('TV yıl % (500):',(y/5).round(0).to_dict(),'medyan',round(np.median(y/5)),'| aylık medyan',round(mo.median(),1),'yıl aylık medyan',mo.groupby(mo.index.year).median().round(0).to_dict(),'poz ay',round((mo>0).mean(),2))
