import pickle, numpy as np, pandas as pd, sys
from eng import *
def tvtr(fn):
    d=pd.read_excel(fn,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
    e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');x=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası')
    return pd.DataFrame({'et':pd.to_datetime(e['Tarih ve saat']),'dir':np.where(e.Tip.str.contains('Kısa'),-1,1),'xt':pd.to_datetime(x['Tarih ve saat']),'pnl':x['Net K/Z USDT'].astype(float),'mae':x['Tersine hareket %'].astype(float).abs()})
def health(tr,lab,cap=500):
    s=tr.groupby(tr.xt.dt.floor('D')).pnl.sum();idx=pd.date_range(s.index.min(),s.index.max(),freq='D');eq=s.reindex(idx,fill_value=0).cumsum()
    pk=eq.cummax();uw=(eq<pk-1e-9);dd=pk-eq
    runs=(uw!=uw.shift()).cumsum();lu=uw.groupby(runs).sum().max()
    y=tr.groupby(tr.xt.dt.year).pnl.sum()/cap*100;mo=eq.resample('ME').last().diff().fillna(eq.resample('ME').last().iloc[0]);w=eq.resample('W').last().diff()
    t=np.arange(len(eq));r2=np.corrcoef(t,eq.values)[0,1]**2
    l12=eq[eq.index>=eq.index[-1]-pd.Timedelta(days=365)];r2_12=np.corrcoef(np.arange(len(l12)),l12.values)[0,1]**2
    print(f'{lab}: net {tr.pnl.sum():.0f} (%{tr.pnl.sum()/cap*100:.0f}) | işlem {len(tr)} | PF {tr.pnl[tr.pnl>0].sum()/-tr.pnl[tr.pnl<=0].sum():.2f} | MAE P95/P99/max {np.percentile(tr.mae,95):.1f}/{np.percentile(tr.mae,99):.1f}/{tr.mae.max():.1f}')
    print(f'   yıl %: {y.round(0).astype(int).to_dict()} | aylık medyan {mo.median():.0f} | poz ay {(mo>0).mean():.2f} hafta {(w>0).mean():.2f}')
    print(f'   ekside geçen gün oranı %{100*uw.mean():.0f} | en uzun eksi dönem {lu} gün | DD P95/P99/max {np.percentile(dd,95):.0f}/{np.percentile(dd,99):.0f}/{dd.max():.0f} | R² tüm {r2:.3f} son12ay {r2_12:.3f} | son 12 ay net {l12.iloc[-1]-l12.iloc[0]:.0f}')
    return eq
for fn in sys.argv[1:]:
    x=pd.ExcelFile(fn);o=pd.read_excel(x,'Özellikler');p=pd.read_excel(x,'Performans')
    print('==',fn,'| aralık:',o.iloc[0,1],'| TV net',p.iloc[2,1],p.iloc[2,2]);health(tvtr(fn),'TV')
V10=pickle.load(open('V10.pkl','rb'))
en=pd.DataFrame({'xt':B5.index[V10[2]].tz_localize(None)+pd.Timedelta(hours=3),'pnl':V10[4],'mae':np.zeros(len(V10[4]))})
tv=tvtr(sys.argv[1]);en2=pd.DataFrame({'et':B5.index[V10[1]].tz_localize(None)+pd.Timedelta(hours=3),'dir':V10[3]})
m=tv.merge(en2,on=['et','dir'],how='outer',indicator=True);print('SYNC',m._merge.value_counts().to_dict())
health(en,'MOTOR V10')
