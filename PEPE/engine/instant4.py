exec(open('instant2.py').read().split("\nB=M(")[0])
def steady(p,xt):
    s=pd.Series(p,index=pd.to_datetime(xt)).sort_index();d_=s.groupby(s.index.floor('D')).sum();eq=d_.reindex(pd.date_range(d_.index.min(),d_.index.max(),freq='D'),fill_value=0).cumsum()
    dd=eq.cummax()-eq;uw=dd>1e-9;runs=(uw!=uw.shift()).cumsum();lu=int(uw.groupby(runs).sum().max())
    w=eq.resample('W').last().diff().dropna();mo=eq.resample('ME').last().diff().dropna();t=np.arange(len(eq));r2=np.corrcoef(t,eq.values)[0,1]**2
    l12=eq[eq.index>=eq.index.max()-pd.Timedelta(days=365)];m12=l12.resample('ME').last().diff().dropna()
    return {'Pozitif hafta %':100*(w>0).mean(),'Pozitif ay %':100*(mo>0).mean(),'Haftalık medyan $':w.median(),'Aylık medyan $':mo.median(),'Aylık CV':mo.std()/mo.mean(),
            'En kötü hafta $':w.min(),'En kötü ay $':mo.min(),'Zirve altı gün %':100*uw.mean(),'En uzun zirve altı (gün)':lu,'Eğri R²':r2,'Son 12 ay pozitif ay':f"{int((m12>0).sum())}/{len(m12)}"}
T={'TV V11.6 (WT yok)':steady(TR.pnl.values,TR.xt)}
for nm,cfg in [('6.5 / 0.15 / SL 15',(.065,.0015,.15)),('6.5 / 0.10 / SL yok',(.065,.001,0.))]:
    p,why,xt,xi=run2(*cfg);T[nm]=steady(p,xt)
pd.set_option('display.width',200);print(pd.DataFrame(T).round(2).to_string())
