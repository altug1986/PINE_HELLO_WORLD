import pickle, numpy as np, pandas as pd
from eng import *
x=pd.ExcelFile('tv8.xlsx');p=pd.read_excel(x,'Performans');print(p.iloc[[2,6]].iloc[:,:3].to_string(index=False));print(p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,:3].to_string(index=False))
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()]
xx=d[d.Tip.str.contains('çıkış')]
mae=xx['Tersine hareket %'].astype(float).abs();pf=xx['Net K/Z USDT'].astype(float)
print('TV işlem',len(xx),'PF',round(pf[pf>0].sum()/-pf[pf<=0].sum(),3),'| MAE% P50/P90/P95/P99/max',np.percentile(mae,[50,90,95,99]).round(2),round(mae.max(),2),'| MAE>8%:',(mae>8).sum(),'>10%:',(mae>10).sum())
V8=pickle.load(open('V8.pkl','rb'));LIB=pickle.load(open('/home/claude/data/LIB.pkl','rb'))
te,tx,td,tp=V8[1],V8[2],V8[3],V8[4]
src=np.select([LIB['RSIX']!=0,LIB['MIK']!=0,LIB['COL']!=0,LIB['B4H']!=0,LIB['BFLIP']!=0],['RSIX','MIK','COL','B4H','BFLIP'],'MID')
ep=np.where(td>0,C[te]+TICK,C[te]-TICK)
mae_e=np.array([100*((ep[k]-L_[e+1:x_+1].min())/ep[k] if d_>0 else (H[e+1:x_+1].max()-ep[k])/ep[k]) if x_>e else 0 for k,(e,x_,d_) in enumerate(zip(te,tx,td))])
print('motor MAE% P95/P99/max',np.percentile(mae_e,[95,99]).round(2),round(mae_e.max(),2))
s=pd.DataFrame({'src':src[te],'pnl':tp,'mae':mae_e,'hold':(tx-te)*5/60,'oos':te>=I_SPLIT})
g=s.groupby('src').agg(n=('pnl','size'),net=('pnl','sum'),pf=('pnl',lambda v:v[v>0].sum()/-v[v<=0].sum()),win=('pnl',lambda v:(v>0).mean()*100),mae95=('mae',lambda v:np.percentile(v,95)),maemax=('mae','max'),hold=('hold','median'))
print(g.round(2).to_string())
print('OOS motor bazında:');print(s[s.oos].groupby('src').agg(n=('pnl','size'),net=('pnl','sum'),pf=('pnl',lambda v:v[v>0].sum()/-v[v<=0].sum())).round(2).to_string())
# MAE vs sonuç: MAE>X olan işlemlerin toplam neti
for X in [3,4,5,6,8]:
    m=s.mae>X;print(f'MAE>{X}%: {m.sum()} işlem, net {s.pnl[m].sum():.0f}, kazanan {(s.pnl[m]>0).sum()}')
