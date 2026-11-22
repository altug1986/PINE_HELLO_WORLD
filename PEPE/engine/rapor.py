import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('h10.py').read().split("for fn in sys.argv")[0])
tr=tvtr('tv10_15123536.xlsx');p=pd.read_excel('tv10_15123536.xlsx','Performans');cap=500
s=tr.groupby(tr.xt.dt.floor('D')).pnl.sum();idx=pd.date_range(s.index.min(),s.index.max(),freq='D');eq=s.reindex(idx,fill_value=0).cumsum()
pk=eq.cummax();dd=pk-eq;uw=dd>1e-9;runs=(uw!=uw.shift()).cumsum();uwl=uw.groupby(runs).sum()
mo=eq.resample('ME').last().diff();mo.iloc[0]=eq.resample('ME').last().iloc[0];w=eq.resample('W').last().diff()
r=s.reindex(idx,fill_value=0)/cap;sh=r.mean()/r.std()*np.sqrt(365);so=r.mean()/np.sqrt((np.minimum(r,0)**2).mean())*np.sqrt(365)
yrs=len(idx)/365.25;cal=tr.pnl.sum()/yrs/dd.max()
w_=tr.pnl[tr.pnl>0];l_=tr.pnl[tr.pnl<=0]
roll=(eq-eq.shift(365)).dropna()/cap*100
H=dict(net=tr.pnl.sum(),pct=tr.pnl.sum()/cap*100,n=len(tr),pf=w_.sum()/-l_.sum(),win=100*len(w_)/len(tr),avgw=w_.mean(),avgl=l_.mean(),sh=sh,so=so,cal=cal,
       dd95=np.percentile(dd,95),dd98=np.percentile(dd,98),dd99=np.percentile(dd,99),ddmax=dd.max(),uwp=100*uw.mean(),uwmax=uwl.max(),uwmed=uwl[uwl>0].median(),
       mae95=np.nanpercentile(tr.mae,95),mae99=np.nanpercentile(tr.mae,99),maemax=tr.mae.max(),pm=(mo>0).mean(),pw=(w>0).mean(),mmed=mo.median(),mcv=mo.std()/mo.mean(),wm=mo.min(),ww=w.min(),
       r12min=roll.min(),r12med=roll.median(),comm=p.iloc[6,1])
for k,v in H.items(): print(k,round(float(v),2))
print('yıl %',(tr.groupby(tr.xt.dt.year).pnl.sum()/5).round(0).to_dict())
print('mum içi MaxDD TV',p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,1].tolist())
f=plt.figure(figsize=(13,12));gs=f.add_gridspec(4,2,height_ratios=[2.2,1,1.4,1.2])
a=f.add_subplot(gs[0,:]);a.plot(eq.index,eq.values/5,color='#1f6feb',lw=1.3);a.fill_between(eq.index,eq.values/5,pk.values/5,where=uw.values,color='#d1242f',alpha=.25,label='zirve altında')
a.axvline(pd.Timestamp('2025-05-20'),ls='--',color='k',lw=.8);a.text(pd.Timestamp('2025-05-25'),eq.max()/5*.05,'OOS →');a.set_title(f'PEPE 5M V10 — TradingView işlemleri, 500$ sabit | net %{H["pct"]:.0f}, PF {H["pf"]:.2f}, {H["n"]} işlem');a.set_ylabel('% (500$)');a.legend(loc='upper left');a.grid(alpha=.3)
b=f.add_subplot(gs[1,:],sharex=a);b.fill_between(dd.index,-dd.values,color='#d1242f',alpha=.6)
for q,cl in zip([95,98,99],['#bf8700','#d1242f','#6e1b1b']): b.axhline(-np.percentile(dd,q),ls='--',lw=.8,color=cl,label=f'P{q} {np.percentile(dd,q):.0f}$')
b.legend(fontsize=8,loc='lower left');b.set_ylabel('DD $');b.grid(alpha=.3)
c=f.add_subplot(gs[2,0]);hm=mo.to_frame('v');hm['y']=hm.index.year;hm['m']=hm.index.month;tab=hm.pivot(index='y',columns='m',values='v')
im=c.imshow(tab.values,cmap='RdYlGn',vmin=-300,vmax=300,aspect='auto');c.set_xticks(range(12));c.set_xticklabels(range(1,13));c.set_yticks(range(len(tab)));c.set_yticklabels(tab.index)
for i_ in range(tab.shape[0]):
    for j_ in range(tab.shape[1]):
        v=tab.values[i_,j_]
        if np.isfinite(v): c.text(j_,i_,f'{v:.0f}',ha='center',va='center',fontsize=7)
c.set_title('Aylık net $')
d=f.add_subplot(gs[2,1]);d.plot(roll.index,roll.values,color='#1f6feb');d.axhline(200,ls='--',color='k',lw=.8);d.set_title('Kayan 12 ay getiri % (500$)');d.grid(alpha=.3)
e=f.add_subplot(gs[3,0]);e.hist(tr.mae.dropna().clip(0,25),bins=50,color='#8250df');[e.axvline(np.nanpercentile(tr.mae,q),ls='--',color=cl,lw=.9) for q,cl in [(95,'#bf8700'),(99,'#d1242f')]];e.set_yscale('log');e.set_title('Pozisyon içi ters hareket (MAE %)')
g=f.add_subplot(gs[3,1]);rs=r.rolling(180).mean()/r.rolling(180).std()*np.sqrt(365);g.plot(rs.index,rs.values,color='#1a7f37');g.axhline(0,color='k',lw=.6);g.set_title('Kayan 180 gün Sharpe');g.grid(alpha=.3)
f.tight_layout();f.savefig('pepe_v10_saglik.png',dpi=105)
