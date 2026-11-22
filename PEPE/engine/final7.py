exec(open('q8.py').read().split("K=['net','pf','n','calmar'")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tfill=np.where((A240>30)&(DI240>0),1,np.where((A240>30)&(DI240<0),-1,0))
base=np.where(pR!=0,pR,np.where(t4!=0,t4,np.where(mrB!=0,mrB,tfill))).astype(np.int64)
hh,ll=donch(48);V7=run_pyr(C,base,C>hh,C<ll,3,48,0,TICK,COMM,CASH);pickle.dump(V7,open('V7.pkl','wb'))
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
print(pd.DataFrame({'V6':M(V6),'V7':M(V7),'V6 OOS':M(V6,I_SPLIT),'V7 OOS':M(V7,I_SPLIT)}).T[K].round(2).to_string())
yrs=N/(288*365.25)
for nm,r in [('V6',V6),('V7',V7)]:
    m=M(r);mo=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff();w=pd.Series(r[0][DAY_I],index=DAY_T).resample('W').last().diff()
    print(nm,'yıllık %/birim',round(100*m['net']/yrs/500),'| net/MaxDD',round(m['net']/m['mdd'],2),'| aylık medyan',round(mo.median()),'| yıl medyan',mo.groupby(mo.index.year).median().round(0).to_dict(),'| poz ay',round((mo>0).mean(),2),'poz hafta',round((w>0).mean(),2),'| en kötü ay',round(mo.min()),'hafta',round(w.min()))
idx=B5.index;Y={}
for a,b in [('2023-05-01','2024-01-01'),('2024-01-01','2025-01-01'),('2025-01-01','2026-01-01'),('2026-01-01','2026-11-20'),('2026-05-19','2026-11-20'),('2026-10-20 14:00','2026-11-20')]:
    i0,i1=max(np.searchsorted(idx,pd.Timestamp(a,tz='UTC')),1),np.searchsorted(idx,pd.Timestamp(b,tz='UTC'))
    Y[a[:10]+'→']={nm:f"{M(r,i0,i1)['net']:.0f}/{M(r,i0,i1)['pf']:.2f}/{M(r,i0,i1)['mdd']:.0f}" for nm,r in [('V6',V6),('V7',V7)]}
print('net/PF/MaxDD\n',pd.DataFrame(Y).T.to_string())
src=pd.Series(np.select([pR!=0,t4!=0,mrB!=0,tfill!=0],['RSIUÇ','V4','YATAY','HTFTREND'],'FLAT'))
print('süre payı %',(100*src.value_counts(normalize=True)).round(1).to_dict(),'| birim dağılımı %',(100*V7[6]/N).round(1))
tp=V7[4];rng=np.random.default_rng(1);q=rng.choice(tp,(3000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1);print('MC net P5',np.percentile(e[:,-1],5).round(0),'DD P95/P99',np.percentile(dd,[95,99]).round(0))
rs=run_pyr(C,base,C>hh,C<ll,3,48,0,3e-7,0.0008,CASH);print('STRESS',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
f,ax=plt.subplots(3,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[3,1,1]})
for r,cl,lb,lw in [(V6,'#bbb','V6',.9),(V7,'#1f6feb','V7 AMFİBİK',1.3)]:
    ax[0].plot(idx,r[0],color=cl,lw=lw,label=lb);dd_=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-dd_,color=cl,lw=lw*.7)
    mo=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff();ax[2].plot(mo.index,mo.rolling(6).median(),color=cl,lw=lw,label=f'{lb} 6-ay kayan medyan')
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');ax[1].set_ylabel('DD $');ax[2].set_ylabel('aylık medyan $');ax[2].legend(fontsize=8,loc='upper left')
for a in ax: a.grid(alpha=.3)
ax[0].set_title('PEPE 5M V7 AMFİBİK — 500$ birim (maks 3), basit getiri, MTM (araştırma motoru)');f.tight_layout();f.savefig('pepe_swing_v7.png',dpi=110)
