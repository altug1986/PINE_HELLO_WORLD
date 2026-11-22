exec(open('fix2.py').read().split("mid=A240<20")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pm=h3();V62=run_x(C,np.where(t6!=0,t6,pm).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH);pickle.dump(V62,open('V62.pkl','wb'))
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
T=pd.DataFrame({'V6':M(V6),'V6.2':M(V62),'V6 OOS':M(V6,I_SPLIT),'V6.2 OOS':M(V62,I_SPLIT)}).T[K];T['ddTV~']=[tvdd(V6),tvdd(V62),np.nan,np.nan];print(T.round(2).to_string())
for nm,r in [('V6',V6),('V6.2',V62)]:
    eq=pd.Series(r[0][DAY_I],index=DAY_T);y=eq.resample('YE').last().diff();y.iloc[0]=eq.resample('YE').last().iloc[0];mo=eq.resample('ME').last().diff();w=eq.resample('W').last().diff()
    print(nm,'| yıl % (500):',(y/5).round(0).astype(int).tolist(),'medyan',round(np.median(y/5)),'| net %',round(r[0][-1]/5),'| aylık medyan',round(mo.median()),'yıl-aylık medyan',mo.groupby(mo.index.year).median().round(0).astype(int).tolist(),'| poz ay',round((mo>0).mean(),2),'hafta',round((w>0).mean(),2),'| en kötü ay',round(mo.min()),'| son30g',round(metrics(r[0],*r[2:6],*W1)['net']),'| 2024-08 pencere',round(metrics(r[0],*r[2:6],*W2)['net']))
win=lambda r:[round(metrics(r[0],*r[2:6],int(a),int(b))['net']) for a,b in zip(np.searchsorted(B5.index,[pd.Timestamp(x,tz='UTC') for x in ['2024-03-12','2025-04-04','2025-11-25','2026-05-05']]),np.searchsorted(B5.index,[pd.Timestamp(x,tz='UTC') for x in ['2024-04-11','2025-05-04','2025-12-25','2026-06-04']]))]
print('diğer kötü 30g pencereler V6',win(V6),'→ V6.2',win(V62))
tp=V62[4];rng=np.random.default_rng(1);q=rng.choice(tp,(5000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05);e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5',np.percentile(e[:,-1],5).round(0),'DD P95/P99',np.percentile(dd,[95,99]).round(0))
rs=run_x(C,np.where(t6!=0,t6,pm).astype(np.int64),prot,Z,0.,0.,0,0,3e-7,0.0008,CASH);print('STRESS',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
idx=B5.index;f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
for r,cl,lb,lw in [(V6,'#bbb','V6',.9),(V62,'#1f6feb','V6.2 (+orta rejim motoru)',1.3)]:
    ax[0].plot(idx,r[0],color=cl,lw=lw,label=lb);d_=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-d_,color=cl,lw=lw*.7)
for a,b in [('2024-07-29','2024-08-28'),('2026-10-20','2026-11-20')]:
    for x in ax: x.axvspan(pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC'),color='orange',alpha=.25)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');[x.grid(alpha=.3) for x in ax];ax[1].set_ylabel('DD $')
ax[0].set_title('PEPE 5M V6.2 — sabit 500$ sermaye/birim, basit getiri (turuncu: düzeltilen 30g pencereler)');f.tight_layout();f.savefig('pepe_v62.png',dpi=110)
