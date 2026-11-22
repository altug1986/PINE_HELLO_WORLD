exec(open('micro.py').read().split("\nrows=[dict(h='V11.1'")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
IG=vstate(C,(oz1>=2)&(bp1>=.6),(oz1>=2)&(bp1<=-.6),0.,.03,24,F,F,0).astype(np.int64)
V12=R13(t=chain2(PL,IG,'colust'));V111=R13(t=base);pickle.dump(V12,open('V12.pkl','wb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def full(r):
    f=M(r);o=M(r,I_SPLIT);te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK);hi=B5.high.values
    m=np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (hi[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
    e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];mo=e.resample('ME').last().diff();w=e.resample('W').last().diff()
    pk=e.cummax();uw=e<pk-1e-9;runs=(uw!=uw.shift()).cumsum();u,upi=U(r);uo,upio=U(r,I_SPLIT,N);l12=e[e.index>=e.index[-1]-pd.Timedelta(days=365)]
    tp=r[4];rng=np.random.default_rng(1);qq=rng.choice(tp,(4000,len(tp)));qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
    qp=np.stack([rng.permutation(tp) for _ in range(3000)]);ecp=np.cumsum(qp,1);ddp=(np.maximum.accumulate(np.maximum(ecp,0),1)-ecp).max(1)
    ret=C[tx]/C[te]-1;sysn=(500*ret*td-0.4).sum();mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-0.4).sum() for _ in range(2000)])
    dur=tx-te;mk2=np.array([(500*(C[s+dur]/C[s]-1)*rng.permutation(td)-0.4).sum() for s in [rng.integers(300,N-dur.max()-1,len(dur)) for _ in range(800)]])
    return {'Net $':f['net'],'Getiri % (500$)':f['net']/5,'İşlem':f['n'],'PF':f['pf'],'Kazanan %':f['win'],'İşlem başı edge (bps)':1e4*f['net']/f['n']/500,'Komisyon $':f['comm'],
            'Sharpe':f['sharpe'],'Sortino':f['sortino'],'Calmar':f['calmar'],'Ulcer $':u,'UPI':upi,'DD P95':f['p95'],'DD P98':f['p98'],'DD P99':f['p99'],'MaxDD':f['mdd'],
            'MAE P95 %':np.percentile(m,95),'MAE P99 %':np.percentile(m,99),'MAE max %':m.max(),'Zirve altı gün %':100*uw.mean(),'En uzun zirve altı (gün)':uw.groupby(runs).sum().max(),
            'Poz ay %':100*(mo>0).mean(),'Poz hafta %':100*(w>0).mean(),'Aylık medyan $':mo.median(),'En kötü ay $':mo.min(),'Aylık CV':f['cv'],
            '2023 %':y.iloc[0]/5,'2024 %':y.iloc[1]/5,'2025 %':y.iloc[2]/5,'2026 %':y.iloc[3]/5,'Son 12 ay $':l12.iloc[-1]-l12.iloc[0],'Son 12 ay R²':np.corrcoef(np.arange(len(l12)),l12.values)[0,1]**2,
            'OOS net $':o['net'],'OOS PF':o['pf'],'OOS Calmar':o['calmar'],'OOS UPI':upio,
            'MC boot net P1':np.percentile(ec[:,-1],1),'MC boot DD P99':np.percentile(ddm,99),'MC perm DD P99':np.percentile(ddp,99),'MC zarar olas.':(ec[:,-1]<=0).mean(),
            'Monkey yön geçen':(mk>=sysn).mean(),'Monkey yön max $':mk.max(),'Monkey zaman geçen':(mk2>=sysn).mean(),'Monkey zaman max $':mk2.max()}
T=pd.DataFrame({'V11.1 (baz)':full(V111),'V12':full(V12)});T['fark']=T['V12']-T['V11.1 (baz)']
pd.set_option('display.width',200);print(T.round(2).to_string())
for lab,tk,cm in [('%0.06+2tick',2e-7,.0006),('%0.08+3tick',3e-7,.0008)]:
    for nm,t_ in [('V11.1',base),('V12',chain2(PL,IG,'colust'))]:
        rs=run_v13(C,t_,R7,thr,ZF,ZI,ps,2.0,EX,.03,144,90.,60.,12,.03,tk,cm,CASH);print('STRES',lab,nm,{k:round(float(M(rs)[k]),1) for k in ['net','pf','calmar','mdd']})
ig_tr=IG[V12[1]]!=0;print('ateşleme motoru işlemleri:',int(ig_tr.sum()),'net',round(V12[4][ig_tr].sum()),'PF',round(V12[4][ig_tr&(V12[4]>0)].sum()/-V12[4][ig_tr&(V12[4]<=0)].sum(),2),'| OOS içinde',int((ig_tr&(V12[1]>=I_SPLIT)).sum()),'işlem net',round(V12[4][ig_tr&(V12[1]>=I_SPLIT)].sum()))
idx=B5.index;f,ax=plt.subplots(3,1,figsize=(12,10),sharex=True,gridspec_kw={'height_ratios':[3,1,1.3]})
for r,cl,lb,lw in [(V111,'#999','V11.1 (baz)',1.),(V12,'#1f6feb','V12 (+BTC OI ateşleme)',1.3)]:
    ax[0].plot(idx,r[0]/5,color=cl,lw=lw,label=lb);dd=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-dd,color=cl,lw=lw*.7)
    e=pd.Series(r[0][DAY_I],index=DAY_T);rl=(e-e.shift(365)).dropna()/5;ax[2].plot(rl.index,rl.values,color=cl,lw=lw)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');ax[0].set_ylabel('% (500$)');ax[0].set_title('PEPE 5M V12 vs V11.1 — sabit 500$, basit getiri (araştırma motoru)')
ax[1].set_ylabel('DD $');ax[2].set_ylabel('kayan 12 ay %');[a.grid(alpha=.3) for a in ax];f.tight_layout();f.savefig('pepe_v12.png',dpi=105)
