exec(open("ahf3.py").read().split("rows=[]")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
thr=4.5*ATRP/100.
V10=run_ahf2(C,t8,R7,thr,WEAK,0,.03,144,1,90.,60.,12,.03,TICK,COMM,CASH);pickle.dump(V10,open('V10.pkl','wb'))
V9e=run_ahf2(C,t8,R7,np.zeros(N),WEAK,0,.03,144,1,90.,60.,12,.03,TICK,COMM,CASH);V81=pickle.load(open('V81.pkl','rb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V8.1':M(V81),'V9':M(V9e),'V10':M(V10),'V8.1 OOS':M(V81,I_SPLIT),'V9 OOS':M(V9e,I_SPLIT),'V10 OOS':M(V10,I_SPLIT)}).T[K].round(2).to_string())
for nm,r in [('V8.1',V81),('V9',V9e),('V10',V10)]:
    m=mae(r);eq=pd.Series(r[0][DAY_I],index=DAY_T);y=eq.resample('YE').last().diff();y.iloc[0]=eq.resample('YE').last().iloc[0];mo=eq.resample('ME').last().diff();w=eq.resample('W').last().diff()
    wk=[metrics(r[0],*r[2:6],a,b)['net'] for a,b in WI];b6=eq.resample('6ME').last().diff().dropna()
    print(f"{nm}: MAE P95/P99/max {np.percentile(m,95):.2f}/{np.percentile(m,99):.2f}/{m.max():.2f} >8%:{(m>8).sum()} >10%:{(m>10).sum()} | yıl % {(y/5).round(0).astype(int).tolist()} | aylık medyan {mo.median():.0f} en kötü ay {mo.min():.0f} hafta {w.min():.0f} | poz ay {(mo>0).mean():.2f} hafta {(w>0).mean():.2f} | zayıf20 {sum(wk):.0f} ({sum(x>0 for x in wk)}/20) | 6ay blok {b6.round(0).astype(int).tolist()}")
r=V10;tp=r[4];rng=np.random.default_rng(1)
for mode in ['boot','perm']:
    q=rng.choice(tp,(5000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(5000)])
    q=np.where(rng.random(q.shape)<.05,0,q-0.05);e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
    print(f'MC {mode}: net P1/P5/P50 {np.percentile(e[:,-1],[1,5,50]).round(0)} | DD P50/P95/P99 {np.percentile(dd,[50,95,99]).round(0)} | zarar olas. {(e[:,-1]<=0).mean():.3f}')
te,tx,td=r[1],r[2],r[3];ret=C[tx]/C[te]-1;cost=0.4+1.0*TICK/C[te]*500;sysnet=(500*ret*td-cost).sum()
mk1=np.array([(500*ret*rng.choice([-1,1],len(ret))-cost).sum() for _ in range(3000)]);dur=tx-te;mk2=[]
for _ in range(1000):
    s=rng.integers(300,N-dur.max()-1,len(dur));mk2.append((500*(C[s+dur]/C[s]-1)*rng.permutation(td)-cost).sum())
mk2=np.array(mk2)
print(f'MONKEY yön: sistem {sysnet:.0f} | maymun P50 {np.median(mk1):.0f} P99 {np.percentile(mk1,99):.0f} max {mk1.max():.0f} geçen {(mk1>=sysnet).mean():.4f}')
print(f'MONKEY zaman: maymun P50 {np.median(mk2):.0f} P99 {np.percentile(mk2,99):.0f} max {mk2.max():.0f} geçen {(mk2>=sysnet).mean():.4f}')
for lab,tk,cm in [('komisyon %0.06 + 2 tick',2e-7,.0006),('komisyon %0.08 + 3 tick',3e-7,.0008)]:
    rs=run_ahf2(C,t8,R7,thr,WEAK,0,.03,144,1,90.,60.,12,.03,tk,cm,CASH);ms=M(rs);print('STRES',lab,{k:round(float(ms[k]),2) for k in ['net','pf','calmar','mdd']})
tag=r[6];print('çıkış türleri:',{k:int((tag==v).sum()) for k,v in [('ana dönüş',0),('mikro kâr al',1),('mikro geri giriş',2),('karşı poza geçiş',4),('ana yöne dönüş',5)]},'| karşı poz işlemlerinin neti',round(tp[np.r_[tag[1:],0]==5].sum() if len(tag)>1 else 0))
idx=B5.index;f,ax=plt.subplots(3,1,figsize=(12,10),sharex=False,gridspec_kw={'height_ratios':[3,1,1.4]})
for rr_,cl,lb,lw in [(V81,'#bbb','V8.1',.9),(V9e,'#f0a35e','V9',.9),(V10,'#1f6feb','V10 (karşı poz)',1.3)]:
    ax[0].plot(idx,rr_[0],color=cl,lw=lw,label=lb);d_=np.maximum.accumulate(np.maximum(rr_[0],0))-rr_[0];ax[1].plot(idx,-d_,color=cl,lw=lw*.7)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');ax[0].set_title('PEPE 5M V10 — sabit 500$, basit getiri (araştırma motoru)');ax[1].set_ylabel('DD $')
for rr_,cl,lb in [(V9e,'#f0a35e','V9'),(V10,'#1f6feb','V10')]: ax[2].hist(np.clip(mae(rr_),0,35),bins=70,histtype='step',color=cl,lw=1.4,label=f'{lb} MAE %')
ax[2].set_yscale('log');ax[2].legend();ax[2].set_xlabel('pozisyon içi ters hareket % (MAE)')
for a in ax: a.grid(alpha=.3)
f.tight_layout();f.savefig('pepe_v10.png',dpi=110)
