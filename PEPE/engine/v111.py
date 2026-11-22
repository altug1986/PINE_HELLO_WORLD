exec(open('liq2.py').read().split("\nrows=[dict(p='V11'")[0])
pl=vstate(C,(dp6<=-4)&(do6<=-4),(dp6>=4)&(do6<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
V111=R13(t=chainP(pl));V11r=R13();pickle.dump(V111,open('V111.pkl','wb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V11':M(V11r),'V11.1':M(V111),'V11 OOS':M(V11r,I_SPLIT),'V11.1 OOS':M(V111,I_SPLIT)}).T[K].round(2).to_string())
for nm,r in [('V11',V11r),('V11.1',V111)]:
    te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK);hi=B5.high.values
    m=np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (hi[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
    e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];mo=e.resample('ME').last().diff();b6=e.resample('6ME').last().diff().dropna()
    pk=e.cummax();uw=e<pk-1e-9;runs=(uw!=uw.shift()).cumsum();u,upi=U(r)
    print(f'{nm}: Ulcer {u:.1f} UPI {upi:.1f} | yıl % {(y/5).round(0).astype(int).tolist()} | aylık medyan {mo.median():.0f} en kötü ay {mo.min():.0f} poz ay {(mo>0).mean():.2f} | MAE P95/P99/max {np.percentile(m,95):.1f}/{np.percentile(m,99):.1f}/{m.max():.1f} | zirve altı %{100*uw.mean():.0f} en uzun {uw.groupby(runs).sum().max()}g | 6ay {b6.round(0).astype(int).tolist()}')
r=V111;tp=r[4];rng=np.random.default_rng(1)
for mode in ['boot','perm']:
    qq=rng.choice(tp,(4000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(4000)])
    qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
    print(f'MC {mode}: net P1/P5 {np.percentile(ec[:,-1],[1,5]).round(0)} DD P95/P99 {np.percentile(ddm,[95,99]).round(0)} zarar olas. {(ec[:,-1]<=0).mean():.3f}')
te,tx,td=r[1],r[2],r[3];ret=C[tx]/C[te]-1;sysn=(500*ret*td-0.4).sum()
mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-0.4).sum() for _ in range(2000)]);dur=tx-te;mk2=[]
for _ in range(800):
    s=rng.integers(300,N-dur.max()-1,len(dur));mk2.append((500*(C[s+dur]/C[s]-1)*rng.permutation(td)-0.4).sum())
print(f'MONKEY yön geçen {(mk>=sysn).mean():.4f} (max {mk.max():.0f}) | zaman geçen {(np.array(mk2)>=sysn).mean():.4f} (max {max(mk2):.0f}) | sistem {sysn:.0f}')
rs=run_v13(C,chainP(pl),R7,thr,ZF,ZI,ps,2.0,EX,.03,144,90.,60.,12,.03,3e-7,0.0008,CASH);print('STRES',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
src=np.where(pl[V111[1]]!=0,1,0);print('likidasyon motoru işlemleri:',int(src.sum()),'net',round(V111[4][src==1].sum()),'PF',round(V111[4][(src==1)&(V111[4]>0)].sum()/-V111[4][(src==1)&(V111[4]<=0)].sum(),2))
