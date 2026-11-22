exec(open('oi2.py').read().split("D=pd.read_pickle('oi1.pkl')")[0])
dp6,do6=chg(C,6),chg(OI,6)
ex=np.where((dp6>=3)&(do6<=-2),1,np.where((dp6<=-3)&(do6<=-2),-1,0)).astype(np.int64)
pl=vstate(C,(dp6<=-10)&(do6<=-5),(dp6>=10)&(do6<=-5),0.,.08,72,R7>60,R7<40,0).astype(np.int64)
print('likidasyon motoru tetik sayısı:',int((np.diff(np.r_[0,(pl!=0).astype(int)])==1).sum()),'| tükeniş kilidi tetik bar sayısı:',int((ex!=0).sum()))
V11=RUN(t=np.where(pl!=0,pl,t10).astype(np.int64),ex=ex,useex=1);V101=RUN();pickle.dump(V11,open('V11.pkl','wb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V10.1':M(V101),'V11':M(V11),'V10.1 OOS':M(V101,I_SPLIT),'V11 OOS':M(V11,I_SPLIT)}).T[K].round(2).to_string())
for nm,r in [('V10.1',V101),('V11',V11)]:
    te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK);hi=B5.high.values
    m=np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (hi[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
    e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];mo=e.resample('ME').last().diff();b6=e.resample('6ME').last().diff().dropna()
    pk=e.cummax();uw=e<pk-1e-9;runs=(uw!=uw.shift()).cumsum()
    print(f'{nm}: yıl % {(y/5).round(0).astype(int).tolist()} | aylık medyan {mo.median():.0f} en kötü ay {mo.min():.0f} poz ay {(mo>0).mean():.2f} | MAE P95/P99/max {np.percentile(m,95):.1f}/{np.percentile(m,99):.1f}/{m.max():.1f} | zirve altı %{100*uw.mean():.0f} en uzun {uw.groupby(runs).sum().max()}g | 6ay {b6.round(0).astype(int).tolist()}')
r=V11;tp=r[4];rng=np.random.default_rng(1)
for mode in ['boot','perm']:
    qq=rng.choice(tp,(4000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(4000)])
    qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
    print(f'MC {mode}: net P1/P5 {np.percentile(ec[:,-1],[1,5]).round(0)} DD P95/P99 {np.percentile(ddm,[95,99]).round(0)} zarar olas. {(ec[:,-1]<=0).mean():.3f}')
te,tx,td=r[1],r[2],r[3];ret=C[tx]/C[te]-1;cost=0.4;sysn=(500*ret*td-cost).sum()
mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-cost).sum() for _ in range(2000)]);dur=tx-te;mk2=[]
for _ in range(800):
    s=rng.integers(300,N-dur.max()-1,len(dur));mk2.append((500*(C[s+dur]/C[s]-1)*rng.permutation(td)-cost).sum())
print(f'MONKEY yön: geçen {(mk>=sysn).mean():.4f} (max {mk.max():.0f}) | zaman: geçen {(np.array(mk2)>=sysn).mean():.4f} (max {max(mk2):.0f}) | sistem {sysn:.0f}')
rs=run_v12(C,np.where(pl!=0,pl,t10).astype(np.int64),R7,thr,ps,2.0,ex,1,Z,0,Z,0,.03,144,90.,60.,12,.03,3e-7,0.0008,CASH);print('STRES %0.08+3tick',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
tg=V11[6];print('tükeniş kilidi ile kapanan',int((tg==6).sum()),'işlem')
