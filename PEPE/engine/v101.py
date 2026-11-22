exec(open('sharpe2.py').read().split("\nrows=[dict(var='V10',W=0")[0])
V101=R(PS[864],2.0,-1.);V10r=R(PS[864],-1.,-1.);pickle.dump(V101,open('V101.pkl','wb'))
def mae(r):
    te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK)
    return np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (B5.high.values[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V10':M(V10r),'V10.1':M(V101),'V10 OOS':M(V10r,I_SPLIT),'V10.1 OOS':M(V101,I_SPLIT)}).T[K].round(2).to_string())
for nm,r in [('V10',V10r),('V10.1',V101)]:
    m=mae(r);e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];mo=e.resample('ME').last().diff();b6=e.resample('6ME').last().diff().dropna()
    tp=r[4];rng=np.random.default_rng(1);qq=rng.choice(tp,(3000,len(tp)));qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
    print(f'{nm}: yıl % {(y/5).round(0).astype(int).tolist()} | aylık medyan {mo.median():.0f} en kötü ay {mo.min():.0f} | MAE P95/P99/max {np.percentile(m,95):.1f}/{np.percentile(m,99):.1f}/{m.max():.1f} | 6ay blok {b6.round(0).astype(int).tolist()} | MC net P5 {np.percentile(ec[:,-1],5):.0f} DD P99 {np.percentile(ddm,99):.0f}')
rs=run_v11(C,t10,R7,thr,PS[864],2.0,-1.,.03,144,90.,60.,12,.03,3e-7,0.0008,CASH);print('STRES %0.08+3tick:',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
tag=V101[6];print('mikro kâr al sayısı V10',int((V10r[6]==1).sum()),'→ V10.1',int((tag==1).sum()))
