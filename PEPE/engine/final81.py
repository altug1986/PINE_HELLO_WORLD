exec(open("reg2.py").read().split("\nrows=[]\n")[0])
order=['RSIX','MIK','COL','B4H','BFLIP','MID'];t,src=build(order);karr=np.zeros(N);karr[(src==3)|(src==5)]=3.0
V81=run_k(O,H,L_,C,t,karr,ATRP,TICK,COMM,CASH);V8=pickle.load(open('V8.pkl','rb'));pickle.dump(V81,open('V81.pkl','wb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
T=pd.DataFrame({'V8':M(V8),'V8.1':M(V81),'V8 OOS':M(V8,I_SPLIT),'V8.1 OOS':M(V81,I_SPLIT)}).T[K];print(T.round(2).to_string())
for nm,r in [('V8',V8),('V8.1',V81)]:
    m=mae(r);eq=pd.Series(r[0][DAY_I],index=DAY_T);y=eq.resample('YE').last().diff();y.iloc[0]=eq.resample('YE').last().iloc[0];mo=eq.resample('ME').last().diff()
    wk=[metrics(r[0],*r[2:6],a,b)['net'] for a,b in WI]
    print(nm,'| MAE% P50/P95/P99/max',np.percentile(m,[50,95,99]).round(2),round(m.max(),2),'>8%:',(m>8).sum(),'| yıl %',(y/5).round(0).astype(int).tolist(),'| aylık medyan',round(mo.median()),'en kötü ay',round(mo.min()),'| zayıf20',round(sum(wk)),'| stop',int(r[6].sum()) if len(r)>6 else 0)
# Monte Carlo (bootstrap + permütasyon, %5 atlama, +1 tick ek maliyet)
tp=V81[4];rng=np.random.default_rng(1)
for mode in ['boot','perm']:
    q=rng.choice(tp,(5000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(5000)])
    q=np.where(rng.random(q.shape)<.05,0,q-0.05);e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
    print(f'MC {mode}: net P1/P5/P50 {np.percentile(e[:,-1],[1,5,50]).round(0)} | DD P50/P95/P99 {np.percentile(dd,[50,95,99]).round(0)} | zarar olasılığı {(e[:,-1]<=0).mean():.3f}')
# Monkey testleri
te,tx,td=V81[1],V81[2],V81[3];ret=C[tx]/C[te]-1;cost=0.4+2*TICK/C[te]*500
sysnet=(500*ret*td-cost).sum()
mk1=np.array([(500*ret*rng.choice([-1,1],len(ret))-cost).sum() for _ in range(3000)])
dur=tx-te;mk2=[]
for _ in range(1000):
    s=rng.integers(300,N-dur.max()-1,len(dur));d_=rng.permutation(td);mk2.append((500*(C[s+dur]/C[s]-1)*d_-cost).sum())
mk2=np.array(mk2)
print(f'MONKEY yön rastgele (aynı zamanlama): sistem {sysnet:.0f}$ | maymun P50 {np.median(mk1):.0f} P99 {np.percentile(mk1,99):.0f} max {mk1.max():.0f} | sistemi geçen oran {(mk1>=sysnet).mean():.4f}')
print(f'MONKEY zaman rastgele (aynı süre/yön): maymun P50 {np.median(mk2):.0f} P99 {np.percentile(mk2,99):.0f} max {mk2.max():.0f} | sistemi geçen oran {(mk2>=sysnet).mean():.4f}')
# Walk-forward blokları (6 ay)
eqs=pd.Series(V81[0][DAY_I],index=DAY_T);b6=eqs.resample('6ME').last().diff().dropna()
print('6 aylık bloklar net:',b6.round(0).astype(int).tolist(),'| pozitif',(b6>0).sum(),'/',len(b6))
