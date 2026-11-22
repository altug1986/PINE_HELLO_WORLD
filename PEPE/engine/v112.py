exec(open('btc2h.py').read().split("\nC1=[]")[0])
do,ex=comps(0);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
rows=[]
for km,kd in itertools.product([1,2,3],[1,2,3,4]):
    E=dict(ENG);E['MIK']=deb(ENG['MIK'].astype(np.int64),km) if km>1 else ENG['MIK'];E['MID']=deb(ENG['MID'].astype(np.int64),kd) if kd>1 else ENG['MID']
    h=EV5(chainY(E,pl),ex);rows.append(dict(MIK=km,MID=kd,**{k:h[k] for k in ['net','pf','n','upi','IS_upi','OOS_upi','mdd','y26','ff','s_net','s_upi']}))
print(pd.DataFrame(rows).round(2).to_string(index=False))
E=dict(ENG);E['MIK']=MIK2;E['MID']=deb(ENG['MID'].astype(np.int64),2);t2=chainY(E,pl)
V2=RUN5(t2,ex);V1=RUN5(chain2(pl),ex);pickle.dump(V2,open('V112.pkl','wb'))
T=B5.index;a=T.searchsorted(pd.Timestamp('2026-11-08 21:30',tz='UTC'));b=T.searchsorted(pd.Timestamp('2026-11-08 23:30',tz='UTC'))
for nm,r in [('V11.1',V1),('V11.2',V2)]:
    m=(r[2]>=a)&(r[1]<b);print(nm,'ekran penceresi:',[( (T[e]+pd.Timedelta(hours=3)).strftime('%H:%M'),(T[x]+pd.Timedelta(hours=3)).strftime('%H:%M'),int(d),round(p,1)) for e,x,d,p in zip(r[1][m],r[2][m],r[3][m],r[4][m])])
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V11.1':M(V1),'V11.2':M(V2),'V11.1 OOS':M(V1,I_SPLIT),'V11.2 OOS':M(V2,I_SPLIT)}).T[K].round(2).to_string())
rng=np.random.default_rng(1)
for nm,r in [('V11.1',V1),('V11.2',V2)]:
    te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK);hi=B5.high.values
    m=np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (hi[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
    e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];mo=e.resample('ME').last().diff();b6=e.resample('6ME').last().diff().dropna()
    pk=e.cummax();uw=e<pk-1e-9;runs=(uw!=uw.shift()).cumsum();u,upi=U(r)
    print(f'{nm}: Ulcer {u:.1f} UPI {upi:.1f} | yıl % {(y/5).round(0).astype(int).tolist()} | aylık medyan {mo.median():.0f} en kötü ay {mo.min():.0f} poz ay {(mo>0).mean():.2f} | MAE P95/P99/max {np.percentile(m,95):.1f}/{np.percentile(m,99):.1f}/{m.max():.1f} | zirve altı %{100*uw.mean():.0f} en uzun {uw.groupby(runs).sum().max()}g | 6ay {b6.round(0).astype(int).tolist()}')
    tp=r[4]
    for mode in ['boot','perm']:
        qq=rng.choice(tp,(4000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(4000)])
        qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
        print(f'  MC {mode}: net P1/P5 {np.percentile(ec[:,-1],[1,5]).round(0)} DD P95/P99 {np.percentile(ddm,[95,99]).round(0)}')
    ret=C[tx]/C[te]-1;sysn=(500*ret*td-0.4).sum();mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-0.4).sum() for _ in range(2000)]);dur=tx-te;mk2=[]
    for _ in range(800):
        s=rng.integers(300,N-dur.max()-1,len(dur));mk2.append((500*(C[s+dur]/C[s]-1)*rng.permutation(td)-0.4).sum())
    print(f'  MONKEY yön geçen {(mk>=sysn).mean():.4f} (max {mk.max():.0f}) | zaman geçen {(np.array(mk2)>=sysn).mean():.4f} (max {max(mk2):.0f}) | sistem {sysn:.0f}')
