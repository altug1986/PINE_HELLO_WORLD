exec(open('ga2.py').read().split("# baz V6.2")[0])
g=[int(c) for c in '30002103001022'];m8,V8=evalw(g,1,True);pickle.dump(V8,open('V8.pkl','wb'))
V62,V63=pickle.load(open('V62.pkl','rb')),pickle.load(open('V63.pkl','rb'))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
K=['net','pf','n','win','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','posm','cv']
print(pd.DataFrame({'V6.2':M(V62),'V6.3':M(V63),'V8':M(V8),'V6.2 OOS':M(V62,I_SPLIT),'V6.3 OOS':M(V63,I_SPLIT),'V8 OOS':M(V8,I_SPLIT)}).T[K].round(2).to_string())
for nm,r in [('V6.2',V62),('V6.3',V63),('V8',V8)]:
    eq=pd.Series(r[0][DAY_I],index=DAY_T);y=eq.resample('YE').last().diff();y.iloc[0]=eq.resample('YE').last().iloc[0];mo=eq.resample('ME').last().diff()
    wk=[metrics(r[0],*r[2:6],a,b)['net'] for a,b,_ in WI]
    print(nm,'| yıl %',(y/5).round(0).astype(int).tolist(),'| aylık medyan',round(mo.median()),'| en kötü ay',round(mo.min()),'| zayıf20: toplam',round(sum(wk)),'pozitif',sum(x>0 for x in wk),'/20 medyan',round(np.median(wk)))
src=np.select([LIB['RSIX']!=0,LIB['MIK']!=0,LIB['COL']!=0,LIB['B4H']!=0,LIB['BFLIP']!=0],['RSIX','MIK','COL','B4H','BFLIP'],'MID')
print('süre payı %',(100*pd.Series(src).value_counts(normalize=True)).round(1).to_dict())
tp=V8[4];rng=np.random.default_rng(1);q=rng.choice(tp,(5000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05);e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5',np.percentile(e[:,-1],5).round(0),'DD P95/P99',np.percentile(dd,[95,99]).round(0))
rs=run_t(C,target(g,1),3e-7,0.0008,CASH);print('STRESS',{k:round(float(M(rs)[k]),2) for k in ['net','pf','calmar','mdd']})
