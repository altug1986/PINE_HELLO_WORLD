exec(open("gate.py").read().split("rows=[dict(")[0])
rows=[dict(L=0,pq=0,blk=0,exc=0,blocks=0,**H2(V10))]
for L,qq,blk,exc in itertools.product([120,150,180,210,240,300,365],[93,95,97,98],[84],[1,2]):
    rp=pd.Series(ddd).rolling(L,min_periods=L).quantile(qq/100).shift(1).values;thr=np.nan_to_num(rp,nan=0.)[dayof]
    r=run_gate(C,sp,ddb,thr,blk,1,exc,src,coreEvt,TICK,COMM,CASH);rows.append(dict(L=L,pq=qq,blk=blk,exc=exc,blocks=r[6],**H2(r)))
E=pd.DataFrame(rows);pd.set_option('display.width',300);E.to_pickle('gate3.pkl')
print(E.iloc[:1][['net','q','mdd','p99','Ocal','l12','r2_12']].round(2).to_string(index=False))
for exc in [1,2]:
    print(f'exc={exc} kalite (net/MaxDD) — satır: pencere gün, sütun: eşik persentil');print(E[E.exc==exc].pivot_table(index='L',columns='pq',values='q').round(1))
    print(f'exc={exc} net:');print(E[E.exc==exc].pivot_table(index='L',columns='pq',values='net').round(0))
    print(f'exc={exc} OOS Calmar:');print(E[E.exc==exc].pivot_table(index='L',columns='pq',values='Ocal').round(2))
