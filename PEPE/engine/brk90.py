exec(open('breaker.py').read().split("\nrows=[row(SH")[0])
rows=[row(SH,X=0,W=0,J=-1)]
for X,W,J in itertools.product([80,90,100],[0,3,7,14],[0,1]): rows.append(row(brk(C,posS,float(X),3*288,W,J,dayid,NDy,TICK,COMM,CASH),X=X,W=W,J=J))
R=pd.DataFrame(rows);pd.set_option('display.width',300)
print(R[['X','W','J','trips','net','n','pf','sharpe','calmar','p95','p98','p99','mdd','ulcer','upi','Inet','Onet','l12']].round(1).to_string(index=False))
