exec(open('rollb.py').read().split("\nRES=[];PF_={}")[0])
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def run_b(K,th,D,mB,em):
    g=ban_gate(N,txs,roll_tr(tps,K)[0],th,D*288,mB==1)
    return runT(C,tgt(~sm|g[teS]),TICK,COMM,CASH) if em=='giriş' else runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH)
L12=DAY_I[-365]
def col(r):
    f=M(r);o=M(r,I_SPLIT);ul,up=U(r);td,tp=r[3],r[4];e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff()
    return {'Net $':f['net'],'Short işlem':int((td<0).sum()),'Short net':tp[td<0].sum(),'PF':f['pf'],'Sharpe':f['sharpe'],'Calmar':f['calmar'],'DD P95':f['p95'],'DD P98':f['p98'],'DD P99':f['p99'],'MaxDD':f['mdd'],'UPI':up,
            'İlk yarı net':r[0][I_SPLIT-1],'İkinci yarı net':o['net'],'Son 12 ay net':r[0][-1]-r[0][L12],'2026 %':y.iloc[-1]/5}
T={'BAZ V11.2':col(SH)}
for K,th,D,mB,em,lab in [(20,1.,3,0,'giriş','son20 PF<1 → 3g yasak'),(40,1.,3,0,'giriş','son40 PF<1 → 3g yasak'),(80,1.,3,0,'giriş','son80 PF<1 → 3g yasak'),(120,1.,3,0,'giriş','son120 PF<1 → 3g yasak'),(80,.9,1,0,'sert','son80 PF<0.9 → 1g yasak')]: T[lab]=col(run_b(K,th,D,mB,em))
pd.set_option('display.width',260);print(pd.DataFrame(T).round(2).to_string())
