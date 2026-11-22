exec(open('rolls.py').read().split("\ntxs,tps=txS[sm],tpS[sm]")[0])
txs,tps=txS[sm],tpS[sm]
def f1(K,th): g=gok(tobar(txs,roll_tr(tps,K)[0]),th);return runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH)
def f2(W,th): g=gok(wpf(sm,W),th);return runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH)
def f3(K,th): g=gok(tobar(txS,roll_tr(tpS,K)[0]),th);return runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH)
EK=['RSIX','MIK','COL','B4H','BFLIP','MID'];ST={k:E[k].astype(np.int64) for k in EK};ST['PL']=pl;VT={k:vtr2(C,ST[k]) for k in ST}
def f4(keys,K,th):
    E2=dict(E);pl2=pl
    for k in keys:
        tx_,pn_,d_=VT[k];ms=d_<0;g=gok(tobar(tx_[ms],roll_tr(pn_[ms],K)[0]),th)
        if k=='PL': pl2=np.where((pl<0)&~g,0,pl).astype(np.int64)
        else: E2[k]=np.where((E[k]<0)&~g,0,E[k])
    return RUN5(chainY(E2,pl2),ex)
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def col(r):
    f=M(r);o=M(r,I_SPLIT);ul,up=U(r);e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();td,tp=r[3],r[4]
    return {'Net $':f['net'],'Getiri %':f['net']/5,'İşlem':f['n'],'Short işlem':int((td<0).sum()),'Short net':tp[td<0].sum(),'Short PF':tp[(td<0)&(tp>0)].sum()/-tp[(td<0)&(tp<=0)].sum(),'PF':f['pf'],'Sharpe':f['sharpe'],'Sortino':f['sortino'],'Calmar':f['calmar'],
            'DD P95':f['p95'],'DD P98':f['p98'],'DD P99':f['p99'],'MaxDD':f['mdd'],'Ulcer':ul,'UPI':up,'IS net (seçim)':r[0][I_SPLIT-1],'OOS net (test)':o['net'],'OOS PF':o['pf'],'2026 %':y.iloc[-1]/5}
T={'BAZ V11.2':col(SH),'F1 son120 short PF≥1.0':col(f1(120,1.)),'F2 son30 gün short PF≥0.7':col(f2(30,.7)),'F3 genel son20 PF≥0.8':col(f3(20,.8)),'F4 yan motor son10 short PF≥0.8':col(f4(['MIK','B4H','BFLIP','MID'],10,.8))}
pd.set_option('display.width',260);print(pd.DataFrame(T).round(2).to_string())
