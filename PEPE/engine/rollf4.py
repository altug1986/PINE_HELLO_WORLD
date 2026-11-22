exec(open('rollf.py').read().split("\nRES=[];BASE=rec(SH)")[0])
eD=SH[0][DAY_I].astype(np.float64)
def sysrun(kind,K,a,b,mode):
    if kind=='K': pf,cal=roll_tr(tpS,K);hpf,hc=tobar(txS,pf),tobar(txS,cal)
    else: pf,cal=roll_day(eD,K);hpf,hc=dbar(pf),dbar(cal)
    g=gate(hpf,hc,a,b)
    if mode=='sert': return runT(C,np.where(g,posS,0),TICK,COMM,CASH)
    tk=g[teS];tg=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(teS[tk],txS[tk],tdS[tk]): tg[e_:x_]=d_
    return runT(C,tg,TICK,COMM,CASH)
EK=['RSIX','MIK','COL','B4H','BFLIP','MID'];ST={k:E[k].astype(np.int64) for k in EK};ST['PL']=pl;VT={k:vtr(C,ST[k]) for k in ST}
def engrun(keys,K,a,b):
    E2=dict(E);pl2=pl
    for k in keys:
        pf,cal=roll_tr(VT[k][1],K);g=gate(tobar(VT[k][0],pf),tobar(VT[k][0],cal),a,b)
        if k=='PL': pl2=np.where(g,pl,0).astype(np.int64)
        else: E2[k]=np.where(g,E[k],0)
    return RUN5(chainY(E2,pl2),ex)
isn=lambda r:r[0][I_SPLIT-1];bIS=isn(SH);bn=len(SH[4]);bU=U(SH,0,I_SPLIT)[1]
cand=[]
for kind,Ks in [('K',[20,40,80,160]),('W',[7,14,30,60])]:
    for K,a,b in itertools.product(Ks,[None,.8,.9,1.,1.1,1.2],[None,0.,.5,1.,2.]):
        if a is None and b is None: continue
        for mode in (['sert'] if kind=='K' else ['sert','islem']):
            ty='PF' if b is None else ('CAL' if a is None else 'PF+CAL');cand.append((ty,f"{'son '+str(K)+' işlem' if kind=='K' else 'son '+str(K)+' gün'} PF≥{a} Cal≥{b} {mode}",lambda kind=kind,K=K,a=a,b=b,mode=mode:sysrun(kind,K,a,b,mode)))
for keys,kn in [(EK+['PL'],'tüm motorlar'),(['MIK','B4H','BFLIP','MID'],'yan motorlar')]:
    for K,a,b in itertools.product([10,20,40],[None,.8,1.,1.2],[None,0.,.5]):
        if a is None and b is None: continue
        cand.append(('MOTOR',f'{kn} son {K} işlem PF≥{a} Cal≥{b}',lambda keys=keys,K=K,a=a,b=b:engrun(keys,K,a,b)))
best={}
for ty,nm,f in cand:
    r=f();act=(len(r[4])<bn) if ty!='MOTOR' else abs(r[4].sum()-SH[4].sum())>1
    if not act: continue
    u=U(r,0,I_SPLIT)[1];ok=isn(r)>=bIS;key=(ok,u)
    if ty not in best or key>best[ty][0]: best[ty]=(key,nm,f)
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def col(r):
    f=M(r);o=M(r,I_SPLIT);ul,up=U(r);e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff()
    return {'Net $':f['net'],'Getiri %':f['net']/5,'İşlem':f['n'],'PF':f['pf'],'Win %':f['win'],'Sharpe':f['sharpe'],'Sortino':f['sortino'],'Calmar':f['calmar'],
            'DD P95':f['p95'],'DD P98':f['p98'],'DD P99':f['p99'],'MaxDD':f['mdd'],'Ulcer':ul,'UPI':up,'IS net (seçim)':isn(r),'OOS net (test)':o['net'],'OOS PF':o['pf'],'OOS Calmar':o['calmar'],'2026 %':y.iloc[-1]/5}
T={'BAZ V11.2':col(SH)};AY={'BAZ V11.2':'filtre yok'}
for ty,lab in [('PF','Kayan PF'),('CAL','Kayan Calmar'),('PF+CAL','PF + Calmar'),('MOTOR','Motor bazlı')]:
    _,nm,f=best[ty];T[lab]=col(f());AY[lab]=nm
D=pd.DataFrame(T);pd.set_option('display.width',250)
print(D.round(2).to_string());print();[print(f'{k}: {v}') for k,v in AY.items()]
