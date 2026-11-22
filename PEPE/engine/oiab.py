exec(open('shsl.py').read().split("\nT={'V11.7 (baz)'")[0])
def runo(s=0,liq=1,j4=1,j2=1):
    d6,o6,(z6,_)=dpn(6),pchg(s,6),bfe(s,6);d12,o12,(z12,b12)=dpn(12),pchg(s,12),bfe(s,12)
    E={'RSIX':LP['RSIX'],'MIK':Z0,'COL':colf(1.,1.,24),'B4H':LP['B4H'],'BFLIP':Z0,'MID':midf(4.,12,3)}
    pl=vst2(C,(d6<=-4)&(o6<=-4),(d6>=4)&(o6<=-4),.08,.08,72,R7>55,R7<45) if liq else Z0
    ex=np.where((d6>=3)&(o6<=-3)&(z6<=-1.5),1,np.where((d6<=-3)&(o6<=-3)&(z6<=-1.5),-1,0)).astype(np.int64) if j4 else Z0
    jj=vst2(C,(d12>=4)&(o12>=4)&(z12>=1)&(b12>=.5),(d12<=-4)&(o12>=4)&(z12>=1)&(b12<=-.5),.03,.03,48,Fb,Fb) if j2 else None
    th=4.5*ATRP/100.;sh_=run16(C,chainY(E,pl,jj,'COL' if j2 else None),R7,th,th,ps,2.,ex,.02,.02,144,90.,60.,12,.03,0.,TICK,COMM,CASH)
    p=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(sh_[1],sh_[2],sh_[3]): p[e_:x_]=d_
    r=cbrk(C,p,2,100.,0,288,0,TICK,COMM,CASH);tp=r[4];e30=r[0][5::6];dd=np.maximum.accumulate(np.maximum(e30,0))-e30
    return tp.sum(),tp[tp>0].sum()/-tp[tp<=0].sum(),len(tp),np.percentile(dd,95),np.percentile(dd,99),dd.max(),r[0][I_SPLIT-1],r[0][-1]-r[0][I_SPLIT-1]
V=[('V11.7 tam (tüm OI açık)',dict()),('OI tamamen kapalı',dict(liq=0,j4=0,j2=0)),('sadece PEPE OI likidasyon dönüşü',dict(j4=0,j2=0)),('sadece BTC+PEPE ortak kurallar',dict(liq=0))]
for s in [0,1,2]:
    print(f'--- OI {s} mum geç ---')
    for nm,kw in V: n_,pf,k,p95,p99,mx,i1,i2=runo(s,**kw);print(f'{nm:34s} net {n_:7.0f} | PF {pf:.3f} | işlem {k} | 30dk MTM DD P95/P99/max {p95:.0f}/{p99:.0f}/{mx:.0f} | ilk/ikinci yarı {i1:.0f}/{i2:.0f}')
