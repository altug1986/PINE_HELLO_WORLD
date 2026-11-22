exec(open('q7.py').read().split("K=['net','pf'")[0])
exec("@numba.njit\ndef run_pyr"+open('arge6.py').read().split("@numba.njit\ndef run_pyr")[1].split("gA=(a30>28)")[0])
_,DI240=adx_htf(240);_,DI60=adx_htf(60)
mrA=vstate(C,(A60<15)&(R7<20),(A60<15)&(R7>80),0.,.05,48,R7>50,R7<50,0)
mrB=vstate(C,(CH240>61.8)&(R7<20),(CH240>61.8)&(R7>80),0.,.05,48,R7>50,R7<50,0)
K=['net','pf','n','calmar','p98','mdd','sharpe','sortino','r2','posm','cv']
rows=[];i6=np.searchsorted(B5.index,pd.Timestamp('2026-05-19',tz='UTC'))
def rec(prm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N);l6=metrics(r[0],*r[2:6],i6,N)
    m=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff();ym=m.groupby(m.index.year).median()
    rows.append(dict(**prm,**{k:f[k] for k in K},nDD=f['net']/max(f['mdd'],1),Ical=i['calmar'],Ocal=o['calmar'],Onet=o['net'],l6=l6['net'],medM=m.median(),Ymed='/'.join(f'{v:.0f}' for v in ym.values),avgU=(r[6]*np.arange(len(r[6]))).sum()/N if len(r)>6 else np.nan))
for mrn,mr in [('none',np.zeros(N,np.int64)),('A1h15',mrA),('C4h62',mrB)]:
    for tf_,tth in [('none',0),('4h',25),('4h',30),('1h',30)]:
        A_,D_=(A240,DI240) if tf_=='4h' else (A60,DI60)
        tfill=np.where((A_>tth)&(D_>0),1,np.where((A_>tth)&(D_<0),-1,0)) if tf_!='none' else np.zeros(N,np.int64)
        base=np.where(pR!=0,pR,np.where(t4!=0,t4,np.where(mr!=0,mr,tfill))).astype(np.int64)
        for gth,bn,Kk,gap in itertools.product([0,25,30],[48,96],[1,2,3],[48,96]):
            if Kk==1 and (gth or bn!=48 or gap!=48): continue
            hh,ll=donch(bn);g=(A240>gth) if gth else np.ones(N,bool)
            aL=g&(C>hh)&((DI240>0) if gth else True);aS=g&(C<ll)&((DI240<0) if gth else True)
            rec(dict(mr=mrn,tf=f'{tf_}{tth}',gth=gth,bn=bn,K=Kk,gap=gap),run_pyr(C,base,aL,aS,Kk,gap,0,TICK,COMM,CASH))
D=pd.DataFrame(rows);D.to_pickle('q8.pkl');pd.set_option('display.width',280)
b=D[(D.mr=='none')&(D.tf=='none0')&(D.K==1)];print(b.round(2).to_string(index=False))
g=D[(D.Ical>0)&(D.Onet>0)]
print('net>=12000 & nDD>=15:');print(g[(g.net>=12000)].sort_values('nDD',ascending=False).head(10).round(2).to_string(index=False))
print('en iyi nDD (K>=2):');print(g[g.K>=2].sort_values('nDD',ascending=False).head(8).round(2).to_string(index=False))
