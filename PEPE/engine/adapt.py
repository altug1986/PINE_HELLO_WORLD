exec(open("ahf3.py").read().split("rows=[]")[0])

def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    if len(r[1]): p[r[1][-1]:]=r[3][-1]
    return p
F=np.zeros(N,bool)
exec(open('arge5.py').read().split("@numba.njit\ndef vstate")[1].split("F=np.zeros")[0].join(["@numba.njit\ndef vstate",""]))
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
a30,_=adx_htf(30);_,_,a5=adx5();mA,mB,mM=mom(4),mom(48),mom(24)
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));gB=(a30>25)&(a5>lag(a5,6));gM=(a30>18)&(a5>lag(a5,3))
P=lambda L,S_,h=24:pos(run(C,L,S_,h,TICK,COMM,CASH))
def tr_(h,l,c): pc=np.r_[np.nan,c[:-1]];t=np.fmax(np.fmax(h-l,np.abs(h-pc)),np.abs(l-pc));t[0]=h[0]-l[0];return t
h1,l1,c1,th1,_=htf(60);a1=tr_(h1,l1,c1)/c1*100
h4,l4,c4,th4,_=htf(240);a4=rma(tr_(h4,l4,c4),14)/c4*100
bd=load('1440min') if False else None
dd_=B5.resample('1D').agg(dict(high='max',low='min',close='last')).dropna();thd=dd_.index.as_unit('s').asi8
ad=rma(tr_(dd_.high.values,dd_.low.values,dd_.close.values),14)/dd_.close.values*100
VOL={'h168':htf_map(rma(a1,168),th1,3600),'h720':htf_map(rma(a1,720),th1,3600),'4h14':htf_map(a4,th4,14400),'d14':htf_map(ad,thd,86400)}
for k in VOL: VOL[k]=pd.Series(VOL[k]).ffill().bfill().values
REF={k:np.nanmedian(v[:I_SPLIT]) for k,v in VOL.items()};print('referans (IS medyan):',{k:round(v,3) for k,v in REF.items()})
LIBp={k:LIB[k].astype(np.int64) for k in ['RSIX','MIK','B4H','BFLIP']}
def system(f,ahfk=4.5,trim=1):
    LA,SA=gA&(mA>=3*f),gA&(mA<=-3*f);LB,SB=gB&(mB>=2.5*f),gB&(mB<=-2.5*f)
    pA=P(LA,SA);pV=P(LA|LB,SA|SB);col=np.sign(pA+pV);pm=vstate(C,gM&(mM>=3*f),gM&(mM<=-3*f),0.,0.,0,F,F,12).astype(np.int64)
    t=np.zeros(N,np.int64)
    for comp in [pm,LIBp['BFLIP'],LIBp['B4H'],col,LIBp['MIK'],LIBp['RSIX']]: t=np.where(comp!=0,comp,t)
    return run_ahf2(C,t,R7,ahfk*ATRP/100.,WEAK,0,.03,144,trim,90.,60.,12,.03,TICK,COMM,CASH)
def H(r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
    eq=pd.Series(r[0][DAY_I],index=DAY_T);pk=eq.cummax();uw=eq<pk-1e-9;runs=(uw!=uw.shift()).cumsum();lu=uw.groupby(runs).sum().max()
    y=eq.resample('YE').last().diff();y.iloc[0]=eq.resample('YE').last().iloc[0]
    l12=eq[eq.index>=eq.index[-1]-pd.Timedelta(days=365)];r2_12=np.corrcoef(np.arange(len(l12)),l12.values)[0,1]**2
    l3=eq[eq.index>=eq.index[-1]-pd.Timedelta(days=91)]
    return dict(net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],mdd=f['mdd'],q=f['net']/f['mdd'],sharpe=f['sharpe'],uw=100*uw.mean(),luw=int(lu),y23=y.iloc[0]/5,y24=y.iloc[1]/5,y25=y.iloc[2]/5,y26=y.iloc[3]/5,l12=l12.iloc[-1]-l12.iloc[0],r2_12=r2_12,l3m=l3.iloc[-1]-l3.iloc[0],Inet=i['net'],Iq=i['net']/i['mdd'],Ipf=i['pf'],Onet=o['net'],Opf=o['pf'],Ocal=o['calmar'])
rows=[]
r=system(np.ones(N));rows.append(dict(vol='-',al=0.,**H(r)))
for vk,al in itertools.product(VOL.keys(),[0.25,0.5,0.75,1.0]):
    f=(VOL[vk]/REF[vk])**al;rows.append(dict(vol=vk,al=al,**H(system(f))))
D=pd.DataFrame(rows);D.to_pickle('adapt.pkl');pd.set_option('display.width',300)
print(D.round(2).to_string(index=False))
