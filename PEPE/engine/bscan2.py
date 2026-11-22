exec(open("bscan.py").read().split("\nrows=[]\n")[0])
exec("def rec"+open('bscan.py').read().split("def rec")[1].split("rec('BAZ62'")[0])
b=pd.read_csv('/home/claude/data/BTC_1m.csv.gz');b.index=pd.to_datetime(b.time,unit='ms',utc=True);END=b.index[-1]
agg=dict(open='first',high='max',low='min',close='last')
BT={}
for tf in [30,60,240,1440]:
    x=b.resample(f'{tf}min').agg(agg).dropna();h,l,c=x.high.values.astype(float),x.low.values.astype(float),x.close.values.astype(float);th=x.index.as_unit('s').asi8
    p_,m_,a_=dmi(h,l,c,14,14);pc=np.r_[np.nan,c[:-1]];t_=np.fmax(np.fmax(h-l,np.abs(h-pc)),np.abs(l-pc));t_[0]=h[0]-l[0];atp=rma(t_,14)/c*100
    rk=pd.Series(atp).rolling({30:1440,60:720,240:180,1440:60}[tf]).rank(pct=True).values
    ok=B5.index<=END
    BT[tf]={k:np.where(ok,htf_map(v,th,tf*60),np.nan) for k,v in dict(adx=a_,di=p_-m_,rk=rk).items()}
rows=[]
def build3(gate=None,dirok=None,dtgt='ALL',fill=None,fA=1.,over=None):
    ones=np.ones(N,bool);g=ones if gate is None else gate
    LA,SA=g&gA&(mA>=3*fA),g&gA&(mA<=-3*fA);LB,SB=g&gB&(mB>=2.5*fA),g&gB&(mB<=-2.5*fA);LM,SM=g&gM&(mM>=3*fA),g&gM&(mM<=-3*fA)
    if dirok is not None:
        lo,so=dirok
        if dtgt in('ALL','AB'): LA&=lo;SA&=so;LB&=lo;SB&=so
        if dtgt in('ALL','M'): LM&=lo;SM&=so
    t=np.sign(P(LA,SA)+P(LA|LB,SA|SB));pm=vstate(C,LM,SM,0.,0.,0,F,F,12) if fill is None else fill
    tg=np.where(pR!=0,pR,np.where(t!=0,t,pm))
    if over is not None: tg=np.where(pR!=0,pR,np.where(over!=0,over,np.where(t!=0,t,pm)))
    return run_x(C,tg.astype(np.int64),pR!=0,np.zeros(N,bool),0.,0.,0,0,TICK,COMM,CASH)
rec('BAZ62',{},build3())
nz=lambda x,v:np.nan_to_num(x,nan=v)
for tf in [60,240,1440]:
    A_,D_,R_=nz(BT[tf]['adx'],99),nz(BT[tf]['di'],0),nz(BT[tf]['rk'],.5)
    for X in [15,20,25]: rec('G1-gate',dict(tf=tf,X=X),build3(gate=A_>X))
    for X,tg in itertools.product([0,20,25],['ALL','M','AB']):
        weak=A_<X if X else np.zeros(N,bool);rec('G2-dir',dict(tf=tf,X=X,tgt=tg),build3(dirok=(weak|(D_>=0),weak|(D_<=0)),dtgt=tg))
    for X in [20,25,30]:
        rec('G3-fill',dict(tf=tf,X=X),build3(fill=np.where(A_>X,np.sign(D_),0).astype(np.int64)))
        rec('G3-fillM',dict(tf=tf,X=X),build3(fill=np.where(A_>X,np.sign(D_),vstate(C,gM&(mM>=3),gM&(mM<=-3),0.,0.,0,F,F,12))))
    for klo,khi in [(0.8,1.2),(0.9,1.1),(1.2,0.8),(0.8,1.0),(1.0,1.2)]:
        rec('G4-scale',dict(tf=tf,klo=klo,khi=khi),build3(fA=np.where(R_<.33,klo,np.where(R_>.67,khi,1.))))
    for X,H_ in itertools.product([25,30,35],[48,144]):
        cross=(A_>X)&(lag(A_,1)<=X)
        ov=vstate(C,cross&(D_>0),cross&(D_<0),0.,0.,H_,F,F,0);rec('G5-flip',dict(tf=tf,X=X,H=H_),build3(over=ov))
D=pd.DataFrame(rows);D.to_pickle('bscan2.pkl');pd.set_option('display.width',270)
b0=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
print('net>9500 toplam:',(D.net>9500).sum(),'| en yüksek net:',round(D.net.max()))
for h,g in D.groupby('h'):
    if h=='BAZ62':continue
    s=g[(g.q>=b0.q)&(g.Wsum>b0.Wsum)&(g.Ocal>=b0.Ocal*0.95)&(g.Ical>=b0.Ical*0.95)]
    print(h,'geçen:',len(s),'/',len(g),'| en yüksek net',round(g.net.max()))
    if len(s): print(s.sort_values('net',ascending=False).head(4).dropna(axis=1,how='all').round(2).to_string(index=False))
