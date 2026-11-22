exec(open('fix.py').read().split("rows=[];RES={}")[0])
BF=pickle.load(open('/home/claude/data/btcF.pkl','rb'))
P=lambda L,S_,h=24:pos(run(C,L,S_,h,TICK,COMM,CASH))
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));gB=(a30>25)&(a5>lag(a5,6));gM=(a30>18)&(a5>lag(a5,3))
mA,mB,mM=mom(4),mom(48),mom(24)
WS=[('2026-10-20','2026-11-20'),('2024-07-29','2024-08-28'),('2024-03-12','2024-04-11'),('2025-04-04','2025-05-04'),('2025-11-25','2025-12-25'),('2026-05-05','2026-06-04')]
WI=[tuple(np.searchsorted(B5.index,[pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC')])) for a,b in WS]
rows=[]
def build(fA=1.,fB=1.,fM=1.,vetoA=None,vetoB=None,vetoM=None,gMx=None,shock=None,prio='fill'):
    ones=np.ones(N,bool)
    LA,SA=gA&(mA>=3*fA),gA&(mA<=-3*fA);LB,SB=gB&(mB>=2.5*fB),gB&(mB<=-2.5*fB)
    gm=gM if gMx is None else gM&gMx;LM,SM=gm&(mM>=3*fM),gm&(mM<=-3*fM)
    if vetoA is not None: LA&=vetoA[0];SA&=vetoA[1]
    if vetoB is not None: LB&=vetoB[0];SB&=vetoB[1]
    if vetoM is not None: LM&=vetoM[0];SM&=vetoM[1]
    t=np.sign(P(LA,SA)+P(LA|LB,SA|SB));pm=vstate(C,LM,SM,0.,0.,0,F,F,12)
    if shock is None: tg=np.where(pR!=0,pR,np.where(t!=0,t,pm))
    elif prio=='top': tg=np.where(pR!=0,pR,np.where(shock!=0,shock,np.where(t!=0,t,pm)))
    else: tg=np.where(pR!=0,pR,np.where(t!=0,t,np.where(shock!=0,shock,pm)))
    return run_x(C,tg.astype(np.int64),pR!=0,np.zeros(N,bool),0.,0.,0,0,TICK,COMM,CASH)
def rec(h,prm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N);dd=tvdd(r)
    ws=[metrics(r[0],*r[2:6],int(a),int(b))['net'] for a,b in WI]
    m=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff()
    rows.append(dict(h=h,**prm,net=f['net'],pf=f['pf'],n=f['n'],ddtv=dd,p98=f['p98'],q=f['net']/dd,W1=ws[0],W2=ws[1],Wsum=sum(ws),Ical=i['calmar'],Ocal=o['calmar'],Inet=i['net'],Onet=o['net'],medM=m.median(),posm=f['posm']))
rec('BAZ62',{},build())
rk=np.nan_to_num(BF['bRank'],nan=0.5)
# H-A rejime göre eşik çarpanı
for klo,khi,cut in itertools.product([0.7,0.8,0.9,1.0],[1.0,1.1,1.25,1.5],[0.25,0.33]):
    if klo==1 and khi==1: continue
    f=np.where(rk<cut,klo,np.where(rk>1-cut,khi,1.0));rec('HA',dict(klo=klo,khi=khi,cut=cut),build(f,f,f))
# H-A2 sürekli ölçek (BTC ATR% / medyan)^a
med=np.nanmedian(BF['bATRp'])
for a in [-0.5,-0.25,0.25,0.5]:
    f=np.nan_to_num((BF['bATRp']/med)**a,nan=1.0);rec('HA2',dict(a=a),build(f,f,f))
# H-B BTC ATR şoku
bC,bA=BF['bC'],BF['bATR5']
for n,Z,Hh,pr in itertools.product([1,3,6,12],[2.5,3,4,5],[12,24,48],['top','fill']):
    z=np.nan_to_num((bC-lag(bC,n))/(bA*np.sqrt(n)),nan=0.)
    sh=vstate(C,z>=Z,z<=-Z,0.,0.,Hh,F,F,0);rec('HB',dict(n_=n,Z=Z,H=Hh,pr=pr),build(shock=sh,prio=pr))
# H-C BTC karşı yönde güçlü ise veto (ATR birimi)
for n,Zc,tgt in itertools.product([3,6,12],[0.5,1.0,2.0],['ALL','M','AB']):
    z=np.nan_to_num((bC-lag(bC,n))/(bA*np.sqrt(n)),nan=0.);v=(z>-Zc,z<Zc)
    rec('HC',dict(n_=n,Zc=Zc,tgt=tgt),build(vetoA=v if tgt in('ALL','AB') else None,vetoB=v if tgt in('ALL','AB') else None,vetoM=v if tgt in('ALL','M') else None))
# H-D göreli ATR kapısı (M motoru)
ratio=BF['pATRp']/BF['bATRp']
for qq in [20,35,50,65]:
    th_=np.nanpercentile(ratio,qq);rec('HD',dict(rq=qq),build(gMx=np.nan_to_num(ratio,nan=99)>=th_))
D=pd.DataFrame(rows);D.to_pickle('bscan.pkl');pd.set_option('display.width',270)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
for h,g in D.groupby('h'):
    if h=='BAZ62':continue
    s=g[(g.q>=b.q)&(g.Wsum>b.Wsum)&(g.Ocal>=b.Ocal*0.95)&(g.Ical>=b.Ical*0.95)]
    print(h,'kalite↑ + sıkıntılı↑ + IS/OOS korunur:',len(s),'/',len(g),'| net>9500:',(g.net>9500).sum())
    print(s.sort_values('net',ascending=False).head(5).dropna(axis=1,how='all').round(2).to_string(index=False))
