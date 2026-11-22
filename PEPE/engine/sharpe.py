exec(open("adapt.py").read().split("\nrows=[]\n")[0])
d_=np.diff(C,prepend=C[0]);ret=np.r_[0.,C[1:]/C[:-1]-1]
pAo=P(gA&(mA>=3),gA&(mA<=-3));pVo=P((gA&(mA>=3))|(gB&(mB>=2.5)),(gA&(mA<=-3))|(gB&(mB<=-2.5)));COL=np.sign(pAo+pVo)
PM=vstate(C,gM&(mM>=3),gM&(mM<=-3),0.,0.,0,F,F,12).astype(np.int64)
ENG={'RSIX':LIBp['RSIX'],'MIK':LIBp['MIK'],'COL':COL,'B4H':LIBp['B4H'],'BFLIP':LIBp['BFLIP'],'MID':PM}
ORDER=['RSIX','MIK','COL','B4H','BFLIP','MID'];SIDE=['MIK','B4H','BFLIP','MID']
dayof=np.searchsorted(DAY_I,np.arange(N))
def eng_sharpe(p,D):
    pr=np.r_[0,p[:-1]]*ret-np.abs(np.diff(np.r_[0,p]))*0.0005
    ds=pd.Series(pr).groupby(dayof).sum()
    sh=(ds.rolling(D,min_periods=D).mean()/ds.rolling(D,min_periods=D).std()*np.sqrt(365)).shift(1)
    return np.nan_to_num(sh.values,nan=1.0)[dayof]          # geçmiş yoksa izinli
def chain(E):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]: t=np.where(E[k]!=0,E[k],t)
    return t
def runv(t): return run_ahf2(C,t,R7,4.5*ATRP/100.,WEAK,0,.03,144,1,90.,60.,12,.03,TICK,COMM,CASH)
rows=[dict(var='V10',**H(runv(chain(ENG))))]
# (A) motor bazında kendi Sharpe'ı > s
for D,s,tgt in itertools.product([20,30,60,90],[0.,0.5,1.0],['SIDE','MID+B4H']):
    E=dict(ENG);ks=SIDE if tgt=='SIDE' else ['MID','B4H']
    for k in ks: E[k]=np.where(eng_sharpe(ENG[k],D)>s,ENG[k],0)
    rows.append(dict(var=f'A: {tgt} kendi Sharpe {D}g > {s}',**H(runv(chain(E)))))
# (B) fiyat Sharpe yön uyumu
for W,s,tgt in itertools.product([288,864,2016,4032],[0.,0.5,1.0],['SIDE','MID+B4H']):
    m=pd.Series(ret).rolling(W).mean();sd=pd.Series(ret).rolling(W).std();ps=np.nan_to_num((m/sd*np.sqrt(W)).values,nan=0.)
    E=dict(ENG);ks=SIDE if tgt=='SIDE' else ['MID','B4H']
    for k in ks: E[k]=np.where(((ENG[k]>0)&(ps>s))|((ENG[k]<0)&(ps<-s)),ENG[k],0)
    rows.append(dict(var=f'B: {tgt} fiyat Sharpe W{W} ±{s}',**H(runv(chain(E)))))
D=pd.DataFrame(rows);D.to_pickle('sharpe.pkl');pd.set_option('display.width',300)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
g=D[(D.net>b.net)&(D.q>=b.q*0.97)&(D.Ocal>=b.Ocal*0.95)]
print('net↑ + kalite korunur + OOS korunur:',len(g),'/',len(D)-1);print(g.sort_values('net',ascending=False).round(2).to_string(index=False))
print('A ailesi net medyan',round(D[D['var'].str.startswith('A')].net.median()),'| B ailesi net medyan',round(D[D['var'].str.startswith('B')].net.median()))
