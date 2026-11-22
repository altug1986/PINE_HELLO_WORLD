exec(open("adapt.py").read().split("\nrows=[]\n")[0])
d_=np.diff(C,prepend=np.nan);R7r=np.nan_to_num(100-100/(1+rma(np.where(d_>0,d_,0.),7)/rma(np.where(d_<0,-d_,0.),7)),nan=50.)
pAo=P(gA&(mA>=3),gA&(mA<=-3));pVo=P((gA&(mA>=3))|(gB&(mB>=2.5)),(gA&(mA<=-3))|(gB&(mB<=-2.5)));COL=np.sign(pAo+pVo)
PM=vstate(C,gM&(mM>=3),gM&(mM<=-3),0.,0.,0,F,F,12).astype(np.int64)
V=VOL['h720'];qs={q:np.nanpercentile(V[:I_SPLIT],q) for q in [20,35,50]}
def sys2(mid,b4h):
    t=np.zeros(N,np.int64)
    for comp in [mid,LIBp['BFLIP'],b4h,COL,LIBp['MIK'],LIBp['RSIX']]: t=np.where(comp!=0,comp,t)
    return run_ahf2(C,t,R7,4.5*ATRP/100.,WEAK,0,.03,144,1,90.,60.,12,.03,TICK,COMM,CASH)
rows=[dict(var='V10',**H(sys2(PM,LIBp['B4H'])))]
for q,v in qs.items():
    low=V<v
    rows.append(dict(var=f'düşükvol<P{q}: ORTA→flat',**H(sys2(np.where(low,0,PM),LIBp['B4H']))))
    rows.append(dict(var=f'düşükvol<P{q}: ORTA+B4H→flat',**H(sys2(np.where(low,0,PM),np.where(low,0,LIBp['B4H'])))))
    for lo,sl in [(20,.03),(15,.03),(25,.02)]:
        mr=vstate(C,low&(R7r<lo),low&(R7r>100-lo),0.,sl,48,R7r>50,R7r<50,0).astype(np.int64)
        rows.append(dict(var=f'düşükvol<P{q}: ORTA→MR rsi{lo} sl{sl}',**H(sys2(np.where(low,mr,PM),LIBp['B4H']))))
D=pd.DataFrame(rows);pd.set_option('display.width',300);print(D.round(2).to_string(index=False))
