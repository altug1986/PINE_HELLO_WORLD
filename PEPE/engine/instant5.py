exec(open('instant2.py').read().split("\nB=M(")[0])
src=open('instant2.py').read().split("@numba.njit\ndef sim2")[1].split("SPLIT=pd.Timestamp")[0]
src=src.replace("(o,h,l,c,si,ei,dr,ep,act,cb,sl):","(o,h,l,c,si,ei,dr,ep,act,cb,sll,sls):").replace("d=dr[k];e=ep[k];armed=False;pk=e;done=False","d=dr[k];e=ep[k];armed=False;pk=e;done=False;sl=sll if d>0 else sls")
exec("@numba.njit\ndef sim3"+src.split("@numba.njit\ndef ibdd")[0])
def run3(act,cb,sll,sls):
    xp,xi,why=sim3(O,H,L,C,SI,EI,dr,EP,act,cb,sll,sls)
    p=TR.pnl.values.copy();ins=why>0;p[ins]=(xp[ins]-EP[ins])*Q[ins]*dr[ins]-(EP[ins]+xp[ins])*Q[ins]*0.0004
    xt=TR.xt.values.copy();xt[ins]=pd.to_datetime(MT[xi[ins]]+60,unit='s')+pd.Timedelta(hours=3);return p,why,pd.to_datetime(xt),xi
def col(p,why,xt,xi):
    m=M(p,why,xt,xi,True);s=pd.Series(p,index=xt).sort_index();mo=s.groupby(s.index.to_period('M')).sum();sh=dr<0
    return {'Net $':m['net'],'PF':m['pf'],'Short net $':p[sh].sum(),'Short PF':p[sh&(p>0)].sum()/-p[sh&(p<=0)].sum(),'SL kapanışı long / short':f"{int(((why==2)&~sh).sum())} / {int(((why==2)&sh).sum())}",
            'SL ile kapanan shortların neti $':p[(why==2)&sh].sum(),'Anlık kâr al':int((why==1).sum()),'DD P95 / P98 / P99':f"{m['p95']:.0f} / {m['p98']:.0f} / {m['p99']:.0f}",'MaxDD gün sonu / 1dk mum içi':f"{m['mdd']:.0f} / {m['ib']:.0f}",
            'UPI':m['upi'],'İlk yarı / ikinci yarı net':f"{m['Inet']:.0f} / {m['Onet']:.0f}",'Pozitif ay %':100*(mo>0).mean(),'Yıl %':m['yil']}
T={'TV V11.6 (WT yok)':col(TR.pnl.values,np.zeros(len(TR),np.int64),TR.xt,np.full(len(TR),-1)),'WT şu an: SL long %15 / short %15':col(*run3(.065,.0015,.15,.15))}
for ss in [.135,.12,.10]: T[f'short SL %{ss*100:g}']=col(*run3(.065,.0015,.15,ss))
pd.set_option('display.width',300);print(pd.DataFrame(T).to_string())
sh=dr<0;p0,w0,_,_=run3(.065,.0015,.15,.15);p1,w1,_,_=run3(.065,.0015,.15,.12);ch=(w1==2)&sh&(w0!=2)
print(f"\n%12'de yeni stoplanan short: {int(ch.sum())} | şu anki ayarda bu işlemlerin neti {p0[ch].sum():.0f}$ → %12 ile {p1[ch].sum():.0f}$ | bunlardan şu an zararla bitenler {int((p0[ch]<0).sum())}, kârla bitenler {int((p0[ch]>0).sum())}")
