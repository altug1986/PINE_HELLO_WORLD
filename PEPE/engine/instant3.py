exec(open('instant2.py').read().split("\nB=M(")[0])
B=M(TR.pnl.values,np.zeros(len(TR),np.int64),TR.xt,np.full(len(TR),-1),True);R=pd.read_pickle('instant2.pkl')
R['c95']=R.net/R.p95;R['c98']=R.net/R.p98;b95=B['net']/B['p95'];b98=B['net']/B['p98']
pd.set_option('display.width',300);K=['act','cb','sl','nTP','nSL','net','pf','p95','p98','p99','mdd','upi','c95','c98','bob','Inet','Onet']
g=R[R.nTP>=300]
print(f"BAZ: net/P95 {b95:.1f} net/P98 {b98:.1f} UPI {B['upi']:.1f} BoB {B['bob']:.2f}")
for col in ['c95','upi','bob']: print(f'\n≥300 anlık kapanış, en iyi {col}:');print(g.sort_values(col,ascending=False).head(4)[K].round(2).to_string(index=False))
def half(cfg):
    p,why,xt,xi=run2(*cfg) if cfg else (TR.pnl.values,None,TR.xt,None);s=pd.Series(p,index=pd.to_datetime(xt)).sort_index();out=[]
    for a,b in [(None,SPLIT),(SPLIT,None)]:
        ss=s[(s.index>=a) if a is not None else (s.index<b)];dd_=ss.groupby(ss.index.floor('D')).sum();eq=dd_.reindex(pd.date_range(dd_.index.min(),dd_.index.max(),freq='D'),fill_value=0).cumsum();dd=eq.cummax()-eq
        out.append((ss.sum(),np.percentile(dd,95),eq.iloc[-1]/(len(eq)/365.25)/np.sqrt((dd**2).mean())))
    return out
hb=half(None);print(f"\nYARILAR (net / DD P95 / UPI) — BAZ: ilk {hb[0][0]:.0f}/{hb[0][1]:.0f}/{hb[0][2]:.1f} | ikinci {hb[1][0]:.0f}/{hb[1][1]:.0f}/{hb[1][2]:.1f}")
for cfg in [(.065,.0015,.15),(.05,.001,0.),(.065,.001,0.),(.08,.001,0.),(.10,.001,0.),(.065,.0015,0.)]:
    h=half(cfg);print(f"  akt {cfg[0]*100:.1f} geri {cfg[1]*100:.2f} SL {cfg[2]*100:.0f}: ilk {h[0][0]:.0f}/{h[0][1]:.0f}/{h[0][2]:.1f} | ikinci {h[1][0]:.0f}/{h[1][1]:.0f}/{h[1][2]:.1f}")
best=g.sort_values('c95',ascending=False).iloc[0];nb=R[(abs(R.act-best.act)<=2.1)&(abs(np.log(R.cb/best.cb))<=0.75)&((R.sl==best.sl)|(R.sl.isin([0,25])))]
print(f"\nKOMŞU (en iyi net/P95 = akt {best.act} geri {best.cb} SL {best.sl}): {len(nb)} ayar | net/P95 bazdan iyi {int((nb.c95>b95).sum())} | UPI bazdan iyi {int((nb.upi>B['upi']).sum())} | net medyan {nb.net.median():.0f}")
