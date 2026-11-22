exec(open('vhf.py').read().split("\nCF=[]")[0])
do,ex=comps(0);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64);tb=chainY(E0,pl)
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b);L12=DAY_I[-365]
def H2(r):
    f=M(r);o=M(r,I_SPLIT);ul,up=U(r);return dict(net=f['net'],n=f['n'],pf=f['pf'],sharpe=f['sharpe'],calmar=f['calmar'],p95=f['p95'],p98=f['p98'],p99=f['p99'],mdd=f['mdd'],ulcer=ul,upi=up,Inet=r[0][I_SPLIT-1],Onet=o['net'],l12=r[0][-1]-r[0][L12])
b=H2(R6(tb,ex));rows=[]
for src,q,aq in itertools.product(['5m24','5m48','5m96','1h14'],[.05,.1,.15],[.85,.9,.95]):
    v=VH[src];m=(v<QI(v,q))&(ATRP>=QI(ATRP,aq));h=H2(R6(np.where(m&(pl==0),0,tb).astype(np.int64),ex))
    rows.append(dict(src=src,q=q,aq=aq,flat_pct=100*m.mean(),dnet=h['net']-b['net'],dupi=h['upi']-b['upi'],dp99=h['p99']-b['p99'],dmdd=h['mdd']-b['mdd'],dOnet=h['Onet']-b['Onet']))
R=pd.DataFrame(rows);ok=(R.dnet>0)&(R.dupi>0)
print(f'komşu ızgara {len(R)} | net↑&UPI↑ {ok.sum()} | Δnet medyan {R.dnet.median():.0f}')
print(R.pivot_table(index=['src','q'],columns='aq',values='dnet').round(0).to_string())
v=VH['5m48'];m=(v<QI(v,.1))&(ATRP>=QI(ATRP,.9));w=H2(R6(np.where(m&(pl==0),0,tb).astype(np.int64),ex))
print('\nAday 5m48 q0.1 ATR p0.9: flat süresi %',round(100*m.mean(),2));print(pd.DataFrame({'BAZ V11.2':b,'VHF+vol flat':w}).round(2).to_string())
