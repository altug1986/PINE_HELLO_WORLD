exec(open('rollf.py').read().split("\nRES=[];BASE=rec(SH)")[0])
dayid=((T5//86400)-(T5[0]//86400)).astype(np.int64);NDy=int(dayid.max())+1
@numba.njit
def brk(c,ps,X,Db,Wd,join,dayid,nd,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    pu=-1;pk=0.;dmax=np.full(nd,-1e18);rday=0;wait=False;last=0;trips=0;paused=False
    for i in range(n):
        e=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.);d=dayid[i];want=ps[i]
        if i<=pu:
            want=0
        else:
            if paused:
                paused=False;pk=e;rday=d;dmax[d]=e;wait=(join==0);last=ps[i]
            if e>pk: pk=e
            if e>dmax[d]: dmax[d]=e
            ref=pk
            if Wd>0:
                ref=-1e18
                for k in range(max(d-Wd+1,rday),d+1):
                    if dmax[k]>ref: ref=dmax[k]
            if ref-e>=X:
                want=0;pu=i+Db;paused=True;trips+=1
            elif wait:
                if ps[i]!=last: wait=False
                else: want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],trips
r0=brk(C,posS,1e9,0,0,1,dayid,NDy,TICK,COMM,CASH);print('regresyon',round(r0[4].sum(),2),'baz',round(tpS.sum(),2))
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
L12=DAY_I[-365];bm=M(SH);bo=M(SH,I_SPLIT);bi=M(SH,0,I_SPLIT);bu=U(SH)
def row(r,**k):
    f=M(r);o=M(r,I_SPLIT);i_=M(r,0,I_SPLIT);ul,up=U(r);_,upi_i=U(r,0,I_SPLIT);_,upi_o=U(r,I_SPLIT,N)
    return dict(**k,net=f['net'],n=f['n'],pf=f['pf'],sharpe=f['sharpe'],calmar=f['calmar'],p95=f['p95'],p98=f['p98'],p99=f['p99'],mdd=f['mdd'],ulcer=ul,upi=up,Inet=i_['net'],Onet=o['net'],Iupi=upi_i,Oupi=upi_o,l12=r[0][-1]-r[0][L12],trips=r[6] if len(r)>6 else 0)
rows=[row(SH,X=0,D=0,W=0,J=-1)]
for X,D,W,J in itertools.product([40,60,80,100,120,150,180,220],[1,2,3,5,7],[0,3,7,14,30],[0,1]):
    rows.append(row(brk(C,posS,float(X),D*288,W,J,dayid,NDy,TICK,COMM,CASH),X=X,D=D,W=W,J=J))
R=pd.DataFrame(rows);B=R.iloc[0];R=R.iloc[1:].copy();R.to_pickle('breaker.pkl');print('koşu',len(R),'süre',round(time.time()-t0),'s')
for k in ['net','upi','Iupi','Oupi','Inet','Onet','p98','p99','mdd','l12']: R['d'+k]=R[k]-B[k]
w=R[(R.dupi>0)&(R.dnet>=0)];w2=R[(R.dIupi>0)&(R.dOupi>0)&(R.dInet>=0)&(R.dOnet>=0)];w3=R[(R.dp99<0)&(R.dmdd<0)]
print(f'tüm dönem UPI↑&net↑ {len(w)} | iki yarıda UPI↑&net↑ {len(w2)} | DD P99↓ ve MaxDD↓ {len(w3)} (bunların net medyanı Δ{w3.dnet.median() if len(w3) else 0:.0f}) | medyan Δnet {R.dnet.median():.0f}')
c1=R[R.dInet>=0];c1=c1 if len(c1) else R;s1=c1.loc[c1.Iupi.idxmax()];c2=R[R.dOnet>=0];c2=c2 if len(c2) else R;s2=c2.loc[c2.Oupi.idxmax()]
print(f'IS seç→OOS: X{s1.X} D{s1.D} W{s1.W} J{s1.J} | IS ΔUPI {s1.dIupi:+.2f} Δnet {s1.dInet:+.0f} → OOS ΔUPI {s1.dOupi:+.2f} Δnet {s1.dOnet:+.0f}')
print(f'OOS seç→IS: X{s2.X} D{s2.D} W{s2.W} J{s2.J} | OOS ΔUPI {s2.dOupi:+.2f} Δnet {s2.dOnet:+.0f} → IS ΔUPI {s2.dIupi:+.2f} Δnet {s2.dInet:+.0f}')
pd.set_option('display.width',320)
q=R[(R.D==3)&(R.W==0)]
print('\n3 GÜN BEKLE, zirve=devamdan beri | J0=yeni sinyal bekle J1=hemen katıl');print(q[['X','J','trips','net','dnet','Onet','dOnet','p95','p98','p99','mdd','upi','l12']].round(1).to_string(index=False))
print('\nen iyi 5 (iki yarının kötüsü):');print(R.assign(m=np.minimum(R.dIupi,R.dOupi)).sort_values('m',ascending=False).head(5)[['X','D','W','J','trips','net','dnet','dInet','dOnet','p98','p99','mdd','upi']].round(1).to_string(index=False))
print(f"\nBAZ: net {B.net:.0f} PF {B.pf:.2f} Sharpe {B.sharpe:.2f} Calmar {B.calmar:.2f} DD P95/98/99 {B.p95:.0f}/{B.p98:.0f}/{B.p99:.0f} MaxDD {B.mdd:.0f} UPI {B.upi:.1f} IS {B.Inet:.0f} OOS {B.Onet:.0f} son12ay {B.l12:.0f}")
