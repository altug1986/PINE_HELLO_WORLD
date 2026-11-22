exec(open('breaker.py').read().split("\nrows=[row(SH")[0])
@numba.njit
def ptgt(c,ps,pid,T,lt,lk,mtm,join,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    cur=-1;base=0.;hit=False;peak=0.;locked=False;wait=False;last=0;nl=0;prevE=0.
    for i in range(n):
        e=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
        if pid[i]!=cur:
            cur=pid[i];base=prevE if mtm==1 else real
            if locked:
                wait=(join==0);last=ps[i]
            locked=False;hit=False;peak=0.
        want=ps[i]
        if locked:
            want=0
        else:
            pn=(e-base) if mtm==1 else (real-base)
            if (not hit) and pn>=T:
                hit=True;peak=pn
                if lt==0:
                    locked=True;nl+=1;want=0
            if hit and not locked:
                if pn>peak: peak=pn
                L=lk*T if lt==1 else (peak-lk*T if lt==2 else lk*peak)
                if pn<L:
                    locked=True;nl+=1;want=0
            if (not locked) and wait:
                if ps[i]!=last: wait=False
                else: want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.);prevE=eq[i]
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],nl
PID={'gün':dayid,'hafta':((T5//86400+3)//7).astype(np.int64),'ay':(B5.index.year*12+B5.index.month).values.astype(np.int64)}
r0=ptgt(C,posS,PID['hafta'],1e9,1,1.,0,1,TICK,COMM,CASH);print('regresyon',round(r0[4].sum(),2))
LK=[(0,0.,'hedefte dur'),(1,1.,'başabaş=hedef'),(1,.75,'kilit %75 hedef'),(1,.5,'kilit %50 hedef'),(2,.5,'iz: zirve-0.5H'),(2,1.,'iz: zirve-1H'),(3,.5,'zirvenin %50'),(3,.7,'zirvenin %70')]
b=row(SH,P='-',T=0,L='-',m=-1,J=-1);b['trips']=0;rows=[b];ED={}
for P,Tp,(lt,lk,ln),m,J in itertools.product(['gün','hafta','ay'],[2,3,5,8,10,15,20,30,40],LK,[0,1],[0,1]):
    r=ptgt(C,posS,PID[P],Tp*5.,lt,lk,m,J,TICK,COMM,CASH);rows.append(row(r,P=P,T=Tp,L=ln,m=m,J=J));ED[len(rows)-1]=r[0][DAY_I]
R=pd.DataFrame(rows);B=R.iloc[0];R=R.iloc[1:].copy();print('koşu',len(R),'süre',round(time.time()-t0),'s')
for k in ['net','upi','Iupi','Oupi','Inet','Onet','p98','p99','mdd','l12','calmar']: R['d'+k]=R[k]-B[k]
w=R[(R.dupi>0)&(R.dnet>=0)];w2=R[(R.dIupi>0)&(R.dOupi>0)&(R.dInet>=0)&(R.dOnet>=0)];w3=R[(R.dp99<0)&(R.dmdd<0)]
print(f'tüm dönem UPI↑&net↑ {len(w)} | iki yarıda UPI↑&net↑ {len(w2)} | P99↓&MaxDD↓ {len(w3)} (net medyan Δ{w3.dnet.median() if len(w3) else 0:.0f}) | medyan Δnet {R.dnet.median():.0f}')
for P in ['gün','hafta','ay']:
    g=R[R.P==P];print(f'  {P}: medyan Δnet {g.dnet.median():.0f} | en iyi Δnet {g.dnet.max():.0f} | UPI↑&net↑ {int(((g.dupi>0)&(g.dnet>=0)).sum())}')
c1=R[R.dInet>=0];c1=c1 if len(c1) else R;s1=c1.loc[c1.Iupi.idxmax()];c2=R[R.dOnet>=0];c2=c2 if len(c2) else R;s2=c2.loc[c2.Oupi.idxmax()]
print(f'IS seç→OOS: {s1.P} %{s1.T} {s1.L} m{s1.m} J{s1.J} | IS ΔUPI {s1.dIupi:+.2f} Δnet {s1.dInet:+.0f} → OOS ΔUPI {s1.dOupi:+.2f} Δnet {s1.dOnet:+.0f}')
print(f'OOS seç→IS: {s2.P} %{s2.T} {s2.L} m{s2.m} J{s2.J} | OOS ΔUPI {s2.dOupi:+.2f} Δnet {s2.dOnet:+.0f} → IS ΔUPI {s2.dIupi:+.2f} Δnet {s2.dInet:+.0f}')
bp=np.diff(np.r_[0.,SH[0][DAY_I]]);tot=0;win=0;ix=list(R.index)
for bb in range(7):
    m_=BLK!=bb;bo=U2(bp[m_]);sc=[]
    for j in ix: p=np.diff(np.r_[0.,ED[j]]);u=U2(p[m_]);sc.append(u[2] if u[0]>=bo[0] else -1e9)
    j=ix[int(np.argmax(sc))];p=np.diff(np.r_[0.,ED[j]]);dn=p[BLK==bb].sum()-bp[BLK==bb].sum();tot+=dn;win+=dn>0
print(f'blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
pd.set_option('display.width',320);C_=['P','T','L','m','J','trips','net','dnet','dInet','dOnet','p98','p99','mdd','upi','calmar','l12']
print('\nen iyi 8 (iki yarının kötüsü):');print(R.assign(mm=np.minimum(R.dIupi,R.dOupi)).sort_values('mm',ascending=False).head(8)[C_].round(1).to_string(index=False))
print('\nSENİN ÖRNEĞİN (başabaş=hedef, kapanışta kontrol, yeni dönemde yeni sinyal):');print(R[(R.L=='başabaş=hedef')&(R.m==0)&(R.J==0)&(R.T.isin([3,5,8,10]))][C_].round(1).to_string(index=False))
print(f"\nBAZ: net {B.net:.0f} PF {B.pf:.2f} Sharpe {B.sharpe:.2f} Calmar {B.calmar:.2f} DD P95/98/99 {B.p95:.0f}/{B.p98:.0f}/{B.p99:.0f} MaxDD {B.mdd:.0f} UPI {B.upi:.1f} IS {B.Inet:.0f} OOS {B.Onet:.0f} son12ay {B.l12:.0f}")
R.to_pickle('ptgt.pkl')
