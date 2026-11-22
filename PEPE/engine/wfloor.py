exec(open('ptgt.py').read().split("\nLK=[")[0])
b=row(SH,T=0,F=0,m=-1,J=-1);b['trips']=0;rows=[b];ED={};W=PID['hafta']
for T_,F_,m,J in itertools.product([6,8,10,12,15,20,25,30],[2,3,4,5,6,8,10,12,15,20],[0,1],[0,1]):
    if F_>=T_: continue
    r=ptgt(C,posS,W,T_*5.,1,F_/T_,m,J,TICK,COMM,CASH);rows.append(row(r,T=T_,F=F_,m=m,J=J));ED[len(rows)-1]=r[0][DAY_I]
R=pd.DataFrame(rows);B=R.iloc[0];R=R.iloc[1:].copy();print('koşu',len(R))
for k in ['net','upi','Iupi','Oupi','Inet','Onet','p98','p99','mdd','l12']: R['d'+k]=R[k]-B[k]
w=R[(R.dupi>0)&(R.dnet>=0)];w2=R[(R.dIupi>0)&(R.dOupi>0)&(R.dInet>=0)&(R.dOnet>=0)]
print(f'tüm dönem UPI↑&net↑ {len(w)} | iki yarıda UPI↑&net↑ {len(w2)} | medyan Δnet {R.dnet.median():.0f}')
c1=R[R.dInet>=0];c1=c1 if len(c1) else R;s1=c1.loc[c1.Iupi.idxmax()];c2=R[R.dOnet>=0];c2=c2 if len(c2) else R;s2=c2.loc[c2.Oupi.idxmax()]
print(f"IS seç→OOS: hedef%{s1['T']} taban%{s1.F} m{s1.m} J{s1.J} | IS Δnet {s1.dInet:+.0f} → OOS Δnet {s1.dOnet:+.0f} ΔUPI {s1.dOupi:+.2f}")
print(f"OOS seç→IS: hedef%{s2['T']} taban%{s2.F} m{s2.m} J{s2.J} | OOS Δnet {s2.dOnet:+.0f} → IS Δnet {s2.dInet:+.0f} ΔUPI {s2.dIupi:+.2f}")
bp=np.diff(np.r_[0.,SH[0][DAY_I]]);tot=0;win=0;ix=list(R.index)
for bb in range(7):
    m_=BLK!=bb;bo=U2(bp[m_]);sc=[]
    for j in ix: p=np.diff(np.r_[0.,ED[j]]);u=U2(p[m_]);sc.append(u[2] if u[0]>=bo[0] else -1e9)
    j=ix[int(np.argmax(sc))];p=np.diff(np.r_[0.,ED[j]]);dn=p[BLK==bb].sum()-bp[BLK==bb].sum();tot+=dn;win+=dn>0
print(f'blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
for m in [1,0]:
    q=R[(R.m==m)&(R.J==0)];print(f"\nΔnet (hedef satır, taban sütun) | {'her mumda açık kâr dahil' if m else 'işlem kapanışında'} | pazartesi yeni sinyal")
    print(q.pivot(index='T',columns='F',values='dnet').round(0).to_string())
pd.set_option('display.width',300);C_=['T','F','m','J','trips','net','dnet','dInet','dOnet','p95','p98','p99','mdd','ulcer','upi','calmar','sharpe','pf','n','l12']
print('\nSENİN KURALIN (hedef %10, taban %5):');print(R[(R['T']==10)&(R.F==5)][C_].round(2).to_string(index=False))
print('\nen iyi 5 (iki yarının kötüsü):');print(R.assign(mm=np.minimum(R.dIupi,R.dOupi)).sort_values('mm',ascending=False).head(5)[C_].round(2).to_string(index=False))
print(f"\nBAZ: net {B.net:.0f} n {B.n:.0f} PF {B.pf:.2f} Sharpe {B.sharpe:.2f} Calmar {B.calmar:.2f} DD {B.p95:.0f}/{B.p98:.0f}/{B.p99:.0f} MaxDD {B.mdd:.0f} Ulcer {B.ulcer:.1f} UPI {B.upi:.1f} IS {B.Inet:.0f} OOS {B.Onet:.0f} 12ay {B.l12:.0f}")
R.to_pickle('wfloor.pkl')
