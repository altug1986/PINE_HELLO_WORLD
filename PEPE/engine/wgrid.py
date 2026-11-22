exec(open('ptgt.py').read().split("\nPID={")[0])
@numba.njit
def ptgt2(c,ps,pid,T,F,mh,mc,join,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    cur=-1;bR=0.;bM=0.;hit=False;locked=False;wait=False;last=0;nl=0;prevE=0.
    for i in range(n):
        e=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
        if pid[i]!=cur:
            cur=pid[i];bR=real;bM=prevE
            if locked:
                wait=(join==0);last=ps[i]
            locked=False;hit=False
        want=ps[i]
        if locked:
            want=0
        else:
            ph=(e-bM) if mh==1 else (real-bR);pc=(e-bM) if mc==1 else (real-bR)
            if (not hit) and ph>=T: hit=True
            if hit and pc<=F:
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
WK={0:((T5//86400+3)//7).astype(np.int64),3:(((T5+10800)//86400+3)//7).astype(np.int64)}
r0=ptgt2(C,posS,WK[0],1e9,0.,0,0,1,TICK,COMM,CASH);print('regresyon',round(r0[4].sum(),2))
pf_=lambda tx,tp,m:tp[m&(tp>0)].sum()/-tp[m&(tp<=0)].sum()
def row2(r,**k):
    h=row(r,**k);tx,tp=r[2],r[4];h.update(Ipf=pf_(tx,tp,tx<I_SPLIT),Opf=pf_(tx,tp,tx>=I_SPLIT));return h
b=row2(SH,T=0,F=0,mh=-1,mc=-1,J=-1,off=-1);b['trips']=0;rows=[b];ED={}
for T_ in range(4,14):
    for F_,mh,mc,J,off in itertools.product(np.arange(0,T_,.5),[0,1],[0,1],[0,1],[0,3]):
        r=ptgt2(C,posS,WK[off],T_*5.,F_*5.,mh,mc,J,TICK,COMM,CASH);rows.append(row2(r,T=T_,F=F_,mh=mh,mc=mc,J=J,off=off));ED[len(rows)-1]=r[0][DAY_I].astype(np.float32)
R=pd.DataFrame(rows);B=R.iloc[0];R=R.iloc[1:].copy();print('koşu',len(R),'süre',round(time.time()-t0),'s')
for k in ['net','pf','upi','Iupi','Oupi','Inet','Onet','Ipf','Opf','p98','p99','mdd','l12','calmar']: R['d'+k]=R[k]-B[k]
ok=(R.dnet>0)&(R.dpf>0);ok2=ok&(R.dInet>0)&(R.dOnet>0)&(R.dIpf>0)&(R.dOpf>0)
print(f'net↑&PF↑: {ok.sum()} | iki yarıda da net↑&PF↑: {ok2.sum()} | medyan Δnet {R.dnet.median():.0f}')
pd.set_option('display.width',330);C_=['T','F','mh','mc','J','off','trips','net','dnet','pf','dInet','dOnet','Opf','p98','p99','mdd','upi','calmar','sharpe','n','l12']
print('\nnet↑&PF↑ olanlar (UPI sırası, ilk 12):');print(R[ok].sort_values('upi',ascending=False).head(12)[C_].round(2).to_string(index=False))
c1=R[(R.dInet>0)&(R.dIpf>0)];
if len(c1):
    s1=c1.loc[c1.Iupi.idxmax()];print(f"\nIS seç (IS net↑&PF↑, en iyi IS UPI)→OOS: T{s1['T']} F{s1.F} mh{s1.mh} mc{s1.mc} J{s1.J} off{s1.off} | IS Δnet {s1.dInet:+.0f} → OOS Δnet {s1.dOnet:+.0f} OOS PF {s1.Opf:.3f} (baz {B.Opf:.3f})")
bp=np.diff(np.r_[0.,SH[0][DAY_I]]);tot=0;win=0;ix=list(R.index)
for bb in range(7):
    m_=BLK!=bb;bo=U2(bp[m_]);sc=[]
    for j in ix: p=np.diff(np.r_[0.,ED[j].astype(np.float64)]);u=U2(p[m_]);sc.append(u[2] if u[0]>=bo[0] else -1e9)
    j=ix[int(np.argmax(sc))];p=np.diff(np.r_[0.,ED[j].astype(np.float64)]);dn=p[BLK==bb].sum()-bp[BLK==bb].sum();tot+=dn;win+=dn>0
print(f'blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
if ok.sum():
    s=R[ok].sort_values('upi',ascending=False).iloc[0]
    nb=R[(R.mh==s.mh)&(R.mc==s.mc)&(R.J==s.J)&(R.off==s.off)&(abs(R['T']-s['T'])<=1)&(abs(R.F-s.F)<=1)]
    print(f"\nKOMŞULAR (T±1, F±1, aynı mod): {len(nb)} ayar | net↑&PF↑ olan {int(((nb.dnet>0)&(nb.dpf>0)).sum())} | Δnet medyan {nb.dnet.median():.0f} min {nb.dnet.min():.0f}")
    print(nb.pivot_table(index='T',columns='F',values='dnet').round(0).to_string())
print(f"\nBAZ: net {B.net:.0f} PF {B.pf:.3f} IS PF {B.Ipf:.3f} OOS PF {B.Opf:.3f} UPI {B.upi:.1f} DD {B.p95:.0f}/{B.p98:.0f}/{B.p99:.0f} MaxDD {B.mdd:.0f}")
R.to_pickle('wgrid.pkl')
