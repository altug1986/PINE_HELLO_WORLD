exec(open('grand3.py').read().split("\nM_=lambda")[0])
A4=dict(A,po=4);SH5=runc(A4);P5=np.zeros(N,np.int64)
for e_,x_,d_ in zip(SH5[1],SH5[2],SH5[3]): P5[e_:x_]=d_
@numba.njit
def cbrk(c,ps,L,X,meas,Db,join,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    st=0;sl=0.;pk=0.;pu=-1;paused=False;wait=False;last=0;trips=0
    for i in range(n):
        want=ps[i]
        if i<=pu:
            want=0
        else:
            if paused:
                paused=False;st=0;sl=0.;pk=real;wait=(join==0);last=ps[i]
            m=sl if meas==0 else pk-real
            if st>=L and m>=X:
                want=0;pu=i+Db;paused=True;trips+=1
            elif wait:
                if ps[i]!=last: wait=False
                else: want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;kk+=1
                if pnl<0:
                    st+=1;sl-=pnl
                else:
                    st=0;sl=0.
                if real>pk: pk=real
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],trips
r0=cbrk(C,P5,999,1e9,0,0,1,TICK,COMM,CASH);print('regresyon',round(r0[4].sum(),2),'V11.5',round(SH5[4].sum(),2))
M_=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def row(r,**k):
    f=M_(r);ul,up=U(r);_,ui=U(r,0,I_SPLIT);_,uo=U(r,I_SPLIT,N);tx,tp=r[2],r[4]
    return dict(**k,net=f['net'],pf=f['pf'],n=f['n'],sharpe=f['sharpe'],calmar=f['calmar'],p95=f['p95'],p98=f['p98'],p99=f['p99'],mdd=f['mdd'],ulcer=ul,upi=up,Inet=r[0][I_SPLIT-1],Onet=r[0][-1]-r[0][I_SPLIT-1],Iupi=ui,Oupi=uo,trips=r[6] if len(r)==7 and np.isscalar(r[6]) else 0)
ISd=DAY_T<SPLIT;mo=(DAY_T.year-2023)*12+DAY_T.month-5;BLK=np.minimum(mo//6,6)
def U2(p):
    e=np.cumsum(p);dd=np.maximum.accumulate(np.maximum(e,0))-e;ul=np.sqrt((dd**2).mean());return e[-1],ul,(e[-1]/(len(p)/365.25))/ul,dd.max()
B=row(SH5);rows=[];ED={}
for L,X,ms,D,J in itertools.product([1,2,3,4,5,6,8],[0,25,50,75,100,125,150],[0,1],[1,2,3,5],[0,1]):
    if L==1 and X==0: continue
    r=cbrk(C,P5,L,float(X),ms,D*288,J,TICK,COMM,CASH);rows.append(row(r,L=L,X=X,ms=ms,D=D,J=J));ED[len(rows)-1]=r[0][DAY_I].astype(np.float32)
R=pd.DataFrame(rows);print('koşu',len(R),'süre',round(time.time()-t0),'s')
for k in ['net','pf','upi','Iupi','Oupi','Inet','Onet','p98','p99','mdd','calmar']: R['d'+k]=R[k]-B[k]
a=(R.dnet>=0)&(R.dmdd<0);b2=a&(R.dp99<0);c2=(R.dIupi>0)&(R.dOupi>0)&(R.dInet>=0)&(R.dOnet>=0)
print(f"net≥baz & MaxDD↓: {a.sum()} | +P99↓: {b2.sum()} | iki yarıda UPI↑&net↑: {c2.sum()} | medyan Δnet {R.dnet.median():.0f} | medyan ΔMaxDD {R.dmdd.median():.0f}")
s1=R[R.dInet>=0].sort_values('Iupi',ascending=False).iloc[0];s2=R[R.dOnet>=0].sort_values('Oupi',ascending=False).iloc[0]
f_=lambda s:f"L{s.L} X{s.X} {'seri zararı' if s.ms==0 else 'zirveden düşüş'} {s.D}g {'hemen' if s.J else 'yeni sinyal'}"
print(f"IS seç→OOS: {f_(s1)} | IS Δnet {s1.dInet:+.0f} → OOS Δnet {s1.dOnet:+.0f} ΔMaxDD {s1.dmdd:+.0f}")
print(f"OOS seç→IS: {f_(s2)} | OOS Δnet {s2.dOnet:+.0f} → IS Δnet {s2.dInet:+.0f}")
bp=np.diff(np.r_[0.,SH5[0][DAY_I]]);tot=0;win=0;ix=list(R.index)
for bb in range(7):
    m_=BLK!=bb;bo=U2(bp[m_]);sc=[]
    for j in ix: p=np.diff(np.r_[0.,ED[j].astype(np.float64)]);u=U2(p[m_]);sc.append(u[2] if u[0]>=bo[0] else -1e9)
    j=ix[int(np.argmax(sc))];p=np.diff(np.r_[0.,ED[j].astype(np.float64)]);dn=p[BLK==bb].sum()-bp[BLK==bb].sum();tot+=dn;win+=dn>0
print(f'blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
pd.set_option('display.width',320)
for ms in [0,1]:
    q=R[(R.D==3)&(R.ms==ms)&(R.J==1)].copy();q['v']=q.dnet.round(0).astype(int).astype(str)+'/'+q.dmdd.round(0).astype(int).astype(str)
    print(f"\n3 GÜN, {'seri zararı' if ms==0 else 'zirveden düşüş'}, hemen katıl | Δnet/ΔMaxDD (satır art arda zarar L, sütun zarar X$)");print(q.pivot(index='L',columns='X',values='v').to_string())
C_=['L','X','ms','D','J','trips','net','dnet','pf','dInet','dOnet','p95','p98','p99','mdd','upi','calmar']
print('\nnet≥baz & MaxDD↓ & P99↓ (Calmar sırası):');print(R[b2].sort_values('calmar',ascending=False).head(8)[C_].round(2).to_string(index=False))
print(f"\nBAZ V11.5: net {B['net']:.0f} PF {B['pf']:.3f} DD {B['p95']:.0f}/{B['p98']:.0f}/{B['p99']:.0f} MaxDD {B['mdd']:.0f} UPI {B['upi']:.1f} Calmar {B['calmar']:.2f} IS {B['Inet']:.0f} OOS {B['Onet']:.0f}")
R.to_pickle('cbrk.pkl')
