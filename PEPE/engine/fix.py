exec(open('diag.py').read().split("V61=pickle.load")[0])
_,DI240=adx_htf(240);_,_,a5=adx5();t6=np.where(pR!=0,pR,t4).astype(np.int64)
@numba.njit
def run_x(c,t,prot,regm,act,gb,kl,cool,tick,comm,cash):
    # t: target; prot: RSI-uç aktif (cooldown'dan muaf); regm: kâr koruma rejimi
    n=c.size;pos=0;lock=0;ep=0.;q=0.;mf=0.;real=0.;eq=np.empty(n);k=0;ei=0;cl=0;cu=-1
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    for i in range(n):
        d=t[i]
        if lock!=0 and d!=lock: lock=0
        eff=d if lock==0 else 0
        if i<cu and not prot[i] and eff!=pos: eff=pos if pos!=0 and eff!=0 and eff==pos else (0 if pos==0 else pos)
        if pos!=0 and eff==pos and act>0 and regm[i]:
            r=(c[i]/ep-1)*pos
            if r>mf: mf=r
            if mf>=act and mf-r>=gb: lock=pos;eff=0
        elif pos!=0 and eff==pos:
            r=(c[i]/ep-1)*pos
            if r>mf: mf=r
        if eff!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
                if pnl<=0: cl+=1
                else: cl=0
                if kl>0 and cl>=kl: cu=i+cool;cl=0
            if eff!=0 and not (i<cu and not prot[i]):
                px=c[i]+tick if eff>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i;mf=0.;pos=eff
            else: pos=0
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[k]=ei;tx[k]=n-1;td[k]=pos;tp[k]=(px-ep)*q*pos-fee;tc[k]=fee;k+=1
    return eq,te[:k],tx[:k],td[:k],tp[:k],tc[:k]
W1=(np.searchsorted(B5.index,pd.Timestamp('2026-10-20',tz='UTC')),N);W2=tuple(np.searchsorted(B5.index,[pd.Timestamp('2024-07-29',tz='UTC'),pd.Timestamp('2024-08-28',tz='UTC')]))
rows=[];RES={}
def rec(h,prm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N);dd=tvdd(r)
    w1=metrics(r[0],*r[2:6],*W1)['net'];w2=metrics(r[0],*r[2:6],*W2)['net']
    m=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff();ym=m.groupby(m.index.year).median()
    rows.append(dict(h=h,**prm,net=f['net'],pf=f['pf'],n=f['n'],ddtv=dd,q=f['net']/dd,W1=w1,W2=w2,Ical=i['calmar'],Ocal=o['calmar'],medM=m.median(),m26=ym.iloc[-1],posm=f['posm']));RES[(h,)+tuple(prm.values())]=r
Z=np.zeros(N,bool);prot=pR!=0
rec('BAZ',{},run_x(C,t6,prot,Z,0.,0.,0,0,TICK,COMM,CASH))
# H3 orta rejim motoru (V4 & RSI boşken)
for n,m,lo,h in itertools.product([6,12,24,48],[1.5,2,3],[0,15,18,20],[12,24]):
    g=(a30>lo)&(a5>lag(a5,3));mo=mom(n);pm=vstate(C,g&(mo>=m),g&(mo<=-m),0.,0.,0,F,F,h)
    rec('H3',dict(n_=n,m=m,lo=lo,hd=h),run_x(C,np.where(t6!=0,t6,pm).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH))
# H1 kayıp kümesi soğuma
for kl,cool in itertools.product([2,3,4],[12,24,48,96,192]): rec('H1',dict(kl=kl,cool=cool),run_x(C,t6,prot,Z,0.,0.,kl,cool,TICK,COMM,CASH))
# H2 ara rejimde kâr koruma
for X,act,gb in itertools.product([20,25,30],[.02,.03,.05,.08],[.01,.02,.03]): rec('H2',dict(X=X,act=act,gb=gb),run_x(C,t6,prot,A240<X,act,gb,0,0,TICK,COMM,CASH))
# H4 ara rejimde 4h yön hizası
for X in [20,25,30]:
    mid=A240<X;t=np.where(mid&(t4>0)&(DI240<0),0,np.where(mid&(t4<0)&(DI240>0),0,t4));rec('H4',dict(X=X),run_x(C,np.where(pR!=0,pR,t).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH))
D=pd.DataFrame(rows);D.to_pickle('fix.pkl');pickle.dump({k:v for k,v in RES.items() if k[0] in('BAZ','H3')},open('/home/claude/data/fixRES.pkl','wb'));pd.set_option('display.width',260)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
for h,g in D.groupby('h'):
    if h=='BAZ':continue
    s=g[(g.W1>b.W1)&(g.W2>b.W2)&(g.q>=b.q*0.97)&(g.Ocal>0)];print(h,'iki pencere+kalite korunan:',len(s),'/',len(g))
    print(s.sort_values('q',ascending=False).head(5).dropna(axis=1,how='all').round(2).to_string(index=False))
