exec(open("ahf3.py").read().split("rows=[]")[0])
V10=run_ahf2(C,t8,R7,4.5*ATRP/100.,WEAK,0,.03,144,1,90.,60.,12,.03,TICK,COMM,CASH)
sp=np.zeros(N,np.int64)
for e,x,d in zip(V10[1],V10[2],V10[3]): sp[e:x]=d
eq=V10[0];ddb=np.maximum.accumulate(np.maximum(eq,0))-eq
ddd=ddb[DAY_I];dayof=np.searchsorted(DAY_I,np.arange(N))      # bar -> gün indeksi
coreEvt=np.r_[False,t8[1:]!=t8[:-1]]&(t8!=0)
@numba.njit
def run_gate(c,sp,dd,thr,blk,cross,exc,srcarr,cev,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;bu=-1;nb=0;fresh=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    for i in range(n):
        if thr[i]>0 and dd[i]>thr[i] and i>bu:
            if cross==0 or (i>0 and dd[i-1]<=thr[i]):
                bu=i+blk;nb+=1;fresh=0
        blocked=i<=bu
        want=sp[i]
        if blocked:
            if exc>=2 and cev[i]: fresh=sp[i]
            if exc>=1 and srcarr[i]==0 and sp[i]!=0: want=sp[i]
            elif exc>=2 and fresh!=0 and sp[i]==fresh: want=sp[i]
            else: want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],nb
def H2(r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
    e=pd.Series(r[0][DAY_I],index=DAY_T);pk=e.cummax();uw=e<pk-1e-9;runs=(uw!=uw.shift()).cumsum();lu=uw.groupby(runs).sum().max()
    y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0]
    l12=e[e.index>=e.index[-1]-pd.Timedelta(days=365)];r12=np.corrcoef(np.arange(len(l12)),l12.values)[0,1]**2
    dd=(pk-e);mo=e.resample('ME').last().diff()
    return dict(net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],p95=f['p95'],p98=f['p98'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],sharpe=f['sharpe'],uw=100*uw.mean(),luw=int(lu),
                y26=y.iloc[3]/5,l12=l12.iloc[-1]-l12.iloc[0],r2_12=r12,wm=mo.min(),Iq=i['net']/i['mdd'],Inet=i['net'],Onet=o['net'],Opf=o['pf'],Ocal=o['calmar'])
rows=[dict(L=0,pq=0,blk=0,cross=0,exc=0,blocks=0,**H2(V10))]
for L,qq,blk,cross,exc in itertools.product([30,60,90,180],[95,98,99],[42,84,168],[1,0],[0,1,2]):
    rp=pd.Series(ddd).rolling(L,min_periods=L).quantile(qq/100).shift(1).values      # sadece önceki günler
    thr=np.nan_to_num(rp,nan=0.)[dayof]
    r=run_gate(C,sp,ddb,thr,blk,cross,exc,src,coreEvt,TICK,COMM,CASH)
    rows.append(dict(L=L,pq=qq,blk=blk,cross=cross,exc=exc,blocks=r[6],**H2(r)))
D=pd.DataFrame(rows);D.to_pickle('gate.pkl');pd.set_option('display.width',300)
print(D.iloc[:1].round(2).to_string(index=False))
print('7 saat (84 bar), P98 — tüm pencereler/tetik/muafiyet:');print(D[(D.blk==84)&(D.pq==98)].round(2).to_string(index=False))
print('genel: q>V10 sayısı',(D.q>D.q.iloc[0]).sum(),'/',len(D)-1,'| net>V10',(D.net>D.net.iloc[0]).sum())
