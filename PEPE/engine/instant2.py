exec(open('instant.py').read().split("@numba.njit\ndef sim")[0])
@numba.njit
def sim2(o,h,l,c,si,ei,dr,ep,act,cb,sl):
    n=si.size;xp=np.full(n,np.nan);xi=np.full(n,-1);why=np.zeros(n,np.int64)
    for k in range(n):
        d=dr[k];e=ep[k];armed=False;pk=e;done=False
        for i in range(si[k],ei[k]):
            hf=(h[i]-o[i])<(o[i]-l[i])                 # açılış tepeye yakın → O-H-L-C
            for st in range(2):
                usehi=(st==0)==hf                       # bu adımda tepe mi dip mi
                if d>0:
                    if usehi:
                        if h[i]>pk: pk=h[i]
                        if (not armed) and pk>=e*(1+act): armed=True
                    else:
                        if armed and l[i]<=pk*(1-cb): xp[k]=min(o[i],pk*(1-cb)) if st==0 else pk*(1-cb);why[k]=1;done=True;break
                        if sl>0 and l[i]<=e*(1-sl): xp[k]=min(o[i],e*(1-sl)) if st==0 else e*(1-sl);why[k]=2;done=True;break
                else:
                    if not usehi:
                        if l[i]<pk: pk=l[i]
                        if (not armed) and pk<=e*(1-act): armed=True
                    else:
                        if armed and h[i]>=pk*(1+cb): xp[k]=max(o[i],pk*(1+cb)) if st==0 else pk*(1+cb);why[k]=1;done=True;break
                        if sl>0 and h[i]>=e*(1+sl): xp[k]=max(o[i],e*(1+sl)) if st==0 else e*(1+sl);why[k]=2;done=True;break
            if not done and armed:                      # kapanış: ekstremden kapanışa dönüşte stop
                if d>0 and c[i]<=pk*(1-cb): xp[k]=pk*(1-cb);why[k]=1;done=True
                elif d<0 and c[i]>=pk*(1+cb): xp[k]=pk*(1+cb);why[k]=1;done=True
            if done: xi[k]=i;break
    return xp,xi,why
@numba.njit
def ibdd(o,h,l,c,si,ei,xi,dr,ep,q,pnl,n1):            # 1m çubuk içi MTM (en kötü fiyat) MaxDD
    eq=0.;pk=0.;mdd=0.;k=0;nt=si.size
    for k in range(nt):
        end=xi[k]+1 if xi[k]>=0 else ei[k]
        for i in range(si[k],end):
            w=l[i] if dr[k]>0 else h[i];u=eq+(w-ep[k])*q[k]*dr[k]
            if pk-u>mdd: mdd=pk-u
            cl=eq+(c[i]-ep[k])*q[k]*dr[k]
            if cl>pk: pk=cl
        eq+=pnl[k]
        if eq>pk: pk=eq
    return mdd
dr=TR.dir.values.astype(np.int64);EP=TR.ep.values;Q=TR.q.values
SPLIT=pd.Timestamp('2025-05-20')
def run2(act,cb,sl):
    xp,xi,why=sim2(O,H,L,C,SI,EI,dr,EP,act,cb,sl)
    p=TR.pnl.values.copy();ins=why>0;p[ins]=(xp[ins]-EP[ins])*Q[ins]*dr[ins]-(EP[ins]+xp[ins])*Q[ins]*0.0004
    xt=TR.xt.values.copy();xt[ins]=pd.to_datetime(MT[xi[ins]]+60,unit='s')+pd.Timedelta(hours=3)
    return p,why,pd.to_datetime(xt),xi
def M(p,why,xt,xi,full=False):
    s=pd.Series(p,index=xt).sort_index();dd_=s.groupby(s.index.floor('D')).sum();eq=dd_.reindex(pd.date_range(dd_.index.min(),dd_.index.max(),freq='D'),fill_value=0).cumsum();dd=eq.cummax()-eq
    yrs=len(eq)/365.25;ul=np.sqrt((dd**2).mean());p98=np.percentile(dd,98);t=np.arange(len(eq));r2=np.corrcoef(t,eq.values)[0,1]**2
    mo=s.groupby(s.index.to_period('M')).sum();cv=mo.std()/mo.mean();bob=(eq.iloc[-1]/yrs/p98)*r2**2*(mo>0).mean()/(1+0.25*min(cv,8))
    r=dict(net=p.sum(),pf=p[p>0].sum()/-p[p<=0].sum(),nTP=int((why==1).sum()),nSL=int((why==2).sum()),p95=np.percentile(dd,95),p98=p98,p99=np.percentile(dd,99),mdd=dd.max(),ulcer=ul,upi=eq.iloc[-1]/yrs/ul,bob=bob,
           Inet=s[s.index<SPLIT].sum(),Onet=s[s.index>=SPLIT].sum(),l12=s[s.index>=s.index.max()-pd.Timedelta(days=365)].sum())
    if full: r['ib']=ibdd(O,H,L,C,SI,EI,xi,dr,EP,Q,p,len(TR));r['yil']='/'.join(map(str,(s.groupby(s.index.year).sum()/5).round(0).astype(int).tolist()))
    return r
B=M(TR.pnl.values,np.zeros(len(TR),np.int64),TR.xt,np.full(len(TR),-1),True);rows=[]
for a,c_,s_ in itertools.product([.02,.03,.04,.05,.065,.08,.10,.12,.15],[.001,.0015,.0025,.005,.0075,.01,.015,.02,.03],[0.,.08,.10,.12,.15,.20,.25]):
    rows.append(dict(act=a*100,cb=c_*100,sl=s_*100,**M(*run2(a,c_,s_))))
R=pd.DataFrame(rows);R.to_pickle('instant2.pkl');print('koşu',len(R),'süre',round(time.time()-t0),'s')
pd.set_option('display.width',300);K=['act','cb','sl','nTP','nSL','net','pf','p95','p98','p99','mdd','upi','bob','Inet','Onet','l12']
print('BAZ TV V11.6:',{k:round(B[k],2) for k in ['net','pf','p95','p98','p99','mdd','ib','upi','bob','Inet','Onet','l12']},B['yil'])
cur=R[(R.act==6.5)&(R.cb==.15)&(R.sl==15)];print('\nSENİN AYARIN (6.5 / 0.15 / SL15), OHLC sıra tahmini, kaymasız:');print(cur[K].round(2).to_string(index=False))
print(f"\nBoB skoru bazdan iyi: {int((R.bob>B['bob']).sum())}/{len(R)} | net bazdan iyi: {int((R.net>B['net']).sum())} | UPI bazdan iyi: {int((R.upi>B['upi']).sum())}")
for nmin in [100,200,300,500]:
    g=R[R.nTP>=nmin]
    if len(g): print(f"\n≥{nmin} anlık kapanış — BoB en iyi 3:");print(g.sort_values('bob',ascending=False).head(3)[K].round(2).to_string(index=False))
