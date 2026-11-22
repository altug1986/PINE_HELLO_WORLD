exec(open('btc2h.py').read().split("\nC1=[]")[0])
import time;t0=time.time()
@numba.njit
def runT(c,t,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    for i in range(n):
        want=t[i]
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
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk]
@numba.njit
def roll_tr(p,K):
    m=p.size;pf=np.full(m,np.nan);cal=np.full(m,np.nan)
    for j in range(K-1,m):
        gp=0.;gl=0.;cum=0.;pk=0.;dd=0.
        for k in range(j-K+1,j+1):
            x=p[k]
            if x>0: gp+=x
            else: gl-=x
            cum+=x
            if cum>pk: pk=cum
            if pk-cum>dd: dd=pk-cum
        pf[j]=gp/gl if gl>0 else 99.
        cal[j]=cum/dd if dd>0 else 99.
    return pf,cal
@numba.njit
def roll_day(e,W):
    m=e.size;pf=np.full(m,np.nan);cal=np.full(m,np.nan)
    for d in range(W,m):
        gp=0.;gl=0.;pk=e[d-W];dd=0.
        for k in range(d-W+1,d+1):
            x=e[k]-e[k-1]
            if x>0: gp+=x
            else: gl-=x
            if e[k]>pk: pk=e[k]
            if pk-e[k]>dd: dd=pk-e[k]
        pf[d]=gp/gl if gl>0 else 99.;net=e[d]-e[d-W];cal[d]=net/dd if dd>0 else 99.
    return pf,cal
@numba.njit
def vtr(c,s):
    n=c.size;tx=np.empty(n,np.int64);pn=np.empty(n);k=0;cur=0;ep=0.
    for i in range(n):
        if s[i]!=cur:
            if cur!=0:
                tx[k]=i;pn[k]=500.*(c[i]/ep-1.)*cur-0.4;k+=1
            cur=s[i];ep=c[i]
    return tx[:k],pn[:k]
AR=np.arange(N)
def tobar(tx,v):
    idx=np.searchsorted(tx,AR,side='right')-1;out=np.full(N,np.nan);ok=idx>=0;out[ok]=v[idx[ok]];return out
DK=np.searchsorted(DAY_I,AR,side='left')-1
def dbar(v): out=np.full(N,np.nan);ok=DK>=0;out[ok]=v[DK[ok]];return out
def gate(hpf,hc,a,b):
    g=np.ones(N,bool)
    if a is not None: g&=np.isnan(hpf)|(hpf>=a)
    if b is not None: g&=np.isnan(hc)|(hc>=b)
    return g
# --- gölge = V11.2
do,ex=comps(0);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
E=dict(ENG);E['MIK']=MIK2;E['MID']=deb(ENG['MID'].astype(np.int64),2);tb=chainY(E,pl)
SH=RUN5(tb,ex);teS,txS,tdS,tpS=SH[1],SH[2],SH[3],SH[4]
posS=np.zeros(N,np.int64)
for e_,x_,d_ in zip(teS,txS,tdS): posS[e_:x_]=d_
print('regresyon gölge',round(tpS.sum(),2),'hedef-takip',round(runT(C,posS,TICK,COMM,CASH)[4].sum(),2))
ISd=DAY_T<SPLIT;mo=(DAY_T.year-2023)*12+DAY_T.month-5;BLK=np.minimum(mo//6,6)
def rec(r):
    e=r[0][DAY_I];p=np.diff(np.r_[0.,e]);tx,tp=r[2],r[4];pf=lambda m:tp[m&(tp>0)].sum()/-tp[m&(tp<=0)].sum()
    return dict(p=p.astype(np.float64),n=len(tp),pf=pf(np.ones(len(tp),bool)),Ipf=pf(tx<I_SPLIT),Opf=pf(tx>=I_SPLIT))
def U2(p):
    e=np.cumsum(p);dd=np.maximum.accumulate(np.maximum(e,0))-e;ul=np.sqrt((dd**2).mean())
    return e[-1],ul,(e[-1]/(len(p)/365.25))/ul,dd.max()
RES=[];BASE=rec(SH)
def add(fam,cfg,r): RES.append(dict(fam=fam,cfg=cfg,**rec(r)))
TH_A=[None,.8,.9,1.,1.1,1.2];TH_B=[None,0.,.5,1.,2.]
for K in [20,40,80,160]:
    pf,cal=roll_tr(tpS,K);hpf,hc=tobar(txS,pf),tobar(txS,cal)
    for a,b in itertools.product(TH_A,TH_B):
        if a is None and b is None: continue
        g=gate(hpf,hc,a,b)
        add('S1 sistem K-işlem SERT',f'K{K} PF≥{a} Cal≥{b}',runT(C,np.where(g,posS,0),TICK,COMM,CASH))
        tk=g[teS];tg=np.zeros(N,np.int64)
        for e_,x_,d_ in zip(teS[tk],txS[tk],tdS[tk]): tg[e_:x_]=d_
        add('S2 sistem K-işlem İŞLEM-BAZLI',f'K{K} PF≥{a} Cal≥{b}',runT(C,tg,TICK,COMM,CASH))
eD=SH[0][DAY_I].astype(np.float64)
for W in [7,14,30,60]:
    pf,cal=roll_day(eD,W);hpf,hc=dbar(pf),dbar(cal)
    for a,b in itertools.product(TH_A,TH_B):
        if a is None and b is None: continue
        g=gate(hpf,hc,a,b)
        add('T1 sistem W-gün SERT',f'W{W} PF≥{a} Cal≥{b}',runT(C,np.where(g,posS,0),TICK,COMM,CASH))
        tk=g[teS];tg=np.zeros(N,np.int64)
        for e_,x_,d_ in zip(teS[tk],txS[tk],tdS[tk]): tg[e_:x_]=d_
        add('T2 sistem W-gün İŞLEM-BAZLI',f'W{W} PF≥{a} Cal≥{b}',runT(C,tg,TICK,COMM,CASH))
EK=['RSIX','MIK','COL','B4H','BFLIP','MID'];ST={k:E[k].astype(np.int64) for k in EK};ST['PL']=pl;VT={k:vtr(C,ST[k]) for k in ST}
for keys,kn in [(EK+['PL'],'tüm motorlar'),(['MIK','B4H','BFLIP','MID'],'yan motorlar')]:
    for K in [10,20,40]:
        H_={k:(lambda pc:(tobar(VT[k][0],pc[0]),tobar(VT[k][0],pc[1])))(roll_tr(VT[k][1],K)) for k in keys}
        for a,b in itertools.product([None,.8,1.,1.2],[None,0.,.5]):
            if a is None and b is None: continue
            E2=dict(E);pl2=pl
            for k in keys:
                g=gate(H_[k][0],H_[k][1],a,b)
                if k=='PL': pl2=np.where(g,pl,0).astype(np.int64)
                else: E2[k]=np.where(g,E[k],0)
            add('E motor-bazlı',f'{kn} K{K} PF≥{a} Cal≥{b}',RUN5(chainY(E2,pl2),ex))
print('koşu',len(RES),'süre',round(time.time()-t0),'s')
bp=BASE['p'];B_full=U2(bp);B_is=U2(bp[ISd]);B_os=U2(bp[~ISd])
print(f"BAZ V11.2: net {B_full[0]:.0f} UPI {B_full[2]:.1f} Ulcer {B_full[1]:.1f} maxDD {B_full[3]:.0f} PF {BASE['pf']:.3f} n {BASE['n']} | IS net {B_is[0]:.0f} UPI {B_is[2]:.1f} | OOS net {B_os[0]:.0f} UPI {B_os[2]:.1f} PF {BASE['Opf']:.3f}")
R=pd.DataFrame(RES)
for col,m in [('f',None),('i',ISd),('o',~ISd)]:
    v=np.array([U2(p if m is None else p[m]) for p in R.p]);R[col+'net']=v[:,0];R[col+'upi']=v[:,2];R[col+'mdd']=v[:,3]
R['dnet']=R.fnet-B_full[0];R['dupi']=R.fupi-B_full[2];R['dIs']=R.iupi-B_is[2];R['dOs']=R.oupi-B_os[2];R['dIn']=R.inet-B_is[0];R['dOn']=R.onet-B_os[0]
pd.set_option('display.width',320)
for fam,g in R.groupby('fam',sort=False):
    w=g[(g.dupi>0)&(g.dnet>=0)];print(f'\n=== {fam}: {len(g)} ayar | UPI↑ & net↑ olan {len(w)} | medyan Δnet {g.dnet.median():.0f} ΔUPI {g.dupi.median():.2f} | işlem medyan {g.n.median():.0f}')
    c1=g[g.dIn>=0];c1=c1 if len(c1) else g;s1=c1.loc[c1.iupi.idxmax()]
    c2=g[g.dOn>=0];c2=c2 if len(c2) else g;s2=c2.loc[c2.oupi.idxmax()]
    print(f'  ÇAPRAZ-1 IS seç→OOS: {s1.cfg} | IS ΔUPI {s1.dIs:+.2f} Δnet {s1.dIn:+.0f} → OOS ΔUPI {s1.dOs:+.2f} Δnet {s1.dOn:+.0f}')
    print(f'  ÇAPRAZ-2 OOS seç→IS: {s2.cfg} | OOS ΔUPI {s2.dOs:+.2f} Δnet {s2.dOn:+.0f} → IS ΔUPI {s2.dIs:+.2f} Δnet {s2.dIn:+.0f}')
    tot=0;win=0;ch=[]
    for b in range(7):
        m=BLK!=b;bo=U2(bp[m]);sc=[];
        for p in g.p: u=U2(p[m]);sc.append(u[2] if u[0]>=bo[0] else -1e9)
        j=int(np.argmax(sc));
        if sc[j]==-1e9: j=int(np.argmax([U2(p[m])[2] for p in g.p]))
        dn=g.p.iloc[j][BLK==b].sum()-bp[BLK==b].sum();tot+=dn;win+=dn>0;ch.append(g.cfg.iloc[j])
    print(f'  ÇAPRAZ-3 blok dışarıda: dışarıdaki bloklarda toplam Δnet {tot:+.0f} | iyileşen blok {win}/7 | seçimler {pd.Series(ch).value_counts().head(3).to_dict()}')
    g2=g.assign(mn=np.minimum(g.dIs,g.dOs)).sort_values('mn',ascending=False).head(3)
    print(g2[['cfg','n','fnet','fupi','fmdd','pf','dIs','dOs','dIn','dOn']].round(2).to_string(index=False))
R.drop(columns='p').to_pickle('rollf.pkl')
