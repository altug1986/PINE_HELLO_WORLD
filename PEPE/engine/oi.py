exec(open('sharpe2.py').read().split("\nrows=[dict(var='V10',W=0")[0])
o=pd.read_csv('/mnt/user-data/uploads/1000PEPEUSDT_OI_BINANCE_USDM_5m_2023-05-05_2026-11-19.csv',usecols=['timestamp_utc','open_interest_qty','sum_taker_long_short_vol_ratio','sum_toptrader_long_short_ratio'])
o.index=pd.to_datetime(o.timestamp_utc,utc=True);o=o.drop(columns='timestamp_utc').replace(0,np.nan)
o=o[~o.index.duplicated()].reindex(B5.index,method='ffill')            # bar açılışında bilinen son değer
OI=o.open_interest_qty.ffill().values
print('OI kapsamı',np.isfinite(OI).mean().round(4))
def chg(x,n): r=np.full(N,np.nan);r[n:]=(x[n:]/x[:-n]-1)*100;return np.nan_to_num(r)
@numba.njit
def run_v12(c,t,R,thr,ps,sT,ex,useex,cok,usec,sk,uses,slc,mc,X,Y,M,z,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce:
            mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos);trs=ps[i]*pos
                if thr[i]>0 and adv>=thr[i] and (usec==0 or cok[i]==-pos):
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif useex==1 and ex[i]==pos:
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=6
                elif ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)) and (sT<0 or trs<sT) and (uses==0 or sk[i]!=pos):
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=1
        elif mode==1:
            g=(c[i]/cxp-1.0)*(-cdir)
            if g<=-slc or i-cxb>=mc:
                mode=0;want=t[i];tag=5
            else:
                want=-cdir
        else:
            if md>0: back=(R[i]<=Y) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Y) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back:
                mode=0;want=t[i];tag=2
            else:
                want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=tag;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
Z=np.zeros(N,np.int64);ps=PS[864]
def RUN(t=t10,ex=Z,useex=0,cok=Z,usec=0,sk=Z,uses=0):
    return run_v12(C,t,R7,thr,ps,2.0,ex,useex,cok,usec,sk,uses,.03,144,90.,60.,12,.03,TICK,COMM,CASH)
rows=[dict(h='V10.1',p='',**H(RUN()))]
for n in [6,12,24,48]:
    dp,do=chg(C,n),chg(OI,n)
    # H1 tükeniş: fiyat pozisyon yönünde >= P%, OI <= -O%  → kâr kilitle
    for P,O_ in itertools.product([3,5,8],[1,2,4]):
        ex=np.where((dp>=P)&(do<=-O_),1,np.where((dp<=-P)&(do<=-O_),-1,0)).astype(np.int64)
        rows.append(dict(h='H1 tükeniş',p=f'n{n} fiyat±{P} OI-{O_}',**H(RUN(ex=ex,useex=1))))
    # H2 karşı poz teyidi: karşı yönde fiyat ve OI artışı
    for O_ in [0.,1,2]:
        cok=np.where(do>O_,np.sign(dp),0).astype(np.int64);rows.append(dict(h='H2 karşı teyit',p=f'n{n} OI>{O_}',**H(RUN(cok=cok,usec=1))))
    # H3 sağlıklı trend: pozisyon yönünde fiyat ve OI birlikte ↑ → mikro kâr alma yok
    for O_ in [0.,1,2]:
        sk=np.where(do>O_,np.sign(dp),0).astype(np.int64);rows.append(dict(h='H3 trim atla',p=f'n{n} OI>{O_}',**H(RUN(sk=sk,uses=1))))
# H4 likidasyon dönüş motoru (en üst öncelik, sanal, RSI14 uç gibi)
for n,P,O_,hold in itertools.product([6,12,24],[4,6,10],[3,5,8],[24,72]):
    dp,do=chg(C,n),chg(OI,n)
    L=(dp<=-P)&(do<=-O_);S=(dp>=P)&(do<=-O_)
    pl=vstate(C,L,S,0.,.08,hold,R7>60,R7<40,0).astype(np.int64)
    rows.append(dict(h='H4 likidasyon',p=f'n{n} fiyat∓{P} OI-{O_} h{hold}',**H(RUN(t=np.where(pl!=0,pl,t10).astype(np.int64)))))
D=pd.DataFrame(rows);D.to_pickle('oi1.pkl');pd.set_option('display.width',300)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
for h,g in D.groupby('h',sort=False):
    if h=='V10.1':continue
    s=g[(g.net>b.net)&(g.q>=b.q*0.98)&(g.Ocal>=b.Ocal*0.95)]
    print(f'{h}: geçen {len(s)}/{len(g)} | net medyan {g.net.median():.0f} | en iyi net {g.net.max():.0f}')
    print(s.sort_values('net',ascending=False).head(4)[['p','net','pf','n','mdd','q','sharpe','y26','l12','r2_12','Iq','Onet','Opf','Ocal']].round(2).to_string(index=False))
