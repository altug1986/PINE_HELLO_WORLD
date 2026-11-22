exec(open('oi.py').read().split("\nrows=[dict(h='V10.1'")[0])
FL=pickle.load(open('/home/claude/data/flowF.pkl','rb'))
def z(x,w=288): s=pd.Series(np.log(x));return np.nan_to_num(((s-s.rolling(w).mean())/s.rolling(w).std()).values)
TZ12=pd.Series(z(FL['TAKER'])).rolling(12).mean().fillna(0).values;RZ=z(FL['RETAIL']);TPd=chg(FL['TOPPOS'],12);OId12=chg(OI,12)
dp6,do6=chg(C,6),chg(OI,6)
EX=np.where((dp6>=3)&(do6<=-2),1,np.where((dp6<=-3)&(do6<=-2),-1,0)).astype(np.int64)
PL=vstate(C,(dp6<=-10)&(do6<=-5),(dp6>=10)&(do6<=-5),0.,.08,72,R7>60,R7<40,0).astype(np.int64)
@numba.njit
def run_v13(c,t,R,thr,thr1,efl,ps,sT,ex,slc,mc,X,Y,M,z,tick,comm,cash):
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
                if (thr[i]>0 and adv>=thr[i]) or (thr1[i]>0 and adv>=thr1[i] and efl[i]==-pos):
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif ex[i]==pos:
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=6
                elif ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)) and (sT<0 or trs<sT):
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
def chainE(E):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]: t=np.where(E[k]!=0,E[k],t)
    return np.where(PL!=0,PL,t).astype(np.int64)
ZF=np.zeros(N);ZI=np.zeros(N,np.int64)
def R13(t=None,thr1=ZF,efl=ZI,ex=EX):
    return run_v13(C,chainE(ENG) if t is None else t,R7,thr,thr1,efl,ps,2.0,ex,.03,144,90.,60.,12,.03,TICK,COMM,CASH)
def U(r,a=0,b=N):
    di=DAY_I[(DAY_I>=a)&(DAY_I<b)];e=r[0][di]-(r[0][a-1] if a>0 else 0);dd=np.maximum.accumulate(np.maximum(e,0))-e;Ul=np.sqrt((dd**2).mean());yrs=(b-a)/(288*365.25)
    return Ul,(e[-1]/yrs)/Ul
def HH(r):
    h=H(r);u,upi=U(r);ui,upii=U(r,0,I_SPLIT);uo,upio=U(r,I_SPLIT,N);h.update(ulcer=u,upi=upi,IS_upi=upii,OOS_upi=upio);return h
rows=[dict(h='V11',p='',**HH(R13()))]
# E1 erken karşı poz (akış aleyhte)
for k1,o_,zt in itertools.product([1.5,2,3],[0.5,1,2],[0.25,0.5,1.0]):
    efl=np.where((OId12>=o_)&(TZ12<=-zt),-1,np.where((OId12>=o_)&(TZ12>=zt),1,0)).astype(np.int64)
    rows.append(dict(h='E1 erken karşı',p=f'k{k1} OI≥{o_} taker|z|≥{zt}',**HH(R13(thr1=k1*ATRP/100.,efl=efl))))
# E2 taker tükenişi (H1'e ek, VEYA)
for P,zt in itertools.product([2,3,5],[0.25,0.5,1.0]):
    ex2=np.where(EX!=0,EX,np.where((dp6>=P)&(TZ12<=-zt),1,np.where((dp6<=-P)&(TZ12>=zt),-1,0))).astype(np.int64)
    rows.append(dict(h='E2 taker tükeniş',p=f'fiyat±{P} taker z∓{zt}',**HH(R13(ex=ex2))))
# E3 retail kalabalık vetosu (yan motorlar)
for r_,tg in itertools.product([1.0,1.5,2.0],['MID+B4H','SIDE']):
    E=dict(ENG);ks=['MID','B4H'] if tg=='MID+B4H' else ['MIK','B4H','BFLIP','MID']
    for k in ks: E[k]=np.where(((ENG[k]>0)&(RZ>=r_))|((ENG[k]<0)&(RZ<=-r_)),0,ENG[k])
    rows.append(dict(h='E3 retail veto',p=f'{tg} |z|≥{r_}',**HH(R13(t=chainE(E)))))
# E4 top trader ayrışma vetosu
for x_,tg in itertools.product([1,2,4],['MID+B4H','SIDE']):
    E=dict(ENG);ks=['MID','B4H'] if tg=='MID+B4H' else ['MIK','B4H','BFLIP','MID']
    for k in ks: E[k]=np.where(((ENG[k]>0)&(TPd>=x_))|((ENG[k]<0)&(TPd<=-x_)),0,ENG[k])
    rows.append(dict(h='E4 top trader veto',p=f'{tg} Δ≥{x_}%',**HH(R13(t=chainE(E)))))
D=pd.DataFrame(rows);D.to_pickle('flow1.pkl');pd.set_option('display.width',300)
b=D.iloc[0];print(D.iloc[:1][['net','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26','r2_12']].round(2).to_string(index=False))
for h,g in D.groupby('h',sort=False):
    if h=='V11':continue
    s=g[(g.upi>b.upi)&(g.net>=b.net*0.99)&(g.OOS_upi>=b.OOS_upi*0.95)&(g.IS_upi>=b.IS_upi*0.95)]
    print(f'{h}: geçen {len(s)}/{len(g)} | UPI medyan {g.upi.median():.1f} | net medyan {g.net.median():.0f}')
    if len(s): print(s.sort_values('upi',ascending=False).head(5)[['p','net','pct','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26','r2_12']].round(2).to_string(index=False))
