exec(open('btc2h.py').read().split("\nC1=[]")[0])
@numba.njit
def run_v15(c,t,R,thr,ps,sT,ex,slc,mc,Xa,Y,M,z,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0;X=Xa[i]
        if ce:
            mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos);trs=ps[i]*pos
                if thr[i]>0 and adv>=thr[i]:
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
X90=np.full(N,90.)
def R6(t,ex,Xa=X90,th=None): return run_v15(C,t,R7,thr if th is None else th,ps,2.0,ex,.03,144,Xa,60.,12,.03,TICK,COMM,CASH)
s5=pd.Series(C);ad=s5.diff().abs()
VH={}
for n_ in [24,48,96,144,288]: VH[f'5m{n_}']=((s5.rolling(n_).max()-s5.rolling(n_).min())/ad.rolling(n_).sum()).fillna(0.5).values
B60,T60=load('60min');c60=pd.Series(B60.close.values);a60=c60.diff().abs()
for n_ in [14,28,48]: VH[f'1h{n_}']=np.nan_to_num(htf_map((((c60.rolling(n_).max()-c60.rolling(n_).min())/a60.rolling(n_).sum()).values),T60,3600),nan=0.5)
QI=lambda x,q:float(np.nanquantile(x[:I_SPLIT],q))
AQ={q:QI(ATRP,q) for q in [.7,.8,.9]};dp12=chg(C,12)
E0=dict(ENG);E0['MIK']=MIK2;E0['MID']=deb(ENG['MID'].astype(np.int64),2)
SIDE={'MID':['MID'],'yan3':['MID','B4H','BFLIP'],'yan3+COL':['MID','B4H','BFLIP','COL']}
CF=[]
for st,src,q in itertools.product(SIDE,VH,[.2,.3,.4]): CF.append(('F1 testerede yan motor sus',f'{st} {src} q{q}',dict(k=1,st=st,src=src,q=q)))
for wh,src,q,x in itertools.product(['yan3','flat'],VH,[.3,.4],[1,2,3]): CF.append(('F2 testere+OI1s şişme veto',f'{wh} {src} q{q} OI≥{x}%',dict(k=2,wh=wh,src=src,q=q,x=x)))
for xc,nt,src in itertools.product([80,85],[0,1],VH): CF.append(('F3 trim VHF modülasyon',f'testere X{xc} trend-trimsiz{nt} {src}',dict(k=3,xc=xc,nt=nt,src=src)))
for kc,kt,src in itertools.product([3.5,6.,0.],[3.5,4.5],VH): CF.append(('F4 karşı-poz VHF',f'testere k{kc} trend k{kt} {src}',dict(k=4,kc=kc,kt=kt,src=src)))
for P,O,vs in itertools.product([3,4,5],[1,2,3],['yok','5m48','1h14']): CF.append(('F5 OI 1s tükeniş kilidi',f'fiyat12≥{P} OI12≤-{O} vhf:{vs}',dict(k=5,P=P,O=O,vs=vs)))
for src,q,aq in itertools.product(VH,[.1,.2],[.7,.8,.9]): CF.append(('F6 yönsüz+yüksek vol flat',f'{src} q{q} ATR≥p{aq}',dict(k=6,src=src,q=q,aq=aq)))
print('aday',len(CF))
rows=[];BASE={}
for s in [0,1,2]:
    do,ex=comps(s);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64);do12=chg(sh(OI,s),12)
    tb=chainY(E0,pl);BASE[s]=HH(R6(tb,ex));rows.append(dict(f='BAZ V11.2',p='',s=s,**BASE[s]))
    for f,p,c in CF:
        k=c['k'];t,e2,Xa,th=tb,ex,X90,None
        if k in (1,2,6):
            v=VH[c['src']];chop=v<QI(v,c['q'])
            if k==1: m=chop
            elif k==2: m=chop&(do12>=c['x'])
            else: m=chop&(ATRP>=AQ[c['aq']])
            if k==1 or (k==2 and c['wh']=='yan3'):
                E2=dict(E0)
                for kk in SIDE[c.get('st','yan3')]: E2[kk]=np.where(m,0,E0[kk])
                t=chainY(E2,pl)
            else: t=np.where(m&(pl==0),0,tb).astype(np.int64)
        elif k==3:
            v=VH[c['src']];chop=v<QI(v,.3);tr=v>QI(v,.7);Xa=np.where(chop,float(c['xc']),np.where(tr&(c['nt']==1),101.,90.))
        elif k==4:
            v=VH[c['src']];chop=v<QI(v,.3);tr=v>QI(v,.7);th=np.where(chop,c['kc']*ATRP/100.,np.where(tr,c['kt']*ATRP/100.,thr))
        elif k==5:
            g=np.ones(N,bool) if c['vs']=='yok' else VH[c['vs']]<QI(VH[c['vs']],.4)
            e2=np.where(ex!=0,ex,np.where(g&(dp12>=c['P'])&(do12<=-c['O']),1,np.where(g&(dp12<=-c['P'])&(do12<=-c['O']),-1,0))).astype(np.int64)
        rows.append(dict(f=f,p=p,s=s,**HH(R6(t,e2,Xa,th))))
D=pd.DataFrame(rows);D.to_pickle('vhf.pkl');print('süre',round(time.time()-t0),'s')
for s in [0,1,2]: b=BASE[s];print(f"baz s={s}: net {b['net']:.0f} UPI {b['upi']:.2f} IS {b['IS_upi']:.2f} OOS {b['OOS_upi']:.2f} mdd {b['mdd']:.0f}")
X=D[D.f!='BAZ V11.2'].copy()
for m in ['net','upi','IS_upi','OOS_upi','mdd']: X['d_'+m]=X.apply(lambda r:r[m]-BASE[r.s][m],axis=1)
X['ok']=(X.d_upi>0)&(X.d_net>=0)&(X.d_OOS_upi>=0)&(X.d_IS_upi>=0)
G=X.groupby(['f','p'],sort=False).agg(ok3=('ok','all'),okn=('ok','sum'),dnet0=('d_net','first'),dnet_min=('d_net','min'),dupi_min=('d_upi','min'),doos_min=('d_OOS_upi','min'),dis_min=('d_IS_upi','min'),dmdd0=('d_mdd','first')).reset_index()
for f,g in G.groupby('f',sort=False): print(f'{f}: 3 kaymada geçen {int(g.ok3.sum())}/{len(g)} | ≥2 kayma {int((g.okn>=2).sum())} | ΔUPI(min) medyan {g.dupi_min.median():.2f} | Δnet(s0) medyan {g.dnet0.median():.0f} en iyi {g.dnet0.max():.0f}')
pd.set_option('display.width',300);print(G.sort_values(['okn','dupi_min'],ascending=False).head(12).round(2).to_string(index=False))
