exec(open('anti.py').read().split("\n# regresyon")[0])
def RUN5(t,ex,R=R7,thr1=ZF,efl=ZI,stress=False):
    return run_v14(C,t,R,thr,thr1,efl,ps,2.0,ex,.03,144,90.,60.,12,.03,3*TICK if stress else TICK,.0008 if stress else COMM,CASH,0,0)
def EV5(t,ex,**k):
    r=RUN5(t,ex,**k);h=HH(r);h.update(churn(r));hs=HH(RUN5(t,ex,stress=True,**k));h.update(s_net=hs['net'],s_upi=hs['upi'],s_oos=hs['OOS_upi']);return h
def chainY(E,pl,extra=None,where=None):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]:
        t=np.where(E[k]!=0,E[k],t)
        if extra is not None and where==k: t=np.where(extra!=0,extra,t)
    return np.where(pl!=0,pl,t).astype(np.int64)
def bq(s,n=24):
    lo_=np.log(sh(bOIraw,s));dn=np.r_[np.full(n,np.nan),lo_[n:]-lo_[:-n]]
    sd=pd.Series(dn).rolling(864,min_periods=200).std().values
    return np.nan_to_num(dn/sd),np.r_[np.zeros(n),(bcl[n:]/bcl[:-n]-1)*100]
dp24=chg(C,24);MIK2=deb(ENG['MIK'].astype(np.int64),2)
C1=[]
for nm,ks in [('MIK2+COL2',['COL']),('MIK2+MID2',['MID']),('MIK2+B4H2+BFLIP2',['B4H','BFLIP'])]: C1.append(('S1 teyit',nm,dict(kind='deb',ks=ks)))
for zt,pt,Hh,ps_ in itertools.product([.5,1,1.5],[.5,1],[24,48],['COL','B4H']): C1.append(('B1 BTC2s takip',f'z{zt} p{pt} H{Hh} {ps_}üstü',dict(kind='fol',zt=zt,pt=pt,H=Hh,w=ps_)))
for zt,pt,v in itertools.product([.5,1,1.5],[.25,.5,1],['Q3','Q34']): C1.append(('B2 BTC2s veto',f'z{zt} p{pt} {v}',dict(kind='veto',zt=zt,pt=pt,v=v)))
for P,zt,pt in itertools.product([2,3],[.5,1,1.5],[.25,.5]): C1.append(('B3 tükeniş teyit',f'pepe{P} z{zt} p{pt}',dict(kind='exh',P=P,zt=zt,pt=pt)))
for zt,pt,x in itertools.product([1,1.5,2],[.5,1],[3,5]): C1.append(('B4 teslimiyet dönüşü',f'z{zt} p{pt} pepe24±{x}',dict(kind='cap',zt=zt,pt=pt,x=x)))
for zt,pt in itertools.product([.5,1,1.5],[.25,.5,1]): C1.append(('B5 trim modülasyonu',f'z{zt} p{pt}',dict(kind='trim',zt=zt,pt=pt)))
for k1,zt,pt in itertools.product([2.5,3.5],[.5,1,1.5],[.5,1]): C1.append(('B6 erken karşı',f'k{k1} z{zt} p{pt}',dict(kind='ahf',k=k1,zt=zt,pt=pt)))
rows=[];BASE={}
for s in [0,1,2]:
    do,ex=comps(s);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
    zO,pB=bq(s);E=dict(ENG);E['MIK']=MIK2;tb=chainY(E,pl)
    rows.append(dict(f='V11.1',p='',s=s,**EV5(chain2(pl),ex)));BASE[s]=EV5(tb,ex);rows.append(dict(f='BAZ V11.2 (MIK2)',p='',s=s,**BASE[s]))
    for f,p,c in C1:
        k=c['kind'];t,e2,kw=tb,ex,{}
        if k=='deb':
            E2=dict(E)
            for kk in c['ks']: E2[kk]=deb(ENG[kk].astype(np.int64),2)
            t=chainY(E2,pl)
        elif k=='fol':
            ig=vstate(C,(pB>=c['pt'])&(zO>=c['zt']),(pB<=-c['pt'])&(zO>=c['zt']),0.,.03,c['H'],F,F,0).astype(np.int64);t=chainY(E,pl,ig,c['w'])
        elif k=='veto':
            bl=(pB<=-c['pt'])&((zO>=c['zt']) if c['v']=='Q3' else (np.abs(zO)>=c['zt']));bs=(pB>=c['pt'])&((zO>=c['zt']) if c['v']=='Q3' else (np.abs(zO)>=c['zt']))
            E2=dict(E)
            for kk in ['B4H','MID']: E2[kk]=np.where(((E[kk]>0)&bl)|((E[kk]<0)&bs),0,E[kk])
            t=chainY(E2,pl)
        elif k=='exh':
            e2=np.where(ex!=0,ex,np.where((dp6>=c['P'])&(pB>=c['pt'])&(zO<=-c['zt']),1,np.where((dp6<=-c['P'])&(pB<=-c['pt'])&(zO<=-c['zt']),-1,0))).astype(np.int64)
        elif k=='cap':
            L=(pB<=-c['pt'])&(zO<=-c['zt'])&(dp24<=-c['x']);S=(pB>=c['pt'])&(zO<=-c['zt'])&(dp24>=c['x'])
            ce_=vstate(C,L,S,0.,.05,48,R7>55,R7<45,0).astype(np.int64);t=chainY(E,np.where(pl!=0,pl,ce_).astype(np.int64))
        elif k=='trim':
            q1=(pB>=c['pt'])&(zO>=c['zt']);q3=(pB<=-c['pt'])&(zO>=c['zt'])
            kw['R']=np.where(q1&(R7>=90),89.9,np.where(q3&(R7<=10),10.1,R7))
        elif k=='ahf':
            kw['thr1']=c['k']*ATRP/100.;kw['efl']=np.where((pB<=-c['pt'])&(zO>=c['zt']),-1,np.where((pB>=c['pt'])&(zO>=c['zt']),1,0)).astype(np.int64)
        rows.append(dict(f=f,p=p,s=s,**EV5(t,e2,**kw)))
D=pd.DataFrame(rows);D.to_pickle('btc2h.pkl');print('süre',round(time.time()-t0),'s');pd.set_option('display.width',320)
c=['net','pf','n','mdd','upi','IS_upi','OOS_upi','y26','blip','ff','s_net','s_upi','s_oos']
print(D[D.f.isin(['V11.1','BAZ V11.2 (MIK2)'])].set_index(['f','s'])[c].round(2).to_string())
X=D[~D.f.isin(['V11.1','BAZ V11.2 (MIK2)'])].copy()
for m in ['net','upi','IS_upi','OOS_upi','s_upi']: X['d_'+m]=X.apply(lambda r:r[m]-BASE[r.s][m],axis=1)
X['ok']=(X.d_upi>0)&(X.d_net>=0)&(X.d_OOS_upi>=0)&(X.d_IS_upi>=0)
G=X.groupby(['f','p'],sort=False).agg(ok3=('ok','all'),okn=('ok','sum'),dnet0=('d_net','first'),dupi_min=('d_upi','min'),doos_min=('d_OOS_upi','min'),dis_min=('d_IS_upi','min'),dnet_min=('d_net','min'),dsupi_min=('d_s_upi','min')).reset_index()
for f,g in G.groupby('f',sort=False): print(f'{f}: 3 kaymada geçen {int(g.ok3.sum())}/{len(g)} | ≥2 kayma {int((g.okn>=2).sum())} | ΔUPI(min) medyan {g.dupi_min.median():.2f} | Δnet(s0) medyan {g.dnet0.median():.0f}')
print(G.sort_values(['okn','dupi_min'],ascending=False).head(12).round(2).to_string(index=False))
