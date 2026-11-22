exec(open('cbrk.py').read().split("\nr0=cbrk(")[0])
@lru_cache(None)
def pchg(s,n): return chg(sh(OI,s),n)
@lru_cache(None)
def bfe(s,n):
    lo_=np.log(sh(bOIraw,s));dn=np.r_[np.full(n,np.nan),lo_[n:]-lo_[:-n]];sd=pd.Series(dn).rolling(864,min_periods=200).std().values
    return np.nan_to_num(dn/sd),np.r_[np.zeros(n),(bcl[n:]/bcl[:-n]-1)*100]
@lru_cache(None)
def dpn(n): return np.nan_to_num(chg(C,n))
def runx(c,s,j=None):
    E={'RSIX':LP['RSIX'] if c['rsix'] else Z0,'MIK':mikf(c['mik']) if c['mik'] else Z0,'COL':colf(c['fA'],c['fB'],c['hc']),'B4H':LP['B4H'] if c['b4h'] else Z0,'BFLIP':LP['BFLIP'] if c['bflip'] else Z0,'MID':midf(c['mm'],c['mh'],c['md']) if c['mid'] else Z0}
    pl=plf(s,c['pp'],c['po'],c['psl'],c['pb'],c['pre']);ex=exf(s,c['ep'],c['eo']) if c['exo'] else Z0;extra=None;wh=None;thr1=ZF;efl=ZI
    if j:
        n=j['n'];dp=dpn(n);do=pchg(s,n);bz,bp=bfe(s,n);k=j['k']
        if k=='J1':
            L=(dp<=-j['x'])&(do<=-j['y'])&(bz<=-j['zb']);S=(dp>=j['x'])&(do<=-j['y'])&(bz<=-j['zb'])
            jj=vstate(C,L,S,0.,c['psl'],c['pb'],R7>c['pre'],R7<100-c['pre'],0).astype(np.int64)
            pl=jj if j['m']=='yerine' else np.where(pl!=0,pl,jj).astype(np.int64)
        elif k=='J2':
            L=(dp>=j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp>=.5);S=(dp<=-j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp<=-.5)
            extra=vstate(C,L,S,0.,.03,48,F,F,0).astype(np.int64);wh=j['w']
        elif k=='J3':
            thr1=j['kk']*ATRP/100.;efl=np.where((dp<=-j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp<0),-1,np.where((dp>=j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp>0),1,0)).astype(np.int64)
        else:
            ex=np.where((dp>=j['x'])&(do<=-j['y'])&(bz<=-j['zb']),1,np.where((dp<=-j['x'])&(do<=-j['y'])&(bz<=-j['zb']),-1,0)).astype(np.int64)
    t=chainY(E,pl,extra,wh)
    sh_=run_v14(C,t,R7,c['k']*ATRP/100.,thr1,efl,ps,c['sT'],ex,c['slc'],c['mc'],c['X'],c['Y'],c['M'],c['z'],TICK,COMM,CASH,0,0)
    p=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(sh_[1],sh_[2],sh_[3]): p[e_:x_]=d_
    r=cbrk(C,p,2,100.,0,288,0,TICK,COMM,CASH);m=LM(r);_,ui=U(r,0,I_SPLIT);_,uo=U(r,I_SPLIT,N);m.update(Iupi=ui,Oupi=uo);return m
CF=[]
for m_,n,x,y,zb in itertools.product(['yerine','ek'],[6,12,24],[3,4,6],[2,3,4],[.5,1,1.5,2]): CF.append(('J1 piyasa geneli tasfiye dönüşü',f'{m_} n{n} fiyat{x} pOI{y} bOIz{zb}',dict(k='J1',m=m_,n=n,x=x,y=y,zb=zb)))
for n,x,y,zb,w in itertools.product([6,12,24],[2,4],[1,2,4],[1,2],['COL','B4H']): CF.append(('J2 ortak kaldıraç takibi',f'n{n} fiyat{x} pOI{y} bOIz{zb} {w}üstü',dict(k='J2',n=n,x=x,y=y,zb=zb,w=w)))
for kk,n,x,y,zb in itertools.product([2.5,3.5],[6,12],[1,2],[1,2],[1,2]): CF.append(('J3 ortak aleyhte birikim erken karşı',f'k{kk} n{n} fiyat{x} pOI{y} bOIz{zb}',dict(k='J3',kk=kk,n=n,x=x,y=y,zb=zb)))
for n,x,y,zb in itertools.product([6,12],[2,3,4],[1,2,3],[.5,1,1.5]): CF.append(('J4 ortak tükeniş kilidi',f'n{n} fiyat{x} pOI{y} bOIz{zb}',dict(k='J4',n=n,x=x,y=y,zb=zb)))
print('aday',len(CF))
rows=[];BASE={}
for s in [0,1,2]:
    BASE[s]=runx(A4,s);rows.append(dict(f='BAZ V11.6',p='',s=s,**BASE[s]))
    for f,p,c_ in CF: rows.append(dict(f=f,p=p,s=s,**runx(A4,s,c_)))
D=pd.DataFrame(rows);D.to_pickle('joint.pkl');print('süre',round(time.time()-t0),'s')
for s in [0,1,2]: b=BASE[s];print(f"baz s={s}: net {b['net']:.0f} PF {b['pf']:.3f} MaxDD {b['mdd']:.0f} UPI {b['upi']:.2f} IS/OOS UPI {b['Iupi']:.1f}/{b['Oupi']:.1f}")
X=D[D.f!='BAZ V11.6'].copy()
for m in ['net','pf','upi','Iupi','Oupi','mdd','Inet','Onet']: X['d'+m]=X.apply(lambda r:r[m]-BASE[r.s][m],axis=1)
X['ok']=(X.dnet>0)&(X.dpf>0)&(X.dInet>=0)&(X.dOnet>=0)
G=X.groupby(['f','p'],sort=False).agg(ok3=('ok','all'),okn=('ok','sum'),dnet0=('dnet','first'),dnet_min=('dnet','min'),dpf_min=('dpf','min'),dupi_min=('dupi','min'),dmdd0=('dmdd','first'),dOnet_min=('dOnet','min')).reset_index()
for f,g in G.groupby('f',sort=False): print(f'{f}: {len(g)} ayar | 3 gecikmede net↑PF↑ ve iki yarı≥ {int(g.ok3.sum())} | ≥2 gecikme {int((g.okn>=2).sum())} | Δnet(s0) medyan {g.dnet0.median():.0f} en iyi {g.dnet0.max():.0f}')
pd.set_option('display.width',300);print(G.sort_values(['okn','dnet_min'],ascending=False).head(12).round(3).to_string(index=False))
