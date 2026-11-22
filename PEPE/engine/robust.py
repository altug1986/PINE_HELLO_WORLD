exec(open('micro.py').read().split("\nbase=chain2(PL)")[0])
import time;t0=time.time()
def sh(x,k): return x if k==0 else np.r_[np.full(k,x[0]),x[:-k]]
bOIraw=BO['bOI'];bcl=pd.Series(BO['bC']).ffill().bfill().values
def bfeat(s):
    lo_=np.log(sh(bOIraw,s));out={}
    for n in [6,12,48]:
        dn=np.r_[np.full(n,np.nan),lo_[n:]-lo_[:-n]];sdn=pd.Series(dn).rolling(864,min_periods=200).std().values
        out[('z',n)]=np.nan_to_num(dn/sdn);out[('p',n)]=np.r_[np.zeros(n),(bcl[n:]/bcl[:-n]-1)*100]
    return out
def comps(s):
    do=chg(sh(OI,s),6)
    ex=np.where((dp6>=3)&(do<=-2),1,np.where((dp6<=-3)&(do<=-2),-1,0)).astype(np.int64)
    return do,ex
def RUN(t,ex): return run_v13(C,t,R7,thr,ZF,ZI,ps,2.0,ex,.03,144,90.,60.,12,.03,TICK,COMM,CASH)
CANDS=[]
for n,z_,p_,Hh in itertools.product([6,12],[1.5,2,2.5],[0.5,1.0,1.5],[24,48]):
    CANDS.append(('C1 yavaş ateşleme',f'n{n} z{z_} btc{p_}% H{Hh}',dict(kind='ign',n=n,z=z_,p=p_ if n==6 else p_*1.5,H=Hh)))
for n,z_ in itertools.product([6,12],[0.5,1.0,1.5]): CANDS.append(('C2 likidasyon+BTC temizlik',f'n{n} z≤-{z_}',dict(kind='liqc',n=n,z=z_)))
for n,z_ in itertools.product([12,48],[0.5,1.0,1.5]): CANDS.append(('C3 yan motor BTC rejim vetosu',f'n{n} |z|≥{z_}',dict(kind='veto',n=n,z=z_)))
for z_ in [1.0,1.5,2.0]: CANDS.append(('C4 tükeniş+BTC teyit',f'n6 z≤-{z_}',dict(kind='exc',z=z_)))
def build(cfg,s):
    do,ex=comps(s);bf=bfeat(s);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
    if cfg is None: return chain2(pl),ex
    k=cfg['kind']
    if k=='ign':
        z1,p1=bf[('z',cfg['n'])],bf[('p',cfg['n'])]
        ig=vstate(C,(z1>=cfg['z'])&(p1>=cfg['p']),(z1>=cfg['z'])&(p1<=-cfg['p']),0.,.03,cfg['H'],F,F,0).astype(np.int64)
        return chain2(pl,ig,'colust'),ex
    if k=='liqc':
        zz=bf[('z',cfg['n'])];pl2=vstate(C,(dp6<=-4)&(do<=-4)&(zz<=-cfg['z']),(dp6>=4)&(do<=-4)&(zz<=-cfg['z']),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
        return chain2(pl2),ex
    if k=='veto':
        zz,pp=bf[('z',cfg['n'])],bf[('p',cfg['n'])];E=dict(ENG)
        bad_long=(zz>=cfg['z'])&(pp<0);bad_short=(zz>=cfg['z'])&(pp>0)     # BTC'de karşı yönde yeni kaldıraç
        for kk in ['B4H','MID']: E[kk]=np.where(((ENG[kk]>0)&bad_long)|((ENG[kk]<0)&bad_short),0,ENG[kk])
        t=np.zeros(N,np.int64)
        for kk in ORDER[::-1]: t=np.where(E[kk]!=0,E[kk],t)
        return np.where(pl!=0,pl,t).astype(np.int64),ex
    if k=='exc':
        zz=bf[('z',6)];ex2=np.where(ex!=0,ex,np.where((dp6>=3)&(zz<=-cfg['z']),1,np.where((dp6<=-3)&(zz<=-cfg['z']),-1,0))).astype(np.int64)
        return chain2(pl),ex2
rows=[]
BASE={}
for s in [0,1,2]:
    t,ex=build(None,s);BASE[s]=HH(RUN(t,ex));rows.append(dict(h='V11.1',p='',s=s,**BASE[s]))
for h,p,cfg in CANDS:
    for s in [0,1,2]:
        t,ex=build(cfg,s);rows.append(dict(h=h,p=p,s=s,**HH(RUN(t,ex))))
D=pd.DataFrame(rows);D.to_pickle('robust.pkl');print('süre',round(time.time()-t0),'s')
for s in [0,1,2]: print(f'baz s={s}: net {BASE[s]["net"]:.0f} UPI {BASE[s]["upi"]:.2f} IS {BASE[s]["IS_upi"]:.2f} OOS {BASE[s]["OOS_upi"]:.2f}')
def ok(r): b=BASE[r.s];return (r.upi>b['upi'])&(r.net>=b['net'])&(r.OOS_upi>=b['OOS_upi'])&(r.IS_upi>=b['IS_upi'])
D['ok']=D.apply(ok,axis=1)
G=D[D.h!='V11.1'].groupby(['h','p']).agg(ok_all=('ok','all'),ok_n=('ok','sum'),upi0=('upi','first'),net0=('net','first'),upi_min_gain=('upi',lambda v:min(v.values[i]-BASE[i]['upi'] for i in range(3))),net_min_gain=('net',lambda v:min(v.values[i]-BASE[i]['net'] for i in range(3))),oos_min_gain=('OOS_upi',lambda v:min(v.values[i]-BASE[i]['OOS_upi'] for i in range(3)))).reset_index()
pd.set_option('display.width',250)
for h,g in G.groupby('h',sort=False): print(f'{h}: 3 kaymanın hepsinde geçen {int(g.ok_all.sum())}/{len(g)} | en az 2 kaymada {int((g.ok_n>=2).sum())}')
print(G[G.ok_n>=2].sort_values(['ok_n','upi_min_gain'],ascending=False).head(12).round(2).to_string(index=False))
