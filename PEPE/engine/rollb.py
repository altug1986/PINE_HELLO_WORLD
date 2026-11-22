exec(open('rolls.py').read().split("\ntxs,tps=txS[sm],tpS[sm]")[0])
txs,tps=txS[sm],tpS[sm]
@numba.njit
def ban_gate(n,txs,pfs,thr,Db,modeB):
    g=np.ones(n,np.bool_);bu=-1;j=0;m=txs.size;cur=np.nan
    for i in range(n):
        while j<m and txs[j]<=i:
            cur=pfs[j]
            if (not np.isnan(cur)) and cur<thr and txs[j]+Db>bu: bu=txs[j]+Db
            j+=1
        ok=i>bu
        if modeB and (not np.isnan(cur)) and cur<thr: ok=False
        g[i]=ok
    return g
RES=[];PF_={}
for K in [10,15,20,30,40,60,80,120]: PF_[K]=roll_tr(tps,K)[0]
for K,th,D,mB,em in itertools.product([10,15,20,30,40,60,80,120],[.8,.9,1.,1.1,1.2],[0,1,2,3,5,7],[0,1],['giriş','sert']):
    if D==0 and mB==0: continue
    g=ban_gate(N,txs,PF_[K],th,D*288,mB==1)
    r=runT(C,tgt(~sm|g[teS]),TICK,COMM,CASH) if em=='giriş' else runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH)
    add('ban',f"K{K} PF≥{th} {D}g {'yasak+PF' if mB else 'yasak'} {em}",r);RES[-1].update(K=K,th=th,D=D,mB=mB,em=em)
print('koşu',len(RES),'süre',round(time.time()-t0),'s')
bp=np.diff(np.r_[0.,SH[0][DAY_I]]);B_f,B_i,B_o=U2(bp),U2(bp[ISd]),U2(bp[~ISd])
R=pd.DataFrame(RES)
for c_,m in [('f',None),('i',ISd),('o',~ISd)]:
    v=np.array([U2(p if m is None else p[m]) for p in R.p]);R[c_+'net']=v[:,0];R[c_+'upi']=v[:,2];R[c_+'mdd']=v[:,3]
R['dnet']=R.fnet-B_f[0];R['dupi']=R.fupi-B_f[2];R['dIs']=R.iupi-B_i[2];R['dOs']=R.oupi-B_o[2];R['dIn']=R.inet-B_i[0];R['dOn']=R.onet-B_o[0]
R['l12']=[p[-365:].sum() for p in R.p];bl12=bp[-365:].sum();R['dl12']=R.l12-bl12
pd.set_option('display.width',320)
w=R[(R.dupi>0)&(R.dnet>=0)];w2=R[(R.dIs>0)&(R.dOs>0)&(R.dIn>=0)&(R.dOn>=0)]
print(f'{len(R)} ayar | tüm dönem UPI↑&net↑ {len(w)} | iki yarıda da UPI↑&net↑ {len(w2)} | medyan Δnet {R.dnet.median():.0f} | son 12 ay Δnet medyan {R.dl12.median():.0f}')
c1=R[R.dIn>=0];s1=c1.loc[c1.iupi.idxmax()];c2=R[R.dOn>=0];s2=c2.loc[c2.oupi.idxmax()]
print(f'IS seç→OOS: {s1.cfg} | IS ΔUPI {s1.dIs:+.2f} Δnet {s1.dIn:+.0f} → OOS ΔUPI {s1.dOs:+.2f} Δnet {s1.dOn:+.0f}')
print(f'OOS seç→IS: {s2.cfg} | OOS ΔUPI {s2.dOs:+.2f} Δnet {s2.dOn:+.0f} → IS ΔUPI {s2.dIs:+.2f} Δnet {s2.dIn:+.0f}')
tot=0;win=0
for b in range(7):
    m=BLK!=b;bo=U2(bp[m]);sc=[(U2(p[m])[2] if U2(p[m])[0]>=bo[0] else -1e9) for p in R.p];j=int(np.argmax(sc))
    dn=R.p.iloc[j][BLK==b].sum()-bp[BLK==b].sum();tot+=dn;win+=dn>0
print(f'blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
for mB in [0,1]:
    q=R[(R.th==1.)&(R.mB==mB)&(R.em=='giriş')]
    print(f"\nPF≥1.0 {'yasak+PF düzelene kadar' if mB else 'sadece D gün yasak'} | Δnet tüm / OOS (satır K, sütun gün)")
    print(q.pivot(index='K',columns='D',values='dnet').round(0).astype(int).astype(str).add(' / ').add(q.pivot(index='K',columns='D',values='dOn').round(0).astype(int).astype(str)).to_string())
print('\nen iyi 6 (iki yarının kötüsüne göre):')
print(R.assign(mn=np.minimum(R.dIs,R.dOs)).sort_values('mn',ascending=False).head(6)[['cfg','n','nS','fnet','fupi','fmdd','dIs','dOs','dIn','dOn','dl12']].round(2).to_string(index=False))
R.drop(columns='p').to_pickle('rollb.pkl')
