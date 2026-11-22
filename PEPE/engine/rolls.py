exec(open('rollf.py').read().split("\nRES=[];BASE=rec(SH)")[0])
tsS=SH[6];sm=tdS<0;ND=len(DAY_I);dix=np.searchsorted(DAY_I,txS,side='left')
@numba.njit
def vtr2(c,s):
    n=c.size;tx=np.empty(n,np.int64);pn=np.empty(n);dd=np.empty(n,np.int64);k=0;cur=0;ep=0.
    for i in range(n):
        if s[i]!=cur:
            if cur!=0:
                tx[k]=i;pn[k]=500.*(c[i]/ep-1.)*cur-0.4;dd[k]=cur;k+=1
            cur=s[i];ep=c[i]
    return tx[:k],pn[:k],dd[:k]
def tgt(mask):
    tg=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(teS[mask],txS[mask],tdS[mask]): tg[e_:x_]=d_
    return tg
def wpf(mask,W):
    gp=np.bincount(dix[mask],weights=np.maximum(tpS[mask],0),minlength=ND);gl=np.bincount(dix[mask],weights=np.maximum(-tpS[mask],0),minlength=ND)
    GP=pd.Series(gp).rolling(W,min_periods=W).sum().values;GL=pd.Series(gl).rolling(W,min_periods=W).sum().values
    return dbar(np.where(GL>0,GP/np.where(GL>0,GL,1),np.where(GP>0,99.,np.nan)))
gok=lambda h,th:np.isnan(h)|(h>=th)
TH=[.6,.7,.8,.9,1.,1.1,1.2,1.3,1.5];RES=[]
def add(fam,cfg,r):
    e=r[0][DAY_I];p=np.diff(np.r_[0.,e]);tx,td,tp=r[2],r[3],r[4]
    RES.append(dict(fam=fam,cfg=cfg,p=p,n=len(tp),pf=tp[tp>0].sum()/-tp[tp<=0].sum(),Lnet=tp[td>0].sum(),Snet=tp[td<0].sum(),nS=int((td<0).sum())))
f=lambda m:(int(m.sum()),round(tpS[m].sum()),round(tpS[m&(tpS>0)].sum()/-tpS[m&(tpS<=0)].sum(),3))
print('BAZ long (n,net,PF)',f(~sm),'| short',f(sm),'| short IS',f(sm&(txS<I_SPLIT)),'OOS',f(sm&(txS>=I_SPLIT)))
cnt=np.zeros(len(tdS),bool);cnt[1:]=(tsS[:-1]==4)&(teS[1:]==txS[:-1]);print('karşı-poz short',f(sm&cnt),'| normal short',f(sm&~cnt))
txs,tps=txS[sm],tpS[sm]
for K in [5,10,15,20,30,40,60,80,120]:
    h=tobar(txs,roll_tr(tps,K)[0])
    for th in TH: g=gok(h,th);add('F1 son K short PF',f'K{K} PF≥{th}',runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH))
for W in [7,14,30,60,90]:
    h=wpf(sm,W)
    for th in TH:
        g=gok(h,th);add('F2 son W gün short PF sert',f'W{W} PF≥{th}',runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH))
        add('F2 son W gün short PF giriş-kararı',f'W{W} PF≥{th}',runT(C,tgt(~sm|g[teS]),TICK,COMM,CASH))
for K in [10,20,40,80]:
    h=tobar(txS,roll_tr(tpS,K)[0])
    for th in TH: g=gok(h,th);add('F3 genel son K PF→short',f'K{K} PF≥{th}',runT(C,np.where((posS<0)&~g,0,posS),TICK,COMM,CASH))
EK=['RSIX','MIK','COL','B4H','BFLIP','MID'];ST={k:E[k].astype(np.int64) for k in EK};ST['PL']=pl;VT={k:vtr2(C,ST[k]) for k in ST}
for keys,kn in [(EK+['PL'],'tüm'),(['MIK','B4H','BFLIP','MID'],'yan')]:
    for K,th in itertools.product([5,10,20,40],[.6,.8,1.,1.2]):
        E2=dict(E);pl2=pl
        for k in keys:
            tx_,pn_,d_=VT[k];ms=d_<0;g=gok(tobar(tx_[ms],roll_tr(pn_[ms],K)[0]),th)
            if k=='PL': pl2=np.where((pl<0)&~g,0,pl).astype(np.int64)
            else: E2[k]=np.where((E[k]<0)&~g,0,E[k])
        add('F4 motor bazlı short PF',f'{kn} K{K} PF≥{th}',RUN5(chainY(E2,pl2),ex))
for K,th in itertools.product([10,20,40,80],[.6,.8,1.,1.2]):
    g=gok(tobar(txs,roll_tr(tps,K)[0]),th);E2={k:np.where((E[k]<0)&~g,0,E[k]) for k in EK}
    add('F5 zincirde short veto',f'K{K} PF≥{th}',RUN5(chainY(E2,np.where((pl<0)&~g,0,pl).astype(np.int64)),ex))
mn=sm&~cnt
for K in [10,20,40,80]:
    h=tobar(txS[mn],roll_tr(tpS[mn],K)[0])
    for th in TH: g=gok(h,th);add('F6 karşı-poz muaf',f'K{K} PF≥{th}',runT(C,tgt(~mn|g[teS]),TICK,COMM,CASH))
print('koşu',len(RES),'süre',round(time.time()-t0),'s')
bp=np.diff(np.r_[0.,SH[0][DAY_I]]);B_f,B_i,B_o=U2(bp),U2(bp[ISd]),U2(bp[~ISd])
R=pd.DataFrame(RES)
for c_,m in [('f',None),('i',ISd),('o',~ISd)]:
    v=np.array([U2(p if m is None else p[m]) for p in R.p]);R[c_+'net']=v[:,0];R[c_+'upi']=v[:,2];R[c_+'mdd']=v[:,3]
R['dnet']=R.fnet-B_f[0];R['dupi']=R.fupi-B_f[2];R['dIs']=R.iupi-B_i[2];R['dOs']=R.oupi-B_o[2];R['dIn']=R.inet-B_i[0];R['dOn']=R.onet-B_o[0]
pd.set_option('display.width',320)
for fam,g in R.groupby('fam',sort=False):
    w=g[(g.dupi>0)&(g.dnet>=0)];w2=g[(g.dIs>0)&(g.dOs>0)&(g.dIn>=0)&(g.dOn>=0)]
    print(f'\n=== {fam}: {len(g)} ayar | tüm dönem UPI↑&net↑ {len(w)} | iki yarıda da UPI↑&net↑ {len(w2)} | medyan Δnet {g.dnet.median():.0f} ΔUPI {g.dupi.median():.2f}')
    c1=g[g.dIn>=0];c1=c1 if len(c1) else g;s1=c1.loc[c1.iupi.idxmax()];c2=g[g.dOn>=0];c2=c2 if len(c2) else g;s2=c2.loc[c2.oupi.idxmax()]
    print(f'  IS seç→OOS: {s1.cfg} | IS ΔUPI {s1.dIs:+.2f} Δnet {s1.dIn:+.0f} → OOS ΔUPI {s1.dOs:+.2f} Δnet {s1.dOn:+.0f}')
    print(f'  OOS seç→IS: {s2.cfg} | OOS ΔUPI {s2.dOs:+.2f} Δnet {s2.dOn:+.0f} → IS ΔUPI {s2.dIs:+.2f} Δnet {s2.dIn:+.0f}')
    tot=0;win=0
    for b in range(7):
        m=BLK!=b;bo=U2(bp[m]);sc=[(U2(p[m])[2] if U2(p[m])[0]>=bo[0] else -1e9) for p in g.p];j=int(np.argmax(sc))
        dn=g.p.iloc[j][BLK==b].sum()-bp[BLK==b].sum();tot+=dn;win+=dn>0
    print(f'  blok dışarıda: Σ Δnet {tot:+.0f} | iyileşen blok {win}/7')
    print(g.assign(mn=np.minimum(g.dIs,g.dOs)).sort_values('mn',ascending=False).head(3)[['cfg','n','nS','fnet','fupi','fmdd','Snet','dIs','dOs','dIn','dOn']].round(2).to_string(index=False))
R.drop(columns='p').to_pickle('rolls.pkl')
