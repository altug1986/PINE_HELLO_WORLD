exec(open('liq2.py').read().split("\nrows=[dict(p='V11'")[0])
BO=pickle.load(open('/home/claude/data/btcOI.pkl','rb'))
lo=np.log(BO['bOI']);d1=np.r_[np.nan,np.diff(lo)];d2=np.r_[np.nan,np.nan,lo[2:]-lo[:-2]]
sd=pd.Series(d1).rolling(288,min_periods=100).std().values
oz1=np.nan_to_num(d1/sd);oz2=np.nan_to_num(d2/(sd*np.sqrt(2)))
bc=BO['bC'];bp1=np.nan_to_num(np.r_[np.nan,(bc[1:]/bc[:-1]-1)*100]);bp2=np.nan_to_num(np.r_[np.nan,np.nan,(bc[2:]/bc[:-2]-1)*100])
dp2=chg(C,2)
PL=vstate(C,(dp6<=-4)&(do6<=-4),(dp6>=4)&(do6<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
def liqeng(L,S): return vstate(C,L,S,0.,.05,48,R7>55,R7<45,0).astype(np.int64)
def chain2(pl,extra=None,pos_='top'):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]:
        t=np.where(ENG[k]!=0,ENG[k],t)
        if extra is not None and pos_=='colust' and k=='COL': t=np.where(extra!=0,extra,t)
    if extra is not None and pos_=='top': t=np.where(extra!=0,extra,t)
    return np.where(pl!=0,pl,t).astype(np.int64)
base=chain2(PL)
rows=[dict(h='V11.1',p='',**HH(R13(t=base)))]
b0=rows[0]
# M1a: likidasyon motoru BTC kaldıraç temizliği teyidi
for zb in [1,2,3]:
    L=(dp6<=-4)&(do6<=-4)&(oz2<=-zb);S=(dp6>=4)&(do6<=-4)&(oz2<=-zb)
    rows.append(dict(h='M1a likidasyon+BTC teyit',p=f'BTC OIz2≤-{zb}',**HH(R13(t=chain2(liqeng(L,S))))))
# M1b: BTC öncülü mikro likidasyon tetiği (VEYA)
for zb,pb,pp in itertools.product([2,3,4],[0.3,0.6,1.0],[1.5,2.5]):
    L=((dp6<=-4)&(do6<=-4))|((dp2<=-pp)&(oz2<=-zb)&(bp2<=-pb));S=((dp6>=4)&(do6<=-4))|((dp2>=pp)&(oz2<=-zb)&(bp2>=pb))
    rows.append(dict(h='M1b BTC öncülü dönüş',p=f'z{zb} btc{pb} pepe{pp}',**HH(R13(t=chain2(liqeng(L,S))))))
# M2: BTC kaldıraç ateşlemesi → PEPE takip
for zb,pb,Hh,ps_ in itertools.product([2,3,4],[0.3,0.6],[12,24],['top','colust']):
    ig=vstate(C,(oz1>=zb)&(bp1>=pb),(oz1>=zb)&(bp1<=-pb),0.,.03,Hh,F,F,0).astype(np.int64)
    rows.append(dict(h='M2 BTC ateşleme takip',p=f'z{zb} btc{pb} H{Hh} {ps_}',**HH(R13(t=chain2(PL,ig,ps_)))))
# M3: mikro risk kapatma (BTC temizliği aleyhte → kâr kilitle)
for zb,pb in itertools.product([1.5,2,3,4],[0.2,0.4,0.6,1.0]):
    ex3=np.where(EX!=0,EX,np.where((oz2<=-zb)&(bp2<=-pb),1,np.where((oz2<=-zb)&(bp2>=pb),-1,0))).astype(np.int64)
    rows.append(dict(h='M3 mikro risk kilidi',p=f'z{zb} btc{pb}',**HH(R13(t=base,ex=ex3))))
# M4: BTC şokunda erken karşı pozisyon
for k1,zb,pb in itertools.product([1.5,2.5],[2,3],[0.3,0.6]):
    efl=np.where((oz2<=-zb)&(bp2<=-pb),-1,np.where((oz2<=-zb)&(bp2>=pb),1,0)).astype(np.int64)
    rows.append(dict(h='M4 BTC şok erken karşı',p=f'k{k1} z{zb} btc{pb}',**HH(R13(t=base,thr1=k1*ATRP/100.,efl=efl))))
D=pd.DataFrame(rows);D.to_pickle('micro.pkl');pd.set_option('display.width',300);b=D.iloc[0]
print(D.iloc[:1][['net','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26']].round(2).to_string(index=False))
for h,g in D.groupby('h',sort=False):
    if h=='V11.1':continue
    s=g[(g.upi>b.upi)&(g.net>=b.net)&(g.OOS_upi>=b.OOS_upi*0.97)&(g.IS_upi>=b.IS_upi*0.97)]
    print(f'{h}: geçen {len(s)}/{len(g)} | UPI medyan {g.upi.median():.1f} | net medyan {g.net.median():.0f} | en iyi net {g.net.max():.0f}')
    if len(s): print(s.sort_values('upi',ascending=False).head(5)[['p','net','pct','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26','r2_12']].round(2).to_string(index=False))
