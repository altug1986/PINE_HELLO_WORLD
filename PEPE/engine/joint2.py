exec(open('joint.py').read().split("\nCF=[]")[0])
import copy
def runj(s,js):
    c=A4;E={'RSIX':LP['RSIX'],'MIK':Z0,'COL':colf(1.,1.,24),'B4H':LP['B4H'],'BFLIP':Z0,'MID':midf(4.,12,3)}
    pl=plf(s,4,4,.08,72,55);ex=Z0;extra=None;wh=None;cnt={}
    for j in js:
        n=j['n'];dp=dpn(n);do=pchg(s,n);bz,bp=bfe(s,n)
        if j['k']=='J4':
            ex=np.where((dp>=j['x'])&(do<=-j['y'])&(bz<=-j['zb']),1,np.where((dp<=-j['x'])&(do<=-j['y'])&(bz<=-j['zb']),-1,0)).astype(np.int64);cnt['J4 sinyal mum']=int((ex!=0).sum())
        if j['k']=='J2':
            L=(dp>=j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp>=.5);S=(dp<=-j['x'])&(do>=j['y'])&(bz>=j['zb'])&(bp<=-.5)
            extra=vstate(C,L,S,0.,.03,48,F,F,0).astype(np.int64);wh='COL';cnt['J2 giriş']=int(((extra!=0)&(np.r_[0,extra[:-1]]==0)).sum())
    t=chainY(E,pl,extra,wh);sh_=run_v14(C,t,R7,4.5*ATRP/100.,ZF,ZI,ps,2.,ex,.02,144,90.,60.,12,.03,TICK,COMM,CASH,0,0)
    p=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(sh_[1],sh_[2],sh_[3]): p[e_:x_]=d_
    r=cbrk(C,p,2,100.,0,288,0,TICK,COMM,CASH);m=LM(r);_,ui=U(r,0,I_SPLIT);_,uo=U(r,I_SPLIT,N);m.update(Iupi=ui,Oupi=uo,**cnt);return m,r,sh_
J4=dict(k='J4',n=6,x=3,y=3,zb=1.5);J2=dict(k='J2',n=12,x=4,y=4,zb=1)
for nm,js in [('V11.6',[]),('+J4',[J4]),('+J2',[J2]),('+J4+J2',[J4,J2])]:
    ms=[runj(s,js)[0] for s in [0,1,2]];print(nm.ljust(8),' | '.join(f"s{s} {m['net']:.0f}/PF {m['pf']:.3f}/MaxDD {m['mdd']:.0f}/UPI {m['upi']:.1f}" for s,m in enumerate(ms)),{k:v for k,v in ms[0].items() if 'J' in k})
b0=[runj(s,[])[0] for s in [0,1,2]]
for base,grid,keys in [(J4,itertools.product([6,12],[2,3,4],[2,3,4],[1,1.5,2]),['n','x','y','zb']),(J2,itertools.product([6,12,24],[3,4,5],[3,4,5],[.5,1,1.5]),['n','x','y','zb'])]:
    res=[]
    for v in grid:
        j=dict(base,**dict(zip(keys,v)));ms=[runj(s,[j])[0] for s in [0,1,2]]
        res.append(dict(zip(keys,v),ok=all(m['net']>b['net'] and m['pf']>=b['pf']-1e-9 for m,b in zip(ms,b0)),dnet0=ms[0]['net']-b0[0]['net'],dnetmin=min(m['net']-b['net'] for m,b in zip(ms,b0))))
    R_=pd.DataFrame(res);print(f"\nKOMŞU {base['k']}: {len(R_)} ayar | 3 gecikmede net↑&PF≥ {int(R_.ok.sum())} | Δnet(s0) >0: {int((R_.dnet0>0).sum())} | medyan Δnet(s0) {R_.dnet0.median():.0f} | min-gecikme Δnet medyan {R_.dnetmin.median():.0f}")
    print(R_.sort_values('dnetmin',ascending=False).head(6).round(1).to_string(index=False))
m,r,sh_=runj(0,[J4]);mb,rb,shb=runj(0,[])
te=pd.Series(r[4],index=r[2]);tb=pd.Series(rb[4],index=rb[2]);print(f"\nJ4 etkisi: işlem {len(rb[4])}→{len(r[4])} | tükeniş kilidiyle kapanan gölge işlem {int((sh_[6]==6).sum())} (bunların neti {sh_[4][sh_[6]==6].sum():.0f}$)")
