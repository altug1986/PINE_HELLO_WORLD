exec(open('grand3.py').read().split("\nT={};RR={}")[0])
A4=dict(A,po=4);MD=dict(BC,mm=4.,mh=6,md=3,mik=0,bflip=0,po=5,psl=.08,pb=72,exo=0,k=6.,slc=.02,mc=72,z=.05)
T={};RR={}
for nm,c in [('BAZ V11.2',BC),('A-OI4',A4),('Min-DD',MD)]: RR[nm]=runc(c);T[nm]=full(RR[nm])
pd.set_option('display.width',250);print(pd.DataFrame(T).round(2).to_string())
for nm,c in [('A-OI4',A4),('Min-DD',MD)]:
    res=[]
    for k,v in SP.items():
        i=v.index(c[k])
        for d in (-1,1):
            if 0<=i+d<len(v): c2=dict(c);c2[k]=v[i+d];m=ev(c2);res.append((k,v[i+d],m['net'],m['pf'],m['mdd']))
    Rn=pd.DataFrame(res,columns=['p','v','net','pf','mdd']);ch=Rn[(Rn.net-ev(c)['net']).abs()>1];okm=(ch.net>=9200)&(ch.pf>=1.65)
    print(f"KOMŞU {nm}: etkili {len(ch)} | hedefi tutan {int(okm.sum())} | net>baz&PF>baz {int(((ch.net>8927)&(ch.pf>1.538)).sum())} | net medyan {ch.net.median():.0f} PF medyan {ch.pf.median():.3f} MaxDD medyan {ch.mdd.median():.0f}")
    r=RR[nm];tp=r[4];o=[]
    for mode in ['boot','perm']:
        qq=rng.choice(tp,(4000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(4000)])
        qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
        o.append(f"MC {mode} net P1/P5 {np.percentile(ec[:,-1],[1,5]).round(0)} DD P95/P99 {np.percentile(ddm,[95,99]).round(0)}")
    te,tx,td=r[1],r[2],r[3];ret=C[tx]/C[te]-1;sysn=(500*ret*td-0.4).sum();mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-0.4).sum() for _ in range(2000)])
    E={'RSIX':LP['RSIX'],'MIK':Z0,'COL':colf(1.,1.,24),'B4H':LP['B4H'],'BFLIP':Z0,'MID':midf(c['mm'],c['mh'],c['md'])};pl=plf(0,c['pp'],c['po'],c['psl'],c['pb'],c['pre'])
    rs=run_v14(C,chainY(E,pl),R7,c['k']*ATRP/100.,ZF,ZI,ps,c['sT'],Z0,c['slc'],c['mc'],c['X'],c['Y'],c['M'],c['z'],3e-7,.0008,CASH,0,0)
    print('   '+' | '.join(o)+f" | monkey yön geçen {(mk>=sysn).mean():.4f} | STRES net {rs[4].sum():.0f} PF {rs[4][rs[4]>0].sum()/-rs[4][rs[4]<=0].sum():.3f}")
pickle.dump(RR['A-OI4'],open('V115.pkl','wb'))
