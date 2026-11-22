exec(open('grand.py').read().split("\nrnd=random.Random(7)")[0])
for s in [1,2]: DO[s]=comps(s)[0]
A=dict(BC,mm=4.,md=3,mik=0,bflip=0,po=3,psl=.08,pb=72,exo=0,slc=.02);Bc=dict(BC,md=3,mik=0,po=3,psl=.08,pb=72,pre=60,exo=0,mc=72,M=24,z=.05)
M_=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b);L12=DAY_I[-365];rng=np.random.default_rng(1)
def full(r):
    f=M_(r);o=M_(r,I_SPLIT);ul,up=U(r);e=pd.Series(r[0][DAY_I],index=DAY_T);y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0];b6=e.resample('6ME').last().diff().dropna()
    return {'Net $':f['net'],'İşlem':f['n'],'PF':f['pf'],'Win %':f['win'],'Sharpe':f['sharpe'],'Sortino':f['sortino'],'Calmar':f['calmar'],'DD P95':f['p95'],'DD P98':f['p98'],'DD P99':f['p99'],'MaxDD':f['mdd'],'Ulcer':ul,'UPI':up,
            'İlk yarı net':r[0][I_SPLIT-1],'İkinci yarı net':o['net'],'İkinci yarı PF':o['pf'],'Son 12 ay':r[0][-1]-r[0][L12],'Yıl %':'/'.join(str(int(v/5)) for v in y.values),'6 ay en kötü':b6.min()}
T={};RR={}
for nm,c in [('BAZ V11.2',BC),('A hedef',A),('B dengeli',Bc)]: RR[nm]=runc(c);T[nm]=full(RR[nm])
pd.set_option('display.width',250);print(pd.DataFrame(T).to_string())
print('\nOI GECİKME (net / PF / MaxDD):')
for s in [0,1,2]: print(f'  s={s}: '+' | '.join(f"{nm} {m['net']:.0f}/{m['pf']:.3f}/{m['mdd']:.0f}" for nm,m in [(n_,LM(runc(c,s))) for n_,c in [('baz',BC),('A',A),('B',Bc)]]))
for nm,c,tg in [('A',A,lambda m:m['net']>=9200 and m['pf']>=1.65),('B',Bc,lambda m:m['net']>=9200 and m['mdd']<=285)]:
    res=[]
    for k,v in SP.items():
        i=v.index(c[k])
        for d in (-1,1):
            if 0<=i+d<len(v): c2=dict(c);c2[k]=v[i+d];m=ev(c2);res.append((k,v[i+d],m['net'],m['pf'],m['mdd'],tg(m)))
    Rn=pd.DataFrame(res,columns=['p','v','net','pf','mdd','ok']);ch=Rn[(Rn.net-ev(c)['net']).abs()>1]
    print(f"\nKOMŞU {nm}: {len(Rn)} komşu (etkili {len(ch)}) | hedefi tutan {int(Rn.ok.sum())} | etkili komşularda tutan {int(ch.ok.sum())}/{len(ch)} | net medyan {ch.net.median():.0f} PF medyan {ch.pf.median():.3f} MaxDD medyan {ch.mdd.median():.0f}")
    print('   bozanlar:',ch[~ch.ok][['p','v','net','pf','mdd']].round(2).values.tolist()[:10])
for nm in ['BAZ V11.2','A hedef','B dengeli']:
    r=RR[nm];tp=r[4];out=[]
    for mode in ['boot','perm']:
        qq=rng.choice(tp,(4000,len(tp))) if mode=='boot' else np.stack([rng.permutation(tp) for _ in range(4000)])
        qq=np.where(rng.random(qq.shape)<.05,0,qq-0.05);ec=np.cumsum(qq,1);ddm=(np.maximum.accumulate(np.maximum(ec,0),1)-ec).max(1)
        out.append(f"MC {mode}: net P1/P5 {np.percentile(ec[:,-1],[1,5]).round(0)} DD P95/P99 {np.percentile(ddm,[95,99]).round(0)}")
    te,tx,td=r[1],r[2],r[3];ret=C[tx]/C[te]-1;sysn=(500*ret*td-0.4).sum();mk=np.array([(500*ret*rng.choice([-1,1],len(ret))-0.4).sum() for _ in range(2000)]);dur=tx-te;mk2=[]
    for _ in range(800):
        s_=rng.integers(300,N-dur.max()-1,len(dur));mk2.append((500*(C[s_+dur]/C[s_]-1)*rng.permutation(td)-0.4).sum())
    c=BC if nm.startswith('BAZ') else (A if nm.startswith('A') else Bc)
    E={'RSIX':LP['RSIX'] if c['rsix'] else Z0,'MIK':mikf(c['mik']) if c['mik'] else Z0,'COL':colf(c['fA'],c['fB'],c['hc']),'B4H':LP['B4H'] if c['b4h'] else Z0,'BFLIP':LP['BFLIP'] if c['bflip'] else Z0,'MID':midf(c['mm'],c['mh'],c['md']) if c['mid'] else Z0}
    pl=plf(0,c['pp'],c['po'],c['psl'],c['pb'],c['pre']);ex=exf(0,c['ep'],c['eo']) if c['exo'] else Z0
    rs=run_v14(C,chainY(E,pl),R7,c['k']*ATRP/100.,ZF,ZI,ps,c['sT'],ex,c['slc'],c['mc'],c['X'],c['Y'],c['M'],c['z'],3e-7,.0008,CASH,0,0)
    print(f"\n{nm}: "+' | '.join(out)+f" | MONKEY yön geçen {(mk>=sysn).mean():.4f} zaman geçen {(np.array(mk2)>=sysn).mean():.4f} | STRES net {rs[4].sum():.0f} PF {rs[4][rs[4]>0].sum()/-rs[4][rs[4]<=0].sum():.3f}")
