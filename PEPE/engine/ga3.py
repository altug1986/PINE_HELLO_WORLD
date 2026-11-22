exec(open('ga2.py').read().split("# baz V6.2")[0])
def evalnet(w,form):
    r=run_t(C,target(w,form),TICK,COMM,CASH);i=metrics(r[0],*r[2:],0,I_SPLIT)
    wk=sum(metrics(r[0],*r[2:],a,b)['net'] for a,b,isw in WI if isw)
    pen=0 if (i['calmar']>=6.0 and wk>=600 and i['pf']>=1.5) else 3000
    return i['net']-pen
sc2={};rng=np.random.default_rng(5);seed=pickle.load(open('/home/claude/data/ga2_sc.pkl','rb'))
def E(g):
    k=tuple(g)
    if k not in sc2: sc2[k]=evalnet(g[:-1],g[-1])
    return sc2[k]
for k in sorted(seed,key=seed.get,reverse=True)[:30]: E(list(k))
for i in range(60): E(list(rng.integers(0,5,len(names)))+[int(rng.integers(0,2))])
for gen in range(30):
    el=sorted(sc2,key=sc2.get,reverse=True)[:8]
    for _ in range(24):
        a,b=(list(el[j]) for j in rng.choice(len(el),2));c=[a[j] if rng.random()<.5 else b[j] for j in range(len(a))]
        for j in range(len(c)-1):
            if rng.random()<.15: c[j]=int(np.clip(c[j]+rng.choice([-1,1]),0,4))
        if rng.random()<.1: c[-1]=1-c[-1]
        E(c)
pickle.dump(sc2,open('/home/claude/data/ga3_sc.pkl','wb'))
rows=[];seen=set()
for k in sorted(sc2,key=sc2.get,reverse=True):
    m,r=evalw(list(k[:-1]),k[-1],True);sig=round(m['net'],1)
    if sig in seen: continue
    seen.add(sig);rows.append(dict(genom=''.join(map(str,k[:-1])),form='ÖNC' if k[-1] else 'OY',ISnet=sc2[k],**m))
    if len(rows)>=8: break
pd.set_option('display.width',290);print(pd.DataFrame(rows).drop(columns=['obj']).round(2).to_string(index=False))
