exec(open("bscan.py").read().split("\nrows=[]\n")[0])
import time
def build2(p):
    gA_=(a30>p['A30'])&(a5>20)&(a5>lag(a5,12));gB_=(a30>p['B30'])&(a5>lag(a5,6));gM_=(a30>p['M30'])&(a5>lag(a5,3))
    mB_=mom(p['Bn']);mM_=mom(p['Mn'])
    LA,SA=gA_&(mA>=p['Am']),gA_&(mA<=-p['Am']);LB,SB=gB_&(mB_>=p['Bm']),gB_&(mB_<=-p['Bm'])
    t=np.sign(pos(run(C,LA,SA,p['hold'],TICK,COMM,CASH))+pos(run(C,LA|LB,SA|SB,p['hold'],TICK,COMM,CASH)))
    pm=vstate(C,gM_&(mM_>=p['Mm']),gM_&(mM_<=-p['Mm']),0.,0.,0,F,F,p['Mh'])
    return run_x(C,np.where(pR!=0,pR,np.where(t!=0,t,pm)).astype(np.int64),pR!=0,np.zeros(N,bool),0.,0.,0,0,TICK,COMM,CASH)
SPACE=dict(A30=[24,26,28,30,32],Am=[2.5,2.75,3,3.25,3.5],B30=[20,22,25,28],Bn=[36,42,48,54,60],Bm=[2,2.25,2.5,2.75,3],M30=[15,18,20,22],Mn=[12,18,24,30,36],Mm=[2.5,2.75,3,3.25,3.5],Mh=[6,12,24],hold=[12,18,24,36])
BASE=dict(A30=28,Am=3,B30=25,Bn=48,Bm=2.5,M30=18,Mn=24,Mm=3,Mh=12,hold=24)
CACHE={}
def ev(p):
    r=build2(p);i=metrics(r[0],*r[2:6],0,I_SPLIT);CACHE[tuple(p.values())]=r;return i['score']*(1 if i['net']>0 else 0)
t0=time.time()
exec(open('/home/claude/tvopt_cs.py').read()) if False else None
rng=np.random.default_rng(7);names=list(SPACE);sc={}
def key(p): return tuple(p[n] for n in names)
def E(p):
    k=key(p)
    if k not in sc: sc[k]=ev(dict(p))
    return sc[k]
E(BASE)
for i in range(30): E({n:SPACE[n][rng.integers(len(SPACE[n]))] for n in names})
for g in range(12):
    elite=sorted(sc,key=sc.get,reverse=True)[:5]
    for _ in range(12):
        a,b=(dict(zip(names,elite[j])) for j in rng.choice(len(elite),2))
        c={n:(a[n] if rng.random()<.5 else b[n]) for n in names}
        for n in names:
            if rng.random()<.3: ix=SPACE[n].index(c[n]);c[n]=SPACE[n][int(np.clip(ix+rng.choice([-1,1]),0,len(SPACE[n])-1))]
        E(c)
    bk=max(sc,key=sc.get);print(f'gen{g} best IS-score {sc[bk]:.3f} evals {len(sc)} t {time.time()-t0:.0f}s',flush=True)
pickle.dump((sc,names),open('/home/claude/data/ga_sc.pkl','wb'))
out=[]
for k in sorted(sc,key=sc.get,reverse=True)[:8]+[key(BASE)]:
    r=CACHE[k];f=metrics(r[0],*r[2:6]);o=metrics(r[0],*r[2:6],I_SPLIT,N);i=metrics(r[0],*r[2:6],0,I_SPLIT);dd=tvdd(r)
    ws=sum(metrics(r[0],*r[2:6],int(a),int(b))['net'] for a,b in WI)
    out.append(dict(**dict(zip(names,k)),ISsc=sc[k],net=f['net'],pf=f['pf'],n=f['n'],q=f['net']/dd,p98=f['p98'],Inet=i['net'],Onet=o['net'],Ical=i['calmar'],Ocal=o['calmar'],Wsum=ws))
pd.set_option('display.width',280);print(pd.DataFrame(out).round(2).to_string(index=False))
