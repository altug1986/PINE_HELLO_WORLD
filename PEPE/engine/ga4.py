exec(open('flow1.py').read().split("\nrows=[dict(h='V11'")[0])
import time
T0=chainE(ENG)
EXc={}
def exarr(P,O_):
    k=(P,O_)
    if k not in EXc: EXc[k]=np.where((dp6>=P)&(do6<=-O_),1,np.where((dp6<=-P)&(do6<=-O_),-1,0)).astype(np.int64)
    return EXc[k]
SP=dict(k=[3.5,4,4.5,5,5.5,6],slc=[.02,.03,.04,.05],mc=[72,144,288],X=[85,88,90,92,95],Y=[50,55,60,65],M=[6,12,24],z=[.02,.03,.04],sT=[1.5,2.0,2.5,3.0,99.],P=[2,3,4,5],O=[1.5,2,3])
BASE=dict(k=4.5,slc=.03,mc=144,X=90,Y=60,M=12,z=.03,sT=2.0,P=3,O=2)
def runp(p,tick=TICK,comm=COMM):
    return run_v13(C,T0,R7,p['k']*ATRP/100.,ZF,ZI,ps,p['sT'],exarr(p['P'],p['O']),p['slc'],p['mc'],float(p['X']),float(p['Y']),p['M'],p['z'],tick,comm,CASH)
names=list(SP);sc={};rng=np.random.default_rng(3);t0=time.time()
def key(p): return tuple(p[n] for n in names)
def E(p):
    k=key(p)
    if k not in sc:
        r=runp(p);sc[k]=U(r,0,I_SPLIT)[1]
    return sc[k]
E(BASE)
for _ in range(40): E({n:SP[n][rng.integers(len(SP[n]))] for n in names})
for g in range(20):
    el=sorted(sc,key=sc.get,reverse=True)[:8]
    for _ in range(14):
        a,b=(dict(zip(names,el[j])) for j in rng.choice(len(el),2));c={n:(a[n] if rng.random()<.5 else b[n]) for n in names}
        for n in names:
            if rng.random()<.2: i_=SP[n].index(c[n]);c[n]=SP[n][int(np.clip(i_+rng.choice([-1,1]),0,len(SP[n])-1))]
        E(c)
print('değerlendirme',len(sc),'süre',round(time.time()-t0),'s | baz IS UPI',round(sc[key(BASE)],2),'| en iyi IS UPI',round(max(sc.values()),2))
pickle.dump(sc,open('/home/claude/data/ga4.pkl','wb'))
rows=[]
for k in [key(BASE)]+sorted(sc,key=sc.get,reverse=True)[:10]:
    p=dict(zip(names,k));r=runp(p);rows.append(dict(**p,ISupi=sc[k],**{kk:v for kk,v in HH(r).items() if kk in ['net','pf','n','mdd','ulcer','upi','OOS_upi','Ocal','Onet','y26','r2_12','sharpe']}))
pd.set_option('display.width',300);print(pd.DataFrame(rows).round(2).to_string(index=False))
