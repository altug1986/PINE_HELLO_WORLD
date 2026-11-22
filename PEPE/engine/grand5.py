exec(open('grand3.py').read().split("\nM_=lambda")[0])
SP2=dict(SP);SP2['po']=[4,5]
def J4(m): return -m['mdd']-2000*(max(0,9200-m['net'])/500.+max(0,1.65-m['pf'])*10)
res=[]
for st in [dict(A,po=4),dict(A,po=4,psl=.05,pb=48,exo=1),dict(A,po=4,exo=1),dict(BC,mm=4.,md=3,mik=0,bflip=0)]:
    c=dict(st)
    for _ in range(3):
        for k,v in SP2.items():
            bb=(J4(ev(c)),c[k])
            for val in v:
                c2=dict(c);c2[k]=val;j=J4(ev(c2))
                if j>bb[0]: bb=(j,val)
            c[k]=bb[1]
    ms=[LM(runc(c,s)) for s in [0,1,2]];res.append((c,ms))
bs=[LM(runc(BC,s)) for s in [0,1,2]]
for c,ms in res:
    ok=all(m['net']>=b['net'] and m['pf']>=b['pf'] for m,b in zip(ms,bs))
    print({k:c[k] for k in SP if c[k]!=BC[k]});print('   '+' | '.join(f"s{s} {m['net']:.0f}/{m['pf']:.3f}/{m['mdd']:.0f}/n{m['n']}" for s,m in enumerate(ms))+f' | 3 gecikme ≥ baz: {ok}')
import pickle;pickle.dump(res,open('grand5.pkl','wb'));print('süre',round(time.time()-t0),'s')
