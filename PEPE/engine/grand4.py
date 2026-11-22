exec(open('grand3.py').read().split("\nM_=lambda")[0])
V={'A':A,'A po4':dict(A,po=4),'A PL baz':dict(A,po=4,psl=.05,pb=48),'A kilit açık':dict(A,exo=1),'A PL baz+kilit (OI dokunulmaz)':dict(A,po=4,psl=.05,pb=48,exo=1),
   'A OI baz + slc.03':dict(A,po=4,psl=.05,pb=48,exo=1,slc=.03),'A psl.08 kilit açık po4':dict(A,po=4,exo=1)}
for nm,c in V.items():
    ms=[LM(runc(c,s)) for s in [0,1,2]];bs=[LM(runc(BC,s)) for s in [0,1,2]]
    ok=all(m['net']>=b['net'] and m['pf']>=b['pf'] for m,b in zip(ms,bs))
    print(f"{nm:32s} "+' | '.join(f"s{s} {m['net']:.0f}/{m['pf']:.3f}/{m['mdd']:.0f}" for s,m in enumerate(ms))+f" | 3 gecikmede net&PF≥baz: {ok}")
print('BAZ                              '+' | '.join(f"s{s} {m['net']:.0f}/{m['pf']:.3f}/{m['mdd']:.0f}" for s,m in enumerate([LM(runc(BC,s)) for s in [0,1,2]])))
