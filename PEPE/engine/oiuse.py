exec(open('grand3.py').read().split("\nM_=lambda")[0])
A4=dict(A,po=4)
for nm,c in [('V11.5/V11.6 çekirdeği (OI: likidasyon dönüşü)',A4),('OI tamamen kapalı (likidasyon motoru yok)',dict(A4,pl=0)),('+ OI tükeniş kilidi geri açık',dict(A4,exo=1))]:
    print(nm.ljust(46),' | '.join(f"s{s} {m['net']:.0f}/PF {m['pf']:.3f}/MaxDD {m['mdd']:.0f}/n{m['n']}" for s,m in enumerate([LM(runc(c,s)) for s in [0,1,2]])))
pl=plf(0,4,4,.08,72,55);r=runc(A4);te=r[1];src=pl[te]!=0
print(f"likidasyon motorunun açtığı işlem {int(src.sum())} | net {r[4][src].sum():.0f}$ | PF {r[4][src&(r[4]>0)].sum()/-r[4][src&(r[4]<=0)].sum():.2f} | sistemin %{100*src.mean():.1f} işlemi")
