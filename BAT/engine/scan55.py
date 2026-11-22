import sys, itertools, time, pandas as pd
from v55 import evalp
fam = sys.argv[1]; G = []
if fam == 'F1': G = [dict(eC=e, aE=a) for e, a in itertools.product([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], [0, 15, 20, 25, 30, 35, 40])]
if fam == 'F2': G = [dict(H=h, xH=x, rH=r) for h, x, r in itertools.product([144, 216, 288, 360, 432, 576], [-3.0, -2.0, -1.0, 0.0, 1.0], [2.0, 4.0, 99.0])]
if fam == 'F3': G = [dict(aT=a, dT=d, sT=s) for a, d, s in itertools.product([25, 30, 35, 40, 45, 50], [0, 10, 20, 30], [1.0, 2.0, 3.0, 4.0, 5.0])]
if fam == 'F4': G = [dict(bvk=b, mo=m, mk=k, sB=s) for b, m, k, s in itertools.product([1.5, 2.0, 3.0, 4.0], [12, 48, 144, 288], [1.0, 2.0, 3.0, 5.0, 8.0], [0.0, 2.0, 4.0])]
if fam == 'F5': G = [dict(nrMin=a, vrMin=b) for a, b in itertools.product([0, 0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.3]) if a + b > 0]
if fam == 'F6':
    for hs in [[15], [4, 5], [4, 5, 15], [15, 20], [4, 5, 12, 15, 20], [4], [5], [20], [12], [9, 10]]: G.append(dict(hm=sum(1 << h for h in hs), hl=str(hs)))
rows = []; t0 = time.time()
for p in G:
    pp = {k: v for k, v in p.items() if k != 'hl'}; rows.append(dict(**p, **evalp(pp)))
pd.DataFrame(rows).to_pickle(f's55_{fam}.pkl'); print(fam, len(rows), f'{time.time()-t0:.0f}s')
