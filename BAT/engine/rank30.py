import numpy as np, pandas as pd, sys, time
import v55
from search57 import build, SP
from v55 import run, simR, D, t
from mtm30 import m30
d = pd.concat([pd.read_pickle(f's57_{i}.pkl') for i in (1, 2)], ignore_index=True)
P = list(SP); q = d[(d.net >= 7500) & (d.OOS >= 1950)].drop_duplicates(subset=P).sort_values('sc', ascending=False).head(500)
base = dict(x11=2.5, on=0.08, onT=0.12, aT=25.0, S=7.0, rA=4.0, H=576, xH=-2.0, aH=20.0, hm=48, nrMin=0.1, g='yok', gtf='1g', gw=1.0, gth=50, gsym=0, zone=0)
rows = []; t0 = time.time(); L = [base] + [r[P].to_dict() for _, r in q.iterrows()]
for p in L[int(sys.argv[1])::2]:
    qq = build(p); tg, _ = run(qq); R = simR(D, tg); m = m30(R); xt = t[R['xb']]
    m['L200'] = round(R['pnl'][xt >= t[-1] - 200 * 864e5].sum()); m['OOS'] = round(R['pnl'][xt >= 1730419200000].sum()); m['n'] = len(R['pnl'])
    u = m['DD P95/P98/P99 30dk'].split('/'); m['p95'], m['p98'], m['p99'] = int(u[0]), int(u[1]), int(u[2]); rows.append(dict(**p, **m))
pd.DataFrame(rows).to_pickle(f'rank30_{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
