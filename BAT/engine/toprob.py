import numpy as np, pandas as pd, sys
import v55
from search57 import build
from v55 import evalp, run, simR, D, t, n
from topfeat import feats
from mtm30 import m30
K = dict(x11=2.5, on=0.04, onT=0.15, aT=20.0, S=7.0, rA=4.0, H=648, xH=-1.0, aH=20.0, hm=0, nrMin=0.1, g='yok', gtf='12s', gw=0.5, gth=40, gsym=1, zone=0)
q0 = build(K); rows = []; Fs = {lag: feats(lag) for lag in (1, 0, 2, 3)}
def go(lab, f='zDIV', N=8640, k=1.0, lag=1, on=True):
    if on:
        x = np.nan_to_num(Fs[lag][f'{f}{N}']) if f'{f}{N}' in Fs[lag] else None
        v55.GL = (x >= k).astype(np.int64); v55.GS = (x <= -k).astype(np.int64)
    else: v55.GL = np.zeros(n, np.int64); v55.GS = np.zeros(n, np.int64)
    e = evalp(q0); tg, _ = run(q0); R = simR(D, tg); m = m30(R)
    rows.append(dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], P200=e['P200'], MaxDD30=m['MaxDD 30dk mum içi'], Calmar30=m['Calmar 30dk'], DD30=m['DD P95/P98/P99 30dk'], UPI=m['Ulcer/UPI 30dk'], uzun=m['en uzun zirve altı (gün)']))
go('BAZ K (top trader yok)', on=False); go('ADAY: zDIV8640 bizimle ≥1.0')
for k in (0.75, 1.25, 1.5): go(f'k={k}', k=k)
for N in (2016,): go(f'N={N}', N=N)
for lag in (0, 2, 3): go(f'gecikme {lag} bar', lag=lag)
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
