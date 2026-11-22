import numpy as np, pandas as pd, itertools, v55
from short_top_lib import to5, zh, DIVh, q0, D, simR, run, evalp, m30, n, t, c
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24)))
v55.GL = ((a >= 1) | (b >= 1.5)).astype(np.int64); v55.GS = ((a <= -1) | (b <= -1.5)).astype(np.int64)      # V6.1 tabanı
def bt(p, lab):
    e = evalp(dict(q0, **p)); R = simR(D, run(dict(q0, **p))[0]); m = m30(R)
    return dict(t=lab, net=e['net'], n=e['n'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], MaxDD30=m['MaxDD 30dk mum içi'], Calmar30=m['Calmar 30dk'], DD30=m['DD P95/P98/P99 30dk'], UPI=m['Ulcer/UPI 30dk'])
