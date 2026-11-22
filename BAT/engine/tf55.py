import numpy as np, pandas as pd, itertools, sys, time
import v55
from v55 import evalp, n
from tr1h import bat_e, btc_e, ADX60
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
rows = []; t0 = time.time(); EB = {k: bat_e(k) for k in (5, 8, 12, 17, 20, 21, 26, 50)}; EX = {k: btc_e(k) for k in (5, 8, 12, 20, 21, 26, 50)}
for (bf, bs), xp, a, gm in itertools.product([(5, 17), (5, 20), (8, 21), (12, 26), (20, 50)], [None, (5, 20), (8, 21), (20, 50)], [0, 20, 25, 30, 35], [1, 2, 3, 4, 5]):
    if gm == 2 and (xp is None or a > 0): continue
    bt = np.nan_to_num(np.sign(EB[bf] - EB[bs])).astype(np.int64)
    if xp is None: ts = bt.copy()
    else:
        xt = np.nan_to_num(np.sign(EX[xp[0]] - EX[xp[1]])).astype(np.int64); ts = np.where(bt == xt, bt, 0)
    if a > 0 and gm != 2: ts = np.where(ADX60 >= a, ts, 0)
    v55.TSG = ts.astype(np.int64); e = evalp(dict(A, gm=gm)); rows.append(dict(bat=f'{bf}/{bs}', btc=str(xp), adx=a, gm=gm, **e))
d = pd.DataFrame(rows); d.to_pickle('tf55.pkl'); print(len(d), f'{time.time()-t0:.0f}s')
