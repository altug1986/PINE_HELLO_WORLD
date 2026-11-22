import numpy as np, pandas as pd, itertools, sys, time
import base56b, v55
from base56b import A, _X, CFG
from v55 import evalp, run, simR, D, c, t
from volnorm import VT, d288, norm_blocks
from ent import g_ent
from vote import vote_pos
from rg55 import vote2, AGREE, ADX60
from met import SPLIT
med = np.nanmedian(VT); VTf = np.where(np.isnan(VT), med, VT); print('tipik 24s hareket medyanı %', round(med, 2))
W = np.array(CFG['w'] + [1.0], float); p1 = CFG['p'][1]; dd = np.nan_to_num(d288)
def ev(kL, kS, sk, lo=4.0, hi=12.0):
    if kL is None: X0 = _X
    else:
        s1 = np.where(dd <= -kL * VTf, 1, np.where(dd >= kS * VTf, -1, 0)).astype(np.int64); X0 = _X.copy(); X0[0] = g_ent(c, s1, int(p1['hold']), p1['sl'])
    sc = vote_pos(X0, W, CFG['on'], CFG['off'], 1)[1]; v55.TG = vote2(sc, ADX60, AGREE, 0.0, 25.0, 0.08, 0.08, 0.12, 0.08, 0.0)
    v55.STP = np.zeros(len(c)) if sk is None else np.clip(sk * VTf, lo, hi)
    e = evalp(A); tg, _ = run(A); R = simR(D, tg); b, v, bn = norm_blocks(R, 2); y, vy, yn = norm_blocks(R, 12)
    xt = t[R['xb']]; return dict(**e, nTot=round(bn.sum()), nIS=round(bn[bn.index < (2024 * 12 + 10) // 2].sum()), nStd=round(bn.std()), nMin=round(bn.min()), y26=round(y.iloc[-1]), y25=round(y.iloc[-2]))
rows = [dict(kL=None, kS=None, sk=None, **ev(None, None, None))]
G = list(itertools.product([None, 0.8, 1.0, 1.2, 1.4], [None, 1.5, 1.8, 2.1, 2.5], [None, 1.5, 2.0, 2.5, 3.0]))[int(sys.argv[1])::2]
for a, b_, sk in G:
    if (a is None) != (b_ is None): continue
    if a is None and sk is None: continue
    rows.append(dict(kL=a, kS=b_, sk=sk, **ev(a, b_, sk)))
pd.DataFrame(rows).to_pickle(f'vscan_{sys.argv[1]}.pkl'); print(len(rows))
