import numpy as np, pandas as pd, numba as nb
import base56b, v55
from base56b import A, SCQ
from v55 import evalp, run, simR, c, t, n, D
from rg55 import AGREE, ADX60
from met import SPLIT
NAMES = ['DÜŞÜŞ', 'YATAY-DÜŞÜŞ', 'YATAY', 'YATAY-YÜKSELİŞ', 'YÜKSELİŞ']
mk = (pd.to_datetime(t, unit='ms').year * 12 + pd.to_datetime(t, unit='ms').month - 1).to_numpy()
um, st = np.unique(mk, return_index=True); en = np.r_[st[1:], n]; mret = (c[en - 1] / np.r_[c[0], c[en - 1][:-1]] - 1) * 100   # ay kapanış-kapanış
def classify(r, a=5.0, b=15.0): return np.select([r <= -b, r <= -a, r < a, r < b], [0, 1, 2, 3], 4)
def reg_bars(a=5.0, b=15.0):
    rm = np.r_[2, classify(mret[:-1], a, b)]          # ayın rejimi = BİR ÖNCEKİ ayın getirisi
    return rm[np.searchsorted(um, mk)], rm
@nb.njit(cache=True)
def vote3(sc, adx, ag, aT, on, onT, rg, addL, addS):
    n = len(sc); tg = np.zeros(n, np.int64); pos = 0
    for i in range(n):
        s = sc[i]; want = pos
        if pos != 0 and s * pos < 0.0: want = 0
        oL = (onT if (ag[i] == -1 and adx[i] >= aT) else on) + addL[rg[i]]
        oS = (onT if (ag[i] == 1 and adx[i] >= aT) else on) + addS[rg[i]]
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        pos = want; tg[i] = pos
    return tg
