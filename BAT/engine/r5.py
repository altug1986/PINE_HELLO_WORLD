import numpy as np, numba as nb, pandas as pd, itertools
from sysB import *
@nb.njit(cache=True)
def g_lk(c, sig, hold, sl):
    # gerçek stop: stop olunca aynı yön sinyali sönene kadar yeniden giriş yok
    L = len(c); tg = np.zeros(L, np.int64); v = 0; e = 0.0; vb = 0; lock = 0
    for i in range(L):
        s = sig[i]
        if lock != 0 and s != lock: lock = 0
        if v != 0:
            if (c[i] / e - 1) * v <= -sl / 100: lock = v; v = 0
            elif i - vb >= hold: v = 0
        if s != 0 and s != lock:
            if v != s: v = s; e = c[i]; vb = i
            else: vb = i
        tg[i] = v
    return tg
def mae(D, R):
    w = [(np.min(D['l'][a:b + 1]) / D['c'][a] - 1) * 100 if s > 0 else -(np.max(D['h'][a:b + 1]) / D['c'][a] - 1) * 100 for a, b, s in zip(R['eb'], R['xb'], R['dir'])]
    return np.percentile(w, 5).round(1), np.percentile(w, 1).round(1), round(min(w), 1)
if __name__ == "__main__":
  D0 = load(5); F0 = feats(D0); DI = cut(D0, SPLIT); FI = feats(DI)
  V3 = dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)
  for lab, D, F in [('IS', DI, FI), ('TAM', D0, F0)]:
      s = sig2(F, D, V3)
      for sl in [99, 5, 8, 10, 12, 15, 20, 30]:
          R = simR(D, g_lk(D['c'], s, 648, sl)); m = metrics(D, R)
          print(lab, 'stop', sl, {k: m[k] for k in ['net', 'n', 'pf', 'p98', 'mddc', 'mddi', 'IS', 'OOS', 'last12']}, 'MAE P95/P99/min', mae(D, R), flush=True)
