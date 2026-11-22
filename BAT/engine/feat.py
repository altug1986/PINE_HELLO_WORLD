import numpy as np, pandas as pd
from v54 import D, BADX, BDI, NR, c, t
from eng import dmi, htf_map
z = np.load('d5.npz'); n = len(c)
def htf(tfms):
    k = t // tfms; hk, st = np.unique(k, return_index=True); en = np.r_[st[1:], n]
    o = D['o'][st]; h = np.maximum.reduceat(D['h'], st); l = np.minimum.reduceat(D['l'], st); cc = c[en - 1]
    return hk, k, o, h, l, cc
hk15, k15, o15, h15, l15, c15 = htf(900000)
p, m, a = dmi(h15, l15, c15, 14, 14)
ADX15 = htf_map(a, hk15, k15); DI15 = htf_map(p - m, hk15, k15)
def chg(x, k): return np.r_[np.full(k, np.nan), (x[k:] / x[:-k] - 1) * 100]
d288 = chg(c, 288); m864 = chg(c, 864); m2016 = chg(c, 2016); xc = z['xc']; b288 = chg(xc, 288); b72 = chg(xc, 72)
lr = np.r_[0, np.diff(np.log(c))]; VOL = pd.Series(lr).rolling(288).std().to_numpy() * np.sqrt(288) * 100
VOLr = pd.Series(VOL).rolling(8640, min_periods=2000).rank(pct=True).to_numpy()   # 30 günlük göreli oynaklık
F = dict(d288=d288, m864=m864, m2016=m2016, adx15=ADX15, di15=DI15, badx=BADX, bdi=BDI, nr=NR, vol=VOL, volr=VOLr, b288=b288, b72=b72)
