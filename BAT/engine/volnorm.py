import numpy as np, pandas as pd, itertools, sys, time
import base56b, v55
from base56b import A, _X, CFG, SCQ
from v55 import evalp, run, simR, D, c, t, n
from ent import g_ent
from vote import vote_pos
from rg55 import vote2, AGREE, ADX60
d288 = np.r_[np.full(288, np.nan), (c[288:] / c[:-288] - 1) * 100]
VT = pd.Series(np.abs(d288)).rolling(8640, min_periods=2000).median().to_numpy()       # son 30 günün tipik 24s hareketi (%)
day = pd.Series(c, index=pd.to_datetime(t, unit='ms')).resample('D').last(); dr = day.pct_change().abs() * 100
def norm_blocks(R, months=2):
    x = pd.to_datetime(t[R['xb']], unit='ms'); k = ((x.year * 12 + x.month - 1) // months).values; b = pd.Series(R['pnl']).groupby(k).sum()
    dk = ((dr.index.year * 12 + dr.index.month - 1) // months); v = dr.groupby(dk).mean().reindex(b.index); return b, v, b * (dr.mean() / v)
if __name__ == '__main__':
    tg, _ = run(A); R = simR(D, tg); b, v, bn = norm_blocks(R, 12)
    print('V5.6B yıl: net $ | BAT ort. günlük hareket % | hareket-normalize $')
    for k in b.index: print(f"  {k//1}: {b[k]:7.0f} | {v[k]:5.2f} | {bn[k]:7.0f}")
    b2, v2, n2 = norm_blocks(R, 2); print('2 aylık: ham std', round(b2.std()), 'normalize std', round(n2.std()), '| ham min', round(b2.min()), 'norm min', round(n2.min()), '| norm toplam', round(n2.sum()))
    # comp1 eşdeğerlik ve oynaklığa ölçekli eşik
    p1 = CFG['p'][1]; s0 = np.where(np.nan_to_num(d288) <= -p1['aL'], 1, np.where(np.nan_to_num(d288) >= p1['aS'], -1, 0)).astype(np.int64)
    print('comp1 eşdeğer mi:', (g_ent(c, s0, int(p1['hold']), p1['sl']) == _X[0]).mean(), p1)
