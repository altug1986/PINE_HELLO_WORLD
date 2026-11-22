import numpy as np, pandas as pd
from v55 import *
from eng import dmi, htf_map
z5 = np.load('d5.npz')
def ema(x, k): return pd.Series(x).ewm(span=k, adjust=False).mean().to_numpy()
B60, BK, BM = z5['b60'], z5['b60_k'], z5['b60_map']; X60, XK, XM = z5['x60'], z5['x60_k'], z5['x60_map']
def bat_e(k): return htf_map(ema(B60[3], k), BK, BM)
def btc_e(k): return htf_map(ema(X60[3], k), XK, XM)
_p, _m, _a = dmi(B60[1], B60[2], B60[3], 14, 14); ADX60 = np.nan_to_num(htf_map(_a, BK, BM)); DI60 = np.nan_to_num(htf_map(_p - _m, BK, BM))
if __name__ == '__main__':
    A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
    tg, why = run(A); R = simR(D, tg); X = pd.DataFrame(dict(eb=R['eb'], xb=R['xb'], dr=R['dir'], pnl=R['pnl']))
    i = X.eb.to_numpy(); s = X.dr.to_numpy()
    bt = np.sign(bat_e(5) - bat_e(17))[i]; xt = np.sign(btc_e(5) - btc_e(20))[i]
    X['rel'] = np.where((bt == xt) & (bt != 0), np.where(bt == s, 'trendle', 'TERS'), 'karışık'); X['adx'] = pd.cut(ADX60[i], [0, 20, 30, 100], labels=['<20', '20-30', '>30'])
    X['per'] = np.where(t[X.xb] >= t[-1] - 365 * 86400000, 'son12', np.where(t[X.xb] >= SPLIT, 'OOS-önce', 'IS'))
    g = X.groupby(['per', 'rel']).pnl.agg(['count', 'sum']).round(0); print(g.unstack(0).to_string())
    print(X[X.rel == 'TERS'].groupby(['per', 'adx'], observed=True).pnl.agg(['count', 'sum']).round(0).unstack(0).to_string())
    l = X[X.per == 'son12']; print('son12 yön × ilişki:'); print(l.groupby(['dr', 'rel']).pnl.agg(['count', 'sum']).round(0).unstack(0).to_string())
