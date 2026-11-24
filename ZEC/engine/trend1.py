import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
def don_state(rule, n_in, n_out):
    def f(H):
        h, l, c = H.high, H.low, H.close
        up = c > h.rolling(n_in).max().shift(1); dn = c < l.rolling(n_in).min().shift(1)
        xl = c < l.rolling(n_out).min().shift(1); xs = c > h.rolling(n_out).max().shift(1)
        st = np.zeros(len(c)); s = 0
        for i in range(len(c)):
            if up.iloc[i]: s = 1
            elif dn.iloc[i]: s = -1
            elif s == 1 and xl.iloc[i]: s = 0
            elif s == -1 and xs.iloc[i]: s = 0
            st[i] = s
        return st
    return np.nan_to_num(z02.htf(B, rule, f))
def ema_state(rule, a, b):
    return np.nan_to_num(z02.htf(B, rule, lambda H: np.sign(z02.ema(H.close.values, a) - z02.ema(H.close.values, b))))
def fast(e, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30
    return x[-1], (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
rows = []
cands = [(f'DON {r} {a}/{b}', lambda r=r, a=a, b=b: don_state(r, a, b)) for r, a, b in itertools.product(['4h', '1D'], [20, 40, 60], [10, 20])] + \
        [(f'EMA {r} {a}/{b}', lambda r=r, a=a, b=b: ema_state(r, a, b)) for r, a, b in itertools.product(['4h', '1D'], [10, 20], [50, 100])]
for lab, fn in cands:
    st = fn()
    for side in ('çift', 'long'):
        S = st if side == 'çift' else np.maximum(st, 0)
        eq, T = z02.sim3(F['o'], F['h'], F['l'], F['c'], F['atr'], S.astype(float), 1.0, 1.0, 1e6, 10**9)
        e = eq; a = fast(e, 0, mid); b = fast(e, mid, len(e)); f = fast(e, 0, len(e))
        rows.append(dict(cfg=lab, side=side, n=len(T), net=e[-1], pct=e[-1] / 5, IS=a[0], OOS=b[0], MaxDD=f[2], Cal=f[1], ISc=a[1], OOSc=b[1], best=T[:, 5].max() if len(T) else 0))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/trend1.csv', index=False)
print(R.sort_values('ISc', ascending=False).round(1).to_string(index=False))
