import sys, itertools, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); n = len(B)
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
def htfx(rule, fn): return z02.htf(B, rule, fn)
def atr4(H, k=14):
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs()))
    return tr.ewm(alpha=1 / k, adjust=False).mean().values
A4 = np.nan_to_num(htfx('4h', atr4), nan=np.nan); A1 = htfx('1D', atr4)
def don_entry(rule, N):
    hh = htfx(rule, lambda H: H.high.rolling(N).max().values); ll = htfx(rule, lambda H: H.low.rolling(N).min().values)
    c = F['c']; return (c > hh), (c < ll)      # 2 dk kapanış, kapanmış HTF kanalını aşınca
def fast(e, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30
    return x[-1], (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
z = np.zeros(n, np.bool_); rows = []
for rule, N in itertools.product(['4h', '1D'], [10, 20, 40, 60]):
    eL, eS = don_entry(rule, N); eL = np.nan_to_num(eL).astype(np.bool_); eS = np.nan_to_num(eS).astype(np.bool_)
    for atrsrc, k in itertools.product(['4h', '1D'], [2, 3, 4, 6, 8]):
        D = (A4 if atrsrc == '4h' else A1) * k
        for side in ('çift', 'long'):
            es = eS if side == 'çift' else z
            eq, T = z02.sim4(F['o'], F['h'], F['l'], F['c'], F['atr'], eL, es, D, z, z, 4.0, 120, 99.0)
            a = fast(eq, 0, mid); b = fast(eq, mid, n); f = fast(eq, 0, n)
            t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
            top = t.pnl.nlargest(3).sum() / max(t.pnl.sum(), 1)
            rows.append(dict(entry=f'DON {rule} {N}', exit=f'{k}xATR{atrsrc}', side=side, n=len(t), net=eq[-1], IS=a[0], OOS=b[0], MaxDD=f[2], Cal=f[1], ISc=a[1], OOSc=b[1], top3=top))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/trend2.csv', index=False)
print(R.sort_values('net', ascending=False).head(15).round(2).to_string(index=False))
print(R[(R.IS > 0) & (R.OOS > 0)].sort_values('Cal', ascending=False).head(10).round(2).to_string(index=False))
