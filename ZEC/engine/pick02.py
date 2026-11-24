import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c, a = F['c'], F['atr']
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
def sig(bs, k):
    L = np.zeros(len(c), bool); S = np.zeros(len(c), bool)
    for b in bs:
        l, s = z01.fade_sig(F, b, k); L |= l; S |= s
    return L & ~S, S & ~L
rows = []
for bs, k in [((20,), 10), ((30,), 10), ((10, 20, 30), 10), ((10, 20, 30), 11), ((10, 20, 30), 9)]:
    L, S = sig(bs, k)
    for s, m, H in itertools.product([4, 6, 10], [6, 10, 99], [60, 120, 240]):
        eq, T = z01.sim2(F['o'], F['h'], F['l'], c, a, L, S, s, m, H, np.ones(len(c)))
        t = pd.DataFrame(T, columns=['ei','xi','side','ep','xp','pnl','mae','own']); t['et'] = B.index[t.ei.astype(int)]
        e = pd.Series(eq, index=B.index)
        A = z01.metrics(e, t, None, mid); O = z01.metrics(e, t, mid, None); f = z01.metrics(e, t)
        rows.append(dict(bs=str(bs), k=k, s=s, m=m, H=H, IS=A['net'], ISc=A['Calmar'], OOS=O['net'], OOSc=O['Calmar'], net=f['net'], n=f['n'], PF=f['PF'],
                         MaxDD=f['MaxDD'], Cal=f['Calmar'], L=t[t.side>0].pnl.sum(), S=t[t.side<0].pnl.sum()))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/pick02.csv', index=False)
print(R.sort_values('ISc', ascending=False).head(15).round(2).to_string(index=False))
