import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json'))
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
f120 = np.r_[c[120:] / c[:-120] - 1, np.full(120, np.nan)] * 1e4
CAP = 500
def isw(cols, thr):   # SEÇİM yalnız IS: işaret = IS bilgi katsayısı (h=120) işareti, |IC| < thr ise 0
    w = np.zeros(len(names))
    for j in cols:
        v = M[:mid, j].astype(float); a = v != 0; ic = np.nanmean((f120[:mid] * v)[a]) if a.sum() > 500 else 0
        w[j] = np.sign(ic) if abs(ic) >= thr else 0
    return w
def fast(e, t, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30; tt = t[(t.ei >= lo) & (t.ei < hi)]
    return x[-1], len(tt), (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
rows = []
for setn, cols in [('P+X', [j for j, k in enumerate(names) if k[0] in 'PX']), ('P+X+R', list(range(len(names))))]:
    for thr in (1.0, 3.0):
        w = isw(cols, thr)
        for Tin, H, s in itertools.product([1, 2, 3, 4, 5, 6], [30, 60, 120, 240], [4, 8]):
            eq, t = z02.run3(B, F, M, w, Tin, -99, s, H); e = eq.values
            a = fast(e, t, 0, mid); b = fast(e, t, mid, len(e))
            rows.append(dict(set=setn, thr=thr, k=int((w != 0).sum()), Tin=Tin, H=H, s=s, n=len(t), IS=a[0], ISc=a[2], OOS=b[0], OOSc=b[2], MaxDD=max(a[3], b[3]),
                             bps=t.pnl.mean() / CAP * 1e4 if len(t) else 0))
        print(setn, thr, {names[j]: int(w[j]) for j in cols if w[j] != 0})
CAP = 500
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/vote2.csv', index=False)
for setn in ('P+X', 'P+X+R'):
    r = R[(R.set == setn) & (R.n >= 1500)].sort_values('ISc', ascending=False)
    print(f'== {setn} (işlem ≥1500) IS Calmar sıralı'); print(r.head(8).round(2).to_string(index=False))
