import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']; CAP = 500
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json'))
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
fw = {h: np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 for h in (60, 120, 240)}
def fast(e, t, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30; tt = t[(t.ei >= lo) & (t.ei < hi)]
    return x[-1], len(tt), (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
rows = []; W = {}
for setn, cols in [('P+X', [j for j, k in enumerate(names) if k[0] in 'PX']), ('P+X+R', list(range(len(names))))]:
    for hz in (60, 120, 240):
        w = np.zeros(len(names))
        for j in cols:   # ağırlık = IS'te oy yönündeki ortalama ileri getiri (bps) — yalnız IS
            v = M[:mid, j].astype(float); a = v != 0
            w[j] = np.nanmean((fw[hz][:mid] * v)[a]) if a.sum() > 500 else 0
        W[(setn, hz)] = w
        S = M.astype(float) @ w
        # kalibrasyon (OOS): skor dilimi → gerçekleşen ileri getiri
        if hz == 120:
            s_o = S[mid:]; f_o = fw[120][mid:]; bins = [-1e9, -30, -15, -8, 8, 15, 30, 1e9]
            cal = pd.Series(f_o).groupby(pd.cut(s_o, bins)).agg(['mean', 'size']); print(setn, 'OOS kalibrasyon h120 (skor bps → gerçek bps):'); print(cal.round(1).T.to_string())
        for Tin, H, s in itertools.product([8, 12, 16, 20, 25, 30, 40], [60, 120, 240], [4, 8]):
            eq, t = z02.run3(B, F, M, w, Tin, -1e9, s, H); e = eq.values
            if len(t) == 0: continue
            a = fast(e, t, 0, mid); b = fast(e, t, mid, len(e))
            rows.append(dict(set=setn, hz=hz, Tin=Tin, H=H, s=s, n=len(t), IS=a[0], ISc=a[2], OOS=b[0], OOSc=b[2], MaxDD=max(a[3], b[3]), bps=t.pnl.mean() / CAP * 1e4))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/vote3.csv', index=False)
np.save('/home/claude/zecwork/vote3_W.npy', np.array([W[k] for k in sorted(W)])); json.dump([list(k) for k in sorted(W)], open('/home/claude/zecwork/vote3_Wkeys.json', 'w'))
for setn in ('P+X', 'P+X+R'):
    r = R[R.set == setn].sort_values('ISc', ascending=False); print(f'== {setn} IS Calmar sıralı'); print(r.head(8).round(2).to_string(index=False))
    print('  IS>0 & OOS>0:', ((R.set == setn) & (R.IS > 0) & (R.OOS > 0)).sum(), '/', (R.set == setn).sum())
