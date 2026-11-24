import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); CAP = 500
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
def fast(e, t, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30; tt = t[(t.ei >= lo) & (t.ei < hi)]
    return x[-1], len(tt), (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
sets = {'AŞIRI6': ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis'],
        'AŞIRI6+4h': ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis', 'P04_htf4h'],
        'AŞIRI6+glb': ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis', 'R21_glb']}
rows = []
for sn, cols in sets.items():
    w = np.zeros(len(names)); w[[ix[k] for k in cols]] = 1      # işaretler IS bilgi katsayısıyla aynı (hepsi +)
    for Tin, H, s in itertools.product([1, 2, 3], [60, 120, 240, 480], [4, 8, 12]):
        if sn == 'AŞIRI6+4h' and Tin == 1: continue            # 4h oyu tek başına giriş açmasın
        eq, t = z02.run3(B, F, M, w, Tin, -1e9, s, H); e = eq.values
        if len(t) < 30: continue
        a = fast(e, t, 0, mid); b = fast(e, t, mid, len(e))
        rows.append(dict(set=sn, Tin=Tin, H=H, s=s, n=len(t), IS=a[0], ISc=a[2], OOS=b[0], OOSc=b[2], MaxDD=max(a[3], b[3]), bps=t.pnl.mean() / CAP * 1e4,
                         L=t[t.side > 0].pnl.sum(), S=t[t.side < 0].pnl.sum()))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/vote4.csv', index=False)
print(R.sort_values('ISc', ascending=False).head(12).round(1).to_string(index=False))
print('IS>0&OOS>0:', ((R.IS > 0) & (R.OOS > 0)).sum(), '/', len(R))
