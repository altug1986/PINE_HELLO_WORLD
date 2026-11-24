import sys, time, itertools, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load, master; import z01, z02
t0 = time.time()
X1 = master('/home/claude/master/')[['open', 'close', 'volume']]
B = load(2, src='/home/claude/master/'); names, M, F = z02.votes(B, X1); del X1
np.save('/home/claude/zecwork/votes.npy', M); json.dump(names, open('/home/claude/zecwork/vote_names.json', 'w'))
print('oylar hazır', round(time.time() - t0), 's', names)
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
def fast(eq, t, lo, hi):
    e = eq[lo:hi] - eq[lo]; e30 = e[::15]; dd = np.maximum.accumulate(e30) - e30
    tt = t[(t.ei >= lo) & (t.ei < hi)]; yrs = (hi - lo) / (720 * 365)
    return e[-1], len(tt), (e[-1] / yrs) / dd.max() if dd.max() > 0 else 0, dd.max()
P = [i for i, k in enumerate(names) if k[0] in 'PX']
w0 = np.zeros(len(names)); w0[P] = 1
rows = []
for Tin, Tout in itertools.product([2, 3, 4, 5, 6, 7, 8], [-1, 0, 1, 2, 3]):
    if Tout >= Tin: continue
    eq, t = z02.run3(B, F, M, w0, Tin, Tout); e = eq.values
    a = fast(e, t, 0, mid); b = fast(e, t, mid, len(e))
    rows.append(dict(Tin=Tin, Tout=Tout, n=len(t), net=e[-1], IS=a[0], ISc=a[2], OOS=b[0], OOSc=b[2], MaxDD=max(a[3], b[3]), bps=t.pnl.mean() / 5))
R = pd.DataFrame(rows); print('HAM (P+X eşit ağırlık, 17 bileşen):'); print(R.round(2).to_string(index=False))
# bileşen katkısı: ham en çok işlem üreten makul ayarda (Tin=3, Tout=0)
eq, t = z02.run3(B, F, M, w0, 3, 0); i = t.ei.astype(int).values; isIS = i < mid
print('\nKATKI (Tin3/Tout0 ham, n=%d, net %.0f): bileşen işlem yönüyle aynı / ters oyda ortalama bps (IS | OOS)' % (len(t), t.pnl.sum()))
for j, k in enumerate(names):
    ag = M[i, j] * t.side.values
    out = []
    for msk in (isIS, ~isIS):
        p = t.pnl.values[msk] / 5; g = ag[msk]
        out.append(f'aynı {p[g > 0].mean():6.1f} ({(g > 0).sum():5d}) ters {p[g < 0].mean():6.1f} ({(g < 0).sum():5d})')
    print(f'{k:11s} IS: {out[0]} | OOS: {out[1]}')
t.assign(**{k: M[i, j] * t.side.values for j, k in enumerate(names)}).to_csv('/home/claude/zecwork/raw_trades_T3.csv', index=False)
