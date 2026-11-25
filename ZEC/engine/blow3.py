import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z01
from zec_data import load
B = load(1, src='/home/claude/master/'); o, h, l, c = (B[k].values.astype(float) for k in ('open', 'high', 'low', 'close')); n = len(c)
c1 = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1))); tr[0] = h[0] - l[0]; atr = z01.rma(tr, 60)
yr = B.index.year.values; mid = n // 2; COST = 8.3
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
print('T1 — sıçrama tespitinden SONRA fiyat (long yönünde, maliyetsiz bps; IS/OOS): short açıksa kalmanın bedeli')
for X, W in itertools.product([2, 3, 5], [5, 10]):
    r = (c / sh(c, W) - 1) * 100; i = np.where((r >= X) & (sh(r, 1) < X))[0]; i = i[(i > 3000) & (i < n - 300)]
    out = []
    for H in (5, 15, 60, 240):
        f = (c[i + H] / c[i] - 1) * 1e4; out.append(f'h{H} {f[i < mid].mean():6.1f}/{f[i >= mid].mean():6.1f}')
    mx = np.array([(h[j + 1:j + 61].max() / c[j] - 1) * 100 for j in i])
    print(f'  {W:2d} barda ≥%{X}: n {len(i):4d} | ' + ' | '.join(out) + f' | sonraki 60 barda ek yükseliş P50/P90 %{np.median(mx):.1f}/%{np.percentile(mx, 90):.1f}')
print('\nT2 — sıçramada LONG + sıkı iz (tepe − k·ATR), süre 240 bar; maliyet sonrası')
for X, W, k in itertools.product([2, 3, 5], [5, 10], [3, 5, 8, 12]):
    r = (c / sh(c, W) - 1) * 100; i = np.where((r >= X) & (sh(r, 1) < X))[0]; i = i[(i > 3000) & (i < n - 300)]
    pn = []; last = -1
    for e in i:
        if e <= last: continue
        ep = c[e] + 0.01; pk = c[e]; stop = c[e] - k * atr[e]; xp = None
        for j in range(e + 1, min(n, e + 241)):
            if l[j] <= stop: xp = min(o[j], stop) - 0.01; break
            pk = max(pk, c[j]); stop = max(stop, pk - k * atr[j])
        if xp is None: xp = c[min(n - 1, e + 240)] - 0.01; j = e + 240
        pn.append(((xp - ep) / ep * 1e4 - COST, e < mid, yr[e])); last = j
    P = pd.DataFrame(pn, columns=['b', 'IS', 'y'])
    print(f'  {W:2d} barda ≥%{X}, iz {k:2d} ATR: n {len(P):4d} ort {P.b.mean():6.1f} bps | IS {P.b[P.IS].mean():6.1f} OOS {P.b[~P.IS].mean():6.1f} | net ${P.b.sum()*0.05:6.0f} | pozitif yıl {(P.groupby("y").b.sum()>0).sum()}/6')
