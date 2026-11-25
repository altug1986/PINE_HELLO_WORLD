import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z01
from zec_data import load
from path_quality import path
B = load(1, src='/home/claude/master/'); o, h, l, c, v = (B[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')); n = len(c)
c1 = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1))); tr[0] = h[0] - l[0]; atr = z01.rma(tr, 60)
oi = B.oi_zec.values.astype(float); yr = B.index.year.values; mid = n // 2
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
COST = 8.3
def events(X, W):
    r = (c / sh(c, W) - 1) * 100; i = np.where(r >= X)[0]; ev = []; last = -10**9
    for j in i:
        if j - last > 360: ev.append(j); last = j
    return np.array(ev)
rows = []
for X, W, L, G in itertools.product([5, 7, 10], [10, 30], [5, 10, 20], [120, 240]):
    ev = events(X, W)
    for e in ev:
        if e + G + 600 >= n or e < 3000: continue
        # tepe: olaydan sonraki G bar içindeki en yüksek; kırılım: tepeden sonra kapanış < son L barın dibi
        seg = h[e:e + G]; pk = e + int(np.argmax(seg)); ll = pd.Series(l[pk:pk + G + L]).rolling(L).min().shift(1).values
        brk = np.where(c[pk:pk + G + L] < ll)[0]
        if len(brk) == 0: continue
        en = pk + brk[0]
        doi = (oi[pk] / oi[max(e - W, 0)] - 1) * 100 if not np.isnan(oi[pk]) and not np.isnan(oi[max(e - W, 0)]) else np.nan
        fw = {H: (1 - c[en + H] / c[en]) * 1e4 - COST for H in (30, 120, 480)}
        rows.append(dict(X=X, W=W, L=L, G=G, ei=en, t=B.index[en], y=yr[en], IS=en < mid, doi=doi, gecik=en - e, pk_pct=(h[pk] / c[e - W] - 1) * 100, **{f'h{k}': v_ for k, v_ in fw.items()}))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/blow1.csv', index=False)
R['OI'] = np.where(R.doi.isna(), 'yok', np.where(R.doi < 0, 'sıkışma (OI↓)', 'yeni para (OI↑)'))
g = R.groupby(['X', 'W', 'L', 'OI'])
S = g.agg(n=('h120', 'size'), h30=('h30', 'mean'), h120=('h120', 'mean'), h480=('h480', 'mean'),
          h120IS=('h120', lambda s: s[R.loc[s.index, 'IS']].mean()), h120OOS=('h120', lambda s: s[~R.loc[s.index, 'IS']].mean()),
          kaz=('h120', lambda s: (s > 0).mean() * 100)).reset_index()
S = S[S.G.isna()] if 'G' in S else S
pd.set_option('display.width', 220)
print(S[(S.n >= 15) & (S.OI != 'yok')].sort_values(['X', 'W', 'L', 'OI']).round(1).to_string(index=False))
