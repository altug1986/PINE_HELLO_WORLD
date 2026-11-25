import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z01
from zec_data import load
B = load(1, src='/home/claude/master/'); o, h, l, c = (B[k].values.astype(float) for k in ('open', 'high', 'low', 'close')); n = len(c)
oi = B.oi_zec.values.astype(float); yr = B.index.year.values; mid = n // 2; COST = 8.3
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
def events(X, W):
    r = (c / sh(c, W) - 1) * 100; i = np.where(r >= X)[0]; ev = []; last = -10**9
    for j in i:
        if j - last > 360: ev.append(j); last = j
    return np.array(ev)
def trade(en, stop, H):
    """short: giriş en kapanışı; zarar kes = stop (bar içi), süre H bar. Döner: pnl bps net, MAE %, MFE %."""
    ep = c[en] - 0.01; worst = 0; best = 0
    for j in range(en + 1, min(n, en + H + 1)):
        worst = max(worst, (h[j] - ep) / ep); best = max(best, (ep - l[j]) / ep)
        if h[j] >= stop:
            xp = max(o[j], stop) + 0.01; return (ep - xp) / ep * 1e4 - COST, worst * 100, best * 100, j
    xp = c[min(n - 1, en + H)] + 0.01; return (ep - xp) / ep * 1e4 - COST, worst * 100, best * 100, en + H
rows = []
for X, W, L, dd, buf, H in itertools.product([5, 7], [10, 30], [3, 5, 10], [0.0, 1.0, 2.0], [0.0, 0.5], [60, 120, 240]):
    for e in events(X, W):
        if e < 3000 or e + 600 >= n: continue
        pk = h[e]; en = -1
        for j in range(e + 1, e + 241):                         # NEDENSEL: tepe = şimdiye kadarki en yüksek
            pk = max(pk, h[j]); lo_ = l[j - L:j].min()
            if c[j] < lo_ and (pk - c[j]) / pk * 100 >= dd:
                en = j; break
        if en < 0: continue
        stop = pk * (1 + buf / 100)
        pnl, mae, mfe, xi = trade(en, stop, H)
        doi = (oi[en] / oi[e - W] - 1) * 100 if oi[en] > 0 and oi[e - W] > 0 else np.nan
        rows.append(dict(X=X, W=W, L=L, dd=dd, buf=buf, H=H, en=en, t=B.index[en], IS=en < mid, y=yr[en], pnl=pnl, mae=mae, mfe=mfe, doi=doi))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/blow2.csv', index=False)
G = R.groupby(['X', 'W', 'L', 'dd', 'buf', 'H'])
S = G.agg(n=('pnl', 'size'), bps=('pnl', 'mean'), IS=('pnl', lambda s: s[R.loc[s.index, 'IS']].mean()), OOS=('pnl', lambda s: s[~R.loc[s.index, 'IS']].mean()),
          kaz=('pnl', lambda s: (s > 0).mean() * 100), maeP90=('mae', lambda s: np.percentile(s, 90)), maeMax=('mae', 'max'),
          yıl=('pnl', lambda s: int((R.loc[s.index].groupby('y').pnl.sum() > 0).sum())), net=('pnl', lambda s: s.sum() * 0.05)).reset_index()
pd.set_option('display.width', 220)
print('NEDENSEL kural, short; net $ = 500 $ ile toplam (bps × 0,05)')
print('IS>0 & OOS>0 oranı %', round(((S.IS > 0) & (S.OOS > 0)).mean() * 100, 1), 'toplam', len(S))
print(S.sort_values('bps', ascending=False).head(15).round(1).to_string(index=False))
print(S.groupby('L')[['bps', 'IS', 'OOS']].median().round(1).T.to_string()); print(S.groupby('dd')[['bps', 'IS', 'OOS']].median().round(1).T.to_string())
