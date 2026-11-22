import numpy as np, v55
from hatatrend import ev, vote_x, SC, ADX60, AGREE, SH, STT, Z, GLb, GSb, q0, TS, simR, run, D
v55.TG = vote_x(SC, ADX60, AGREE, SH, Z, STT, Z, 20.0, 0.04, 0.15, 0.12, -0.15); v55.GL = GLb.astype(np.int64); v55.GS = GSb.astype(np.int64)
tg, _ = run(dict(q0, **TS)); R = simR(D, tg); c = D['c']; print('pozisyonsuz zaman %', round((tg == 0).mean() * 100, 1))
eb, xb, dr, ep, pl = R['eb'], R['xb'], R['dir'], R['ep'], R['pnl']
for x in (2.0, 3.0, 5.0):
    hit = np.array([((c[e:min(e + 48, b) + 1] / p - 1) * d * 100).min() <= -x for e, b, d, p in zip(eb, xb, dr, ep)])
    print(f'ilk 4 saatte −%{x:g} görenler: {hit.sum()} işlem | kârla kapanan %{(pl[hit] > 0).mean()*100:.0f} | toplam {pl[hit].sum():.0f}$ | ortalama {pl[hit].mean():.2f}$')
