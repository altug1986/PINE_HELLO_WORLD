import pickle, numpy as np, pandas as pd
from p1m import build, run, trades, TICK
D5 = build(5); D1 = pickle.load(open('D1.pkl', 'rb')); R, _ = trades(D5, run(D5)); t5 = D5['t']; c5 = D5['c']; t1 = D1['t']; c1 = D1['c']
for lim in (7.0, 5.0, 10.0):
    g = []; k = 0
    for eb, xb, d, ep, q, pl in R:
        a = np.searchsorted(t1, t5[eb] + 300000); b = np.searchsorted(t1, t5[xb] + 300000)     # girişten sonraki 1 dk mumlar → 5 dk çıkış mumu sonu
        x = (c1[a:b] / ep - 1) * d * 100; hit = np.flatnonzero(x <= -lim)
        if len(hit):
            k += 1; xp1 = c1[a + hit[0]] - TICK * d; xp5 = c5[xb] - TICK * d; g.append(d * (xp1 - xp5) * q)
    g = np.array(g); print(f'1 dk bekçi stop −%{lim:g}: tetiklenen işlem {k} | 5 dk çıkışına göre fark {g.sum():+.0f}$ (işlem başı {g.mean() if k else 0:+.2f}$, iyileşen %{(g > 0).mean()*100 if k else 0:.0f})')
