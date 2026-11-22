import numpy as np, pandas as pd, v55
from tvcmp63 import Mdf
from hatatrend import vote_x, SC, ADX60, AGREE, SH, STT, Z, q0, TS, simR, run, D
from stlib import GL0, GS0
v55.TG = vote_x(SC, ADX60, AGREE, SH, Z, STT, Z, 20.0, 0.04, 0.15, 0.12, -0.15); v55.GL = GL0.astype(np.int64); v55.GS = GS0.astype(np.int64)
R = simR(D, run(dict(q0, **TS))[0]); t = D['t']
m = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms', 'high', 'low', 'close']); T1 = m.open_time_ms.to_numpy(); H = m.high.to_numpy(); L = m.low.to_numpy(); C = m.close.to_numpy()
et = t[R['eb']] + 300000; dr = R['dir']; ep = R['ep']; j0 = np.searchsorted(T1, et)
print(f"işlem {len(dr)} | ort net/işlem {R['pnl'].mean():.2f}$ | 1 tick / fiyat (son) %{1e-5 / C[-1] * 100:.3f} | gidiş-dönüş maliyet ≈ %{0.08 + 2e-5 / C[-1] * 100:.3f}")
for k in (5, 15, 30):
    fav = np.array([((L[a:a + k].min() if d > 0 else H[a:a + k].max()) / p - 1) * -d * 100 for a, d, p in zip(j0, dr, ep)])   # lehine geri çekilme %
    print(f'ilk {k} dk lehine geri çekilme: medyan %{np.median(fav):.2f}, ortalama %{fav.mean():.2f}')
print('limit emir testi (15 dk bekle, dolmazsa 15. dk kapanışından piyasa):')
for x in (0.1, 0.2, 0.3, 0.5, 1.0):
    g = []
    for a, d, p in zip(j0, dr, ep):
        lim = p * (1 - x / 100 * d); w = slice(a, a + 15)
        hit = (L[w] <= lim).any() if d > 0 else (H[w] >= lim).any()
        g.append(x if hit else -(C[a + 14] / p - 1) * d * 100)
    g = np.array(g); print(f'  limit %{x}: dolma %{np.mean(g == x)*100:.0f} | işlem başı %{g.mean():+.3f} → {g.mean() * 5 * len(g):+.0f}$ toplam')
