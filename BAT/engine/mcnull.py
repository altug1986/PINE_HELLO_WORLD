import numpy as np, pandas as pd
from p1m import build, run, trades, TICK, SPL
from teco import mask_from
D = build(5); c = D['c']; t = D['t']; R, op = trades(D, run(D)); eb, xb, d, ep, q, pl = R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl']
# günlük MTM → her yıl en kârlı 3 ayrık 7 günlük pencere (büyük kâr haftaları), sabit maske
n = len(c); cum = np.zeros(n); np.add.at(cum, xb, pl); cum = np.cumsum(cum); opn = np.zeros(n)
for a, b, dd, e, qq in zip(eb, xb, d, ep, q): opn[a + 1:b] = (c[a + 1:b] - e) * qq * dd
s = pd.Series(cum + opn, index=pd.to_datetime(t, unit='ms')).resample('D').last().ffill(); dp = s.diff().fillna(s.iloc[0]); M = mask_from(dp, 7, 3)
day = pd.to_datetime(t[xb], unit='ms').normalize(); big = M.reindex(day).fillna(False).to_numpy()
gross = d * (c[xb] - c[eb]) * q; cost = gross - pl                                       # gerçek maliyet (komisyon + kayma)
rng = np.random.default_rng(7); NS = 10000
S = rng.choice([-1.0, 1.0], size=(NS, len(pl)))                                          # MAYMUN: aynı giriş/çıkış zamanı, rastgele yön, aynı maliyet
sim = S * (np.abs(gross) * np.sign(gross) * d)[None, :] * d[None, :]
sim = S * (c[xb] - c[eb])[None, :] * q[None, :] - cost[None, :]
print(f'V6.6 · {len(pl)} işlem · büyük kâr haftası {int(M.sum())} gün (%{M.mean()*100:.0f}) · rastgele yönlü {NS} maymun, aynı zamanlama ve maliyet')
for lab, m in [('TÜM DÖNEM', np.ones(len(pl), bool)), ('BÜYÜK KÂR HAFTALARI', big), ('DİĞER DÖNEMLER', ~big), ('DİĞER · 2024-11 öncesi', ~big & (t[eb] < SPL)), ('DİĞER · 2024-11 sonrası', ~big & (t[eb] >= SPL))]:
    real = pl[m].sum(); nul = sim[:, m].sum(1); p = (np.sum(nul >= real) + 1) / (NS + 1)
    w = pl[m]; pf = w[w > 0].sum() / -w[w < 0].sum()
    print(f'{lab:26s} işlem {m.sum():5d} | gerçek {real:7.0f}$ | PF {pf:.2f} | maymun ort {nul.mean():7.0f}$ (%95 üst {np.quantile(nul, .95):6.0f}$) | p = {p:.4f}')
