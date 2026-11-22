import numpy as np, pandas as pd, v55
from donhold import D, c, n
from short_top_lib import to5, zh, DIVh, q0, simR, run
from met import SPLIT
from v55 import t
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24))); v55.GL = ((a >= 1) | (b >= 1.5)).astype(np.int64); v55.GS = ((a <= -1) | (b <= -1.5)).astype(np.int64)
R = simR(D, run(q0)[0]); N = 576; ad = np.abs(np.diff(c, prepend=c[0])); cs = np.cumsum(ad)
ret = np.r_[np.full(N, 0.0), (c[N:] / c[:-N] - 1) * 100]; er = np.nan_to_num(np.abs(np.r_[np.full(N, 0.0), c[N:] - c[:-N]]) / np.r_[np.full(N, np.nan), cs[N:] - cs[:-N]])
e = np.quantile(er[(t < SPLIT) & (er > 0)], 0.9); i = R['eb']; s = R['dir']; p = R['pnl']
cnt = ((er[i] >= e) & (ret[i] * s <= -3))                       # düzgün trende TERS açılan işlemler
for lab, m in [('düzgün trende ters (2 gün, %3+)', cnt), ('diğer tüm işlemler', ~cnt)]:
    print(f'{lab}: {m.sum()} işlem | net {p[m].sum():.0f}$ | işlem başı {p[m].mean():.2f}$ | kazanma %{(p[m] > 0).mean()*100:.0f} | ilk dönem {p[m & (t[R["xb"]] < SPLIT)].sum():.0f}$ / ikinci dönem {p[m & (t[R["xb"]] >= SPLIT)].sum():.0f}$')
from eng import COMM
m = cnt; ep, xp, q, dr = R['ep'][m], R['xp'][m], R['q'][m], s[m]; pp = p[m]; gross = (xp - ep) * q; fee = COMM * q * (ep + xp)
srt = np.sort(pp)[::-1]; print('kâr yoğunluğu: toplam', round(pp.sum()), '| en iyi 5 hariç', round(pp.sum() - srt[:5].sum()), '| en iyi 10 hariç', round(pp.sum() - srt[:10].sum()), '| en iyi 20 hariç', round(pp.sum() - srt[:20].sum()))
yr = pd.Series(pp, index=pd.to_datetime(t[R['xb']][m], unit='ms').year).groupby(level=0).agg(['count', 'sum']).round(0); print('yıllara göre:', {k: (int(v['count']), int(v['sum'])) for k, v in yr.iterrows()})
rng = np.random.default_rng(7); B = 20000
bs = rng.choice(pp, (B, len(pp))).sum(1); print(f'bootstrap: toplam P5 {np.percentile(bs, 5):.0f}$ | zarar olasılığı %{(bs < 0).mean()*100:.1f}')
sg = rng.choice([-1, 1], (B, len(pp))); mk = (gross[None, :] * sg - fee[None, :]).sum(1); print(f'maymun (aynı zaman, rastgele yön): ortalama {mk.mean():.0f}$ | gerçek {pp.sum():.0f}$ | p = {(mk >= pp.sum()).mean():.4f}')
pool = np.nonzero((er >= e) & (np.abs(ret) >= 3))[0]; pool = pool[pool < n - 1500]; hold = (R['xb'] - R['eb'])[m]
tot = np.zeros(B)
for k in range(B):
    ei = rng.choice(pool, len(pp)); hx = np.minimum(ei + rng.choice(hold, len(pp)), n - 1); d_ = -np.sign(ret[ei]); e_ = c[ei] + 1e-5 * d_; x_ = c[hx] - 1e-5 * d_; qq = 500 / c[ei]
    tot[k] = ((x_ - e_) * qq * d_ - COMM * qq * (e_ + x_)).sum()
print(f'maymun (aynı rejimde rastgele zaman, trende ters, aynı süreler): ortalama {tot.mean():.0f}$ | P95 {np.percentile(tot, 95):.0f}$ | gerçek {pp.sum():.0f}$ | p = {(tot >= pp.sum()).mean():.4f}')
