import numpy as np, pandas as pd
from p1m import build, run, report, trades
from eng import rma, trr
D = build(5); c = D['c']; h = D['h']; l = D['l']
atr = rma(trr(h, l, c, True), 14) / c * 100; m30d = pd.Series(atr).rolling(8640, min_periods=2000).mean().to_numpy(); ratio = atr / m30d
out = run(D); R, _ = trades(D, out); eb = R['eb']; d = R['dir']; q = R['q']
print(f"baz: {len(eb)} işlem, net {R['pnl'].sum():.0f}$")
for thr in (1.5, 2.0):
    hi = np.nan_to_num(ratio) >= thr; m = hi[eb]
    print(f'ATR ≥ 30 günlük ortalamanın {thr}x: zamanın %{hi.mean()*100:.1f}\'i | bu anlarda açılan {m.sum()} işlem, toplam {R["pnl"][m].sum():.0f}$')
    for k in (1, 2):
        g = d * (c[eb] - c[np.maximum(eb - k, 0)]) * q; print(f'   KAHİN üst sınır: bu işlemlere {k} mum önce girilseydi {g[m].sum():+.0f}$ (işlem başı {g[m].mean():+.2f}$) | tüm işlemlerde {g.sum():+.0f}$')
rows = [report(D, out, 'V6.6 baz')]
for thr in (1.5, 2.0):
    hi = np.nan_to_num(ratio) >= thr
    for f in (0.5, 0.75): rows.append(report(D, run(D, hi=hi, hfac=f), f'ATR ≥ {thr}x iken giriş eşikleri ×{f}'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
