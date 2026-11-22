import numpy as np, pandas as pd
from edge2 import D, c, t, run, report, trades, ent, sig
from p1m import SPL
rows = [report(D, run(D), 'V6.6')]
cap = lambda a=(5, 4), H=144, sl=5.0, lag=0: ent(sig('Kapitülasyon', a, lag), c, H, sl)
btc = lambda a=10, H=288, sl=10.0: ent(sig('BTC çöküşü', a), c, H, sl)
V = [('KAPİTÜLASYON 5/4 · 12 s · stop %5', cap()), ('  OI 1 mum gecikmeli', cap(lag=1)), ('  OI 2 mum gecikmeli', cap(lag=2)), ('  eşik 4/3', cap((4, 3))), ('  eşik 6/5', cap((6, 5))),
     ('  eşik 5/3', cap((5, 3))), ('  eşik 4/4', cap((4, 4))), ('  12 s → 8 s', cap(H=96)), ('  12 s → 18 s', cap(H=216)), ('  stop %5 → %3', cap(sl=3.0)), ('  stop %5 → %7', cap(sl=7.0)),
     ('BTC ÇÖKÜŞÜ ≤ −%10 · 24 s', btc())]
o = cap(); b = btc(); V.append(('KAPİTÜLASYON + BTC ÇÖKÜŞÜ', np.where(o != 0, o, b)))
for lab, ov in V: rows.append(report(D, run(D, ovr=ov), lab))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
# yıllara göre katkı ve olay sayısı
R0, _ = trades(D, run(D)); R1, _ = trades(D, run(D, ovr=cap()))
y = lambda R: pd.Series(R['pnl'], index=pd.to_datetime(t[R['xb']], unit='ms').year).groupby(level=0).sum()
ev = sig('Kapitülasyon', (5, 4)); st = np.flatnonzero((ev != 0) & (np.r_[0, ev[:-1]] == 0))
print('yıl Δ$:', (y(R1) - y(R0)).round(0).astype(int).to_dict(), '| olay başlangıcı/yıl:', pd.Series(pd.to_datetime(t[st], unit='ms').year).value_counts().sort_index().to_dict())
