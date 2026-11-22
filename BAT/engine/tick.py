import numpy as np, numba as nb, pandas as pd, itertools
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
@nb.njit(cache=True)
def g_tk(c, dP, aL, aS, hold, sl, exL, exS, mode):
    # 24s ekran dönüşü; çıkış: 24s değişim normale dönünce (long: dP ≥ exL, short: dP ≤ exS) ya da süre/stop
    # mode 0: sinyal sürdükçe süre yenilenir (V1.0) | mode 1: giriş = ekran dönüşü (dP eşik altından geri çıkınca)
    n = len(c); tg = np.zeros(n, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(1, n):
        x = dP[i]; x1 = dP[i - 1]
        if np.isnan(x) or np.isnan(x1): tg[i] = v; continue
        if v != 0:
            r = (c[i] / e - 1) * v
            if r <= -sl / 100 or i - vb >= hold or (v > 0 and x >= exL) or (v < 0 and x <= exS): v = 0
        if mode == 0: s = 1 if x <= -aL else (-1 if x >= aS else 0)
        else: s = 1 if (x1 <= -aL and x > -aL) else (-1 if (x1 >= aS and x < aS) else 0)
        if s != 0:
            if v != s: v = s; e = c[i]; vb = i
            elif mode == 0: vb = i
        tg[i] = v
    return tg
D0 = load(5); c0 = D0['c']; dP0 = (c0 / np.r_[np.full(288, np.nan), c0[:-288]] - 1) * 100; DI = cut(D0, SPLIT); k = len(DI['c']); dPI = dP0[:k]
rows = []
for mode, ex, hold, sl in itertools.product([0, 1], [-99, -2, -1, 0, 1, 2, 3], [648, 1296], [3, 8]):
    exL, exS = (99, -99) if ex == -99 else (ex, -ex)
    m = metrics(DI, simR(DI, g_tk(DI['c'], dPI, 3.5, 5.5, hold, sl, exL, exS, mode)))
    rows.append(dict(mode=['V1.0 giriş', 'ekran dönüşü girişi'][mode], çıkış='yok (süre)' if ex == -99 else f'24s ≥ {ex}', hold=hold, sl=sl, IS=m['net'], n=m['n'], pf=m['pf'], sharpe=m['sharpe'], p98=m['p98'], sc=m['net'] - 1.5 * m['p98']))
df = pd.DataFrame(rows); pd.set_option('display.width', 250); print(df.sort_values('sc', ascending=False).head(10).to_string(index=False))
print(df[(df.mode == 'V1.0 giriş') & (df.hold == 648) & (df.sl == 3)].to_string(index=False))
