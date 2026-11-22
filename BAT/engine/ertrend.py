import numpy as np, pandas as pd, itertools, v55
from donhold import bt, D, c, n
from short_top_lib import to5, zh, DIVh
from met import SPLIT
from v55 import t
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24))); WL = (a >= 1) | (b >= 1.5); WS = (a <= -1) | (b <= -1.5)
v55.GL = WL.astype(np.int64); v55.GS = WS.astype(np.int64)
rows = [bt({}, 'V6.1 (taban)')]
ad = np.abs(np.diff(c, prepend=c[0])); cs = np.cumsum(ad)
for N in (288, 576, 864):
    ret = np.r_[np.full(N, 0.0), (c[N:] / c[:-N] - 1) * 100]; path = np.r_[np.full(N, np.nan), cs[N:] - cs[:-N]]
    er = np.nan_to_num(np.abs(np.r_[np.full(N, 0.0), c[N:] - c[:-N]]) / path)
    isq = er[(t < SPLIT) & (er > 0)]
    for qq, r in itertools.product([0.7, 0.8, 0.9], [3.0, 6.0]):
        e = np.quantile(isq, qq)                                            # eşik yalnız ilk dönemden
        dn = (er >= e) & (ret <= -r); up = (er >= e) & (ret >= r)
        v55.GL = (WL | dn).astype(np.int64); v55.GS = (WS | up).astype(np.int64)
        rows.append(bt({}, f'{N//288} gün, düzgünlük ≥ %{int(qq*100)}lik ({e:.2f}), hareket ≥ %{r:.0f}: trende ters giriş yok'))
r_ = pd.DataFrame(rows); pd.set_option('display.width', 240); print(r_.to_string(index=False))
b0 = r_.iloc[0]; print('V6.1\'den iyi (net + MaxDD30 + Calmar30):', ((r_.net > b0.net) & (r_.MaxDD30 <= b0.MaxDD30) & (r_.Calmar30 > b0.Calmar30)).sum(), '/', len(r_) - 1)
