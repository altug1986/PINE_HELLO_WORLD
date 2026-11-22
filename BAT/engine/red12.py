import numpy as np, pandas as pd, itertools
import v55
from redrsi import go, R15, z, c, t, n, D, hr
k12 = t // 43200000; u, st = np.unique(k12, return_index=True); en = np.r_[st[1:], n]; red12 = c[en - 1] < D['o'][st]
ii = np.searchsorted(u, k12); prev12 = np.where(ii > 0, red12[np.maximum(ii - 1, 0)], False); h12 = hr % 12
rows = []
for W, th in itertools.product([12, 6], [60, 55, 65, 50]):
    rows.append(go(prev12 & (h12 < W) & (R15 <= th), z, f'long: kırmızı 12s mum sonrası {W}s RSI15>{th}'))
d = pd.DataFrame(rows); pd.set_option('display.width', 300); print(d.to_string(index=False)); d.to_pickle('red12.pkl')
