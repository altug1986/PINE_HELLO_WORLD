import numpy as np, pandas as pd, itertools, v55
from donhold import bt, D, c, n
rows = [bt({}, 'V6.1 (taban)')]
for N, m in itertools.product([100, 200, 300], [2.0, 4.0, 6.0]):
    mom = np.r_[np.full(N, 0.0), (c[N:] / c[:-N] - 1) * 100]
    ll = pd.Series(D['l']).rolling(N).min().shift(1).fillna(0).to_numpy(); hh = pd.Series(D['h']).rolling(N).max().shift(1).fillna(1e18).to_numpy()
    v55.DLL = np.where(mom >= m, ll, 1e18); v55.DHH = np.where(mom <= -m, hh, -1e18)      # kararlı trend: son N barda ≥ m% bizim yönde
    rows.append(bt(dict(dOn=1, dMin=0.0), f'{N} bar: trend ≥%{m:.0f} ve dip/tepe kırılmadıkça tut'))
r = pd.DataFrame(rows); pd.set_option('display.width', 230); print(r.to_string(index=False))
