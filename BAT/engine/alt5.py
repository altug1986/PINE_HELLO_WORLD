import numpy as np, pandas as pd
from eng import load
D = load(5); t = D['t']; out = {}
for s in ['ADAUSDT', 'DOGEUSDT', 'LTCUSDT', 'ETCUSDT', 'XMRUSDT']:
    a = pd.read_csv(f'alt/{s}.csv.gz', usecols=['time', 'close']); k = (a.time.to_numpy() // 300000).astype(np.int64)
    st = np.r_[0, np.flatnonzero(np.diff(k)) + 1]; en = np.r_[st[1:], len(a)]; kk = k[st]; cc = a.close.to_numpy()[en - 1]
    j = np.searchsorted(kk, t // 300000, side='right') - 1; out[s] = np.where(j >= 0, cc[np.clip(j, 0, None)], np.nan); print(s, np.isnan(out[s]).mean().round(3))
np.savez('alt5.npz', **out)
