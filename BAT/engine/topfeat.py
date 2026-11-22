import numpy as np, pandas as pd
from v55 import t, n
d = pd.read_pickle('top5.pkl')
def align(col, lag=1):
    v = d[col].to_numpy(float); v = np.where(v > 0, v, np.nan); ms = d.ms.to_numpy()
    j = np.searchsorted(ms, t, side='right') - 1; x = np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
    x = pd.Series(x).ffill(limit=6).to_numpy()            # en çok 30 dk boşluk doldur
    return np.r_[np.full(lag, np.nan), x[:-lag]] if lag else x
def z(x, N): s = pd.Series(x); return ((s - s.rolling(N, min_periods=N // 2).mean()) / s.rolling(N, min_periods=N // 2).std()).to_numpy()
def feats(lag=1):
    TA, TP, GA = np.log(align('count_toptrader_long_short_ratio', lag)), np.log(align('sum_toptrader_long_short_ratio', lag)), np.log(align('count_long_short_ratio', lag))
    F = {}
    for N in (288, 2016, 8640):
        F[f'zTP{N}'] = z(TP, N); F[f'zTA{N}'] = z(TA, N); F[f'zDIV{N}'] = z(TP - TA, N); F[f'zTG{N}'] = z(TA - GA, N)
    for k in (12, 48, 288): F[f'dTP{k}'] = (TP - np.r_[np.full(k, np.nan), TP[:-k]]) * 100
    F['TP'] = TP; F['TA'] = TA; return F
if __name__ == '__main__':
    F = feats(); ok = ~np.isnan(F['zTP2016']); print('özellik dolu bar oranı %', round(ok.mean() * 100, 1), '| ilk dolu', pd.to_datetime(t[np.argmax(ok)], unit='ms'))
