import numpy as np, v55
import stochwh as S
s30 = S.SER['HES'].resample('30min', label='left').last().dropna(); k, _ = S.stochrsi(s30)
for lag in (0, 6, 12):
    K = np.nan_to_num(S.to5m(k, 1800000), nan=50.0)
    if lag: K = np.r_[np.full(lag, 50.0), K[:-lag]]
    r = S.bt(S.WL | (K >= 90), S.WS | (K <= 10), f'V6.1 + StochRSI30 HES 90/10, {lag * 5} dk geç'); print(r)
