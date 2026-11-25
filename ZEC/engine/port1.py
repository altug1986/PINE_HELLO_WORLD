import json, base64, gzip, numpy as np, pandas as pd
d = json.load(open('/home/claude/port/data.json'))
X = np.frombuffer(gzip.decompress(base64.b64decode(d['b64'])), '<i4').reshape(d['rows'], d['L']).astype(np.float64).cumsum(1) * d['scale']
nom = [300, 95, 250, 180, 226, 226, 200, 200, 220, 200, 500, 370, 210, 250, 200, 320, 250, 150, 250]   # canlı 24 Kas
rows = [19, 20] + list(range(2, 19))                                                                     # Path-Lock = V4.1 LIQ
P = sum(n_ * X[r] for n_, r in zip(nom, rows))
t = pd.to_datetime(d['t0'], unit='ms') + pd.to_timedelta(np.arange(d['L']) * 10, 'min')     # TR saati (t0 = 2023-05-08 00:00 TR)
# t0 ms zaten UTC
S = pd.Series(P, index=t); S.to_pickle('/home/claude/port/portfoy_10dk_utc.pkl')
dd = S.cummax() - S; print('Portföy:', t[0], '→', t[-1], 'net', round(S.iloc[-1]), 'MaxDD', round(dd.max()), 'tarih', dd.idxmax())
D = S.resample('1D').last().diff().dropna(); print('son 30 gün portföy', round(S.iloc[-1] - S[S.index >= S.index[-1] - pd.Timedelta(days=30)].iloc[0], 1))
