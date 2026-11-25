import json, base64, gzip, numpy as np, pandas as pd
d = json.load(open('/home/claude/general-quant-algo-code-center/research/CYBORG/2026-11-24_CANLI_4600_LIMIT/regulator_data_20261124.json'))
X = np.frombuffer(gzip.decompress(base64.b64decode(d['b64'])), '<i4').reshape(d['rows'], d['L']).astype(np.float64).cumsum(1) * d['scale']
nom = [300, 95, 250, 180, 226, 226, 200, 200, 220, 200, 500, 370, 210, 250, 200, 320, 250, 150, 250]
rows = [d['lim41']['0'], d['lim41']['1']] + [d['lim'][str(i)] for i in range(2, 19)]
P = pd.Series(sum(n_ * X[r] for n_, r in zip(nom, rows)), index=pd.to_datetime(d['t0'], unit='ms') + pd.to_timedelta(np.arange(d['L']) * 10, 'min'))
P.to_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl')
dd = P.cummax() - P; print('Canlı portföy (limit):', P.index[0], '→', P.index[-1], 'net', round(P.iloc[-1]), 'MaxDD', round(dd.max(), 1), dd.idxmax())
M = P.resample('MS').last().diff(); print('Son 6 ay:', M.tail(6).round(1).to_dict())
Dd = P.resample('1D').last().diff(); print('Kasım günlük:', Dd['2026-11-01':].round(1).to_dict())
print('Ekim 1–24:', round(P['2026-10-24 23:50':].iloc[0] - P[:'2026-10-01'].iloc[-1], 1), '· Kasım 1–24:', round(P.iloc[-1] - P[:'2026-11-01'].iloc[-1], 1), '· 3–11 Kas:', round(P[:'2026-11-11 23:50'].iloc[-1] - P[:'2026-11-03'].iloc[-1], 1))
