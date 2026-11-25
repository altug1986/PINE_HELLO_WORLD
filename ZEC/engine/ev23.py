import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf
D1 = ztf.build(1); e1, t1 = ztf.run(D1); D2 = ztf.build(2); e2, t2 = ztf.run(D2)
w0, w1 = pd.Timestamp('2026-11-23 10:00'), pd.Timestamp('2026-11-24 00:00')
X = D1['B'].loc[w0:w1]
b = X[['open', 'high', 'low', 'close', 'volume', 'oi_zec', 'basis_bps', 'tt_pos_ls', 'wh_buy_q50k', 'wh_sell_q50k']].resample('15min').agg(
    {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'oi_zec': 'last', 'basis_bps': 'mean', 'tt_pos_ls': 'last', 'wh_buy_q50k': 'sum', 'wh_sell_q50k': 'sum'})
b['oi%'] = (b.oi_zec / b.oi_zec.iloc[0] - 1) * 100; b['TR'] = b.index + pd.Timedelta('3h')
print(b[['TR', 'open', 'high', 'low', 'close', 'volume', 'oi%', 'basis_bps', 'tt_pos_ls', 'wh_buy_q50k', 'wh_sell_q50k']].round(2).to_string())
for lab, t in (('1 dk R2', t1), ('2 dk R2', t2)):
    tt = t[(t.et >= w0 - pd.Timedelta('6h')) & (t.et <= w1)]
    print(f'\n{lab} işlemleri (UTC):'); print(tt.assign(xt=D1['B'].index[0] if False else None)[['et', 'side', 'ep', 'xp', 'pnl', 'mae']].round(3).to_string(index=False))
