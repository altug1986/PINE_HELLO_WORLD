import numpy as np, pandas as pd
D = '/home/claude/pepe/'
m = pd.concat([pd.read_csv(D + 'data/1000PEPEUSDT_1m_FULL_20230505_THROUGH_20261119.csv.gz', usecols=['open_time_ms', 'open', 'high', 'low', 'close']).rename(columns={'open_time_ms': 'time'}),
               pd.read_csv(D + 'data/1000PEPEUSDT_1m_PATCH_UTC_20261119_THROUGH_20261123.csv.gz', usecols=['time', 'open', 'high', 'low', 'close'])]).drop_duplicates('time', keep='last').sort_values('time')
m.index = pd.to_datetime(m.time, unit='ms'); O, H, L, C = (m[k].values for k in ('open', 'high', 'low', 'close')); T = m.index.values
def trades(df):
    df = df.rename(columns=lambda s: s.strip())
    g = []
    for no, x in df.groupby('İşlem numarası'):
        en = x[x.Tip.str.contains('giriş')].iloc[0]; ex = x[x.Tip.str.contains('çıkış')].iloc[0]
        if str(ex['Tarih ve saat']).startswith('Açık'): continue
        side = 1 if 'Uzun' in en.Tip else -1
        et = pd.Timestamp(en['Tarih ve saat']) - pd.Timedelta('3h') + pd.Timedelta('5min')
        xo = pd.Timestamp(ex['Tarih ve saat']) - pd.Timedelta('3h'); intra = str(ex.Sinyal).startswith('EXIT-ALL')
        g.append(dict(no=int(no), side=side, et=et, xo=xo, intra=intra, ep=float(en['Fiyat USDT']), xp=float(ex['Fiyat USDT']), pnl=float(en['Net K/Z USDT']),
                      tvMFE=float(en['Doğrusal hareket %']), tvMAE=float(en['Tersine hareket %'])))
    return pd.DataFrame(g)
t_full = trades(pd.read_excel(D + 'v117_full.xlsx', sheet_name='İşlemler'))
t_new = trades(pd.read_csv(D + 'tv_v117_20261125.csv'))
t = pd.concat([t_full, t_new[t_new.et > t_full.et.max()]], ignore_index=True).sort_values('et').reset_index(drop=True)
t = t[t.xo < m.index[-1]]
print('işlem', len(t), t.et.min(), '→', t.et.max(), '| TV net', round(t.pnl.sum(), 1))
def path(r):
    i0 = np.searchsorted(T, np.datetime64(r.et)); i1 = np.searchsorted(T, np.datetime64(r.xo + pd.Timedelta('5min')))
    return i0, max(i1, i0 + 1)
mfe = []; mae = []
for r in t.itertuples():
    i0, i1 = path(r)
    if r.side > 0: mfe.append((H[i0:i1].max() / r.ep - 1) * 100); mae.append((L[i0:i1].min() / r.ep - 1) * 100)
    else: mfe.append((1 - L[i0:i1].min() / r.ep) * 100); mae.append((1 - H[i0:i1].max() / r.ep) * 100)
t['mfe1m'] = mfe; t['mae1m'] = mae
t.to_pickle(D + 'trades.pkl')
print('\nTV ile 1 dk yol farkı (puan): MFE medyan', round((t.mfe1m - t.tvMFE).median(), 2), 'P90', round((t.mfe1m - t.tvMFE).quantile(.9), 2), '| MAE medyan', round((t.mae1m - t.tvMAE).median(), 2), 'P10', round((t.mae1m - t.tvMAE).quantile(.1), 2))
w = t[t.et >= '2026-11-21']
print(w[['no', 'side', 'et', 'xo', 'intra', 'ep', 'xp', 'pnl', 'tvMFE', 'mfe1m', 'tvMAE', 'mae1m']].round(4).to_string(index=False))
# 23 Kas 1 dk yolu
x = m.loc['2026-11-23 13:30':'2026-11-23 18:10'].copy(); x['TR'] = x.index + pd.Timedelta('3h')
x5 = x[['open', 'high', 'low', 'close']].resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}); x5['TR'] = x5.index + pd.Timedelta('3h')
print('\n23 Kas 5 dk barlar (TR açılış):'); print(x5.loc['2026-11-23 13:40':'2026-11-23 14:40'].round(7).to_string())
print('\n1 dk 13:50–14:10 UTC:'); print(x.loc['2026-11-23 13:50':'2026-11-23 14:10', ['TR', 'open', 'high', 'low', 'close']].round(7).to_string())
