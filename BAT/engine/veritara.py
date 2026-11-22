import numpy as np, pandas as pd
from p1m import build, run, trades
from teco import mask_from
D = build(5); c = D['c']; t = D['t']; R, _ = trades(D, run(D)); eb, xb, d, pl = R['eb'], R['xb'], R['dir'], R['pnl']; n = len(c)
def at5(ms, v):                                          # önceki kapanmış 5 dk değeri (bar açılışından önceki kayıt)
    j = np.searchsorted(ms, t, side='left') - 1; return np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
z = lambda x, N: ((pd.Series(x) - pd.Series(x).rolling(N, min_periods=N // 2).mean()) / pd.Series(x).rolling(N, min_periods=N // 2).std()).to_numpy()
m = pd.read_pickle('top5.pkl'); F = {}
F['Taker al/sat hacim oranı (z 24s)'] = z(np.log(at5(m.ms.to_numpy() + 300000, m.sum_taker_long_short_vol_ratio.where(m.sum_taker_long_short_vol_ratio > 0).to_numpy())), 288)
F['Genel L/S hesap oranı (z 30g)'] = z(np.log(at5(m.ms.to_numpy() + 300000, m.count_long_short_ratio.where(m.count_long_short_ratio > 0).to_numpy())), 8640)
k = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms', 'volume', 'number_of_trades', 'taker_buy_base_asset_volume']); g = k.open_time_ms // 300000
a = k.groupby(g)[['volume', 'number_of_trades', 'taker_buy_base_asset_volume']].sum().reindex(t // 300000).to_numpy()
tb = pd.Series(a[:, 2]).rolling(6).sum() / pd.Series(a[:, 0]).rolling(6).sum(); F['Kline taker alış payı 30 dk (z 24s)'] = z(tb.to_numpy(), 288)
nt = pd.Series(a[:, 1]).rolling(6).sum(); F['İşlem adedi 30 dk / 24 s ort.'] = (nt / (pd.Series(a[:, 1]).rolling(288).sum() / 48)).to_numpy()
bd = pd.read_csv('btcdom1m.csv.gz', usecols=['time', 'close']); bdc = bd.groupby(bd.time // 300000).close.last().reindex(t // 300000).ffill().to_numpy()
F['BTC dominans 24 s değişim %'] = (bdc / np.r_[np.full(288, np.nan), bdc[:-288]] - 1) * 100
F['BTC dominans 7 g değişim %'] = (bdc / np.r_[np.full(2016, np.nan), bdc[:-2016]] - 1) * 100
# büyük kâr haftaları (sabit maske)
cum = np.zeros(n); np.add.at(cum, xb, pl); s = pd.Series(np.cumsum(cum), index=pd.to_datetime(t, unit='ms')).resample('D').last().ffill(); M = mask_from(s.diff().fillna(0), 7, 3)
big = M.reindex(pd.to_datetime(t[xb], unit='ms').normalize()).fillna(False).to_numpy()
print(f'V6.6 {len(pl)} işlem | özellik değeri işlem yönüyle çarpıldı (+ = pozisyon yönünde, − = ters)')
rows = []
for lab, x in F.items():
    v = x[eb] * d; ok = ~np.isnan(v); q = pd.qcut(v[ok], 5, labels=False, duplicates='drop'); p = pl[ok]; nb = ~big[ok]
    qq = [p[q == i].mean() for i in range(5)]; qn = [p[nb & (q == i)].mean() for i in range(5)]
    rows.append({'veri': lab, 'kapsanan işlem': int(ok.sum()), 'Q1 ort $': round(qq[0], 2), 'Q2': round(qq[1], 2), 'Q3': round(qq[2], 2), 'Q4': round(qq[3], 2), 'Q5 ort $': round(qq[4], 2),
                 'Q1−Q5 fark $': round(qq[0] - qq[4], 2), 'normal dönem Q1 / Q5 $': f'{qn[0]:.2f} / {qn[4]:.2f}', 'sıra korel.': round(pd.Series(v[ok]).corr(pd.Series(p), method='spearman'), 3)})
pd.set_option('display.width', 260); print(pd.DataFrame(rows).to_string(index=False)); print('işlem başı ortalama', round(pl.mean(), 2), '$')
