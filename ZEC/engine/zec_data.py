"""ZECUSDT.P veri yükleyici — GitHub data/ → 2 dk bar + güvenli hizalanmış OI / balina / taker / funding.

Kullanım:
    from zec_data import load
    D = load(tf_min=2)          # dict: bars (DataFrame), feat (DataFrame, bar index'li)

Zaman kuralı (denetim: ZEC/data_quality/KALITE_RAPORU.json):
  * 1m `time` = bar AÇILIŞI (UTC ms). TF bar = label left, closed left; dolum = bar kapanışı.
  * metrics / OI damgası = SON etiketli (anlık görüntü t anında). Bar kapanışı T'de yalnız
    damga <= T - 5 dk değerleri kullanılır (1 periyot güvenlik payı). `lag` ile ek gecikme testi (0/1/2).
  * taker 5m `open_time_ms` = pencere açılışı; değer pencere kapanışında (`available_from_utc`) bilinir.
  * funding: olay zamanında bilinir; olaydan sonra ffill.
  * Boşluk doldurulmaz (2022-02-26→03-01, 2022-04-01→04-03 Binance'te de veri yok). Sıfır OI → NaN.
"""
import pandas as pd, numpy as np

U = 'https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/data/'
YEARS = ['20210101_THROUGH_20211231', '20220101_THROUGH_20221231', '20230101_THROUGH_20231231',
         '20240101_THROUGH_20241231', '20250101_THROUGH_20251231', '20260101_THROUGH_20261123']
F_METRICS = 'ZECUSDT_METRICS_5m_UTC_20211201_THROUGH_20261123.csv.gz'
F_TAKER = 'ZECUSDT_PERP_TAKER_BUY_SELL_5m_UTC_20210101_THROUGH_20261123.csv.gz'
F_FUND = 'ZECUSDT_FUNDING_EVENTS_20210101_20261123.csv'
TICK = 0.01
FIVE = pd.Timedelta('5min')


def _asof(series, when, lag_steps=0, tol='20min'):
    """series: damga index'li; when: bilgi kesim zamanları. lag_steps: ek 5 dk gecikme."""
    s = series.sort_index()
    if lag_steps:
        s = s.shift(lag_steps)
    return s.reindex(when, method='ffill', tolerance=pd.Timedelta(tol)).values


def load(tf_min=2, src=U, lag=0):
    m = pd.concat([pd.read_csv(src + f'ZECUSDT_1m_UTC_{y}.csv.gz') for y in YEARS], ignore_index=True)
    m.index = pd.to_datetime(m.time, unit='ms')
    b = m.resample(f'{tf_min}min', label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    b[['open', 'high', 'low', 'close']] = (b[['open', 'high', 'low', 'close']] / TICK).round() * TICK
    tclose = b.index + pd.Timedelta(minutes=tf_min)
    cut = tclose - FIVE                                  # metrics/OI bilgi kesimi
    x = pd.read_csv(src + F_METRICS)
    x.index = pd.to_datetime(x.timestamp_ms, unit='ms')
    x.loc[x.open_interest_zec <= 0, ['open_interest_zec', 'open_interest_usdt']] = np.nan
    feat = pd.DataFrame(index=b.index)
    for c in ['open_interest_zec', 'open_interest_usdt', 'top_trader_accounts_ls',
              'top_trader_positions_ls', 'global_accounts_ls', 'taker_long_short_vol_ratio']:
        feat[c] = _asof(x[c], cut, lag)
    tk = pd.read_csv(src + F_TAKER, usecols=['open_time_ms', 'taker_buy_usdt', 'taker_sell_usdt', 'total_usdt'])
    tk.index = pd.to_datetime(tk.open_time_ms, unit='ms') + FIVE    # bilindiği an = pencere kapanışı
    for c in ['taker_buy_usdt', 'taker_sell_usdt', 'total_usdt']:
        feat[c] = _asof(tk[c], tclose, lag)
    fu = pd.read_csv(src + F_FUND)
    fu.index = pd.to_datetime(fu.funding_time_ms, unit='ms')
    feat['funding_pct'] = fu.funding_rate_percent.sort_index().reindex(tclose, method='ffill', tolerance=pd.Timedelta('9h')).values  # 2026-11 boş (kaynak 10-31'de bitiyor)
    return {'bars': b, 'feat': feat, 'tclose': tclose}


if __name__ == '__main__':
    import sys
    D = load(int(sys.argv[1]) if len(sys.argv) > 1 else 2, src=sys.argv[2] if len(sys.argv) > 2 else U)
    b, f = D['bars'], D['feat']
    print(len(b), b.index[0], b.index[-1])
    print(f.notna().mean().round(3).to_string())
