"""ZECUSDT.P veri yükleyici — MASTER 1m (tam senkron) → istenen zaman dilimi.

    from zec_data import load
    X = load(tf_min=2)            # DataFrame, index = bar AÇILIŞI (UTC), her değer bar KAPANIŞINDA bilinir
    X = load(tf_min=2, lag=1)     # metrics/OI serileri +1 periyot (5 dk) ek gecikme (gecikme testi)

Kaynak: GitHub data/ZECUSDT_MASTER_1m_SYNC_{yıl}.parquet (üretici: build_master.py, denetim: MASTER_SENKRON_DENETIM.json).
Kural sözlüğü: ZEC/README.md §2.1. Önbellek: ~/.cache/zec_master/.
"""
import os, urllib.request, pandas as pd

U = 'https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/data/'
YEARS = range(2021, 2027)
CACHE = os.path.expanduser('~/.cache/zec_master/')
SUM = ['volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'taker_delta_base']
MAX = ['wh_buy_max', 'wh_sell_max']
METRICS = ['oi_zec', 'oi_usdt', 'tt_acc_ls', 'tt_pos_ls', 'glb_ls', 'taker_ls_5m', 'btc_oi', 'xmr_oi']


def master(src=U, years=YEARS):
    os.makedirs(CACHE, exist_ok=True); parts = []
    for y in years:
        fn = f'ZECUSDT_MASTER_1m_SYNC_{y}.parquet'; p = CACHE + fn
        if not os.path.exists(p):
            if src.startswith('http'):
                urllib.request.urlretrieve(src + fn, p)
            else:
                p = src + fn
        parts.append(pd.read_parquet(p))
    X = pd.concat(parts, ignore_index=True)
    X.index = pd.to_datetime(X.pop('time'), unit='ms')
    X[['open', 'high', 'low', 'close']] = X[['open', 'high', 'low', 'close']].astype('float64').round(2)  # tick 0,01 (float32 kalıntısı yok)
    return X


def load(tf_min=2, lag=0, src=U, years=YEARS):
    X = master(src, years)
    if lag:
        X[METRICS] = X[METRICS].shift(5 * lag)
    if tf_min == 1:
        return X
    agg = {c: 'last' for c in X.columns}
    agg.update({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    agg.update({c: 'sum' for c in SUM + [c for c in X.columns if c.startswith('wh_') and c not in MAX]})
    agg.update({c: 'max' for c in MAX})
    B = X.resample(f'{tf_min}min', label='left', closed='left').agg(agg)
    return B[B.close.notna()]


if __name__ == '__main__':
    import sys
    B = load(int(sys.argv[1]) if len(sys.argv) > 1 else 2, src=sys.argv[2] if len(sys.argv) > 2 else U)
    print(len(B), B.index[0], B.index[-1], B.shape[1], 'kolon')
    print(B.notna().mean().round(3).to_string())
