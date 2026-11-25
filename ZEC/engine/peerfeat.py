# Akran özellikleri (1 s kapanmış bar → 1 dk'ya T−1ms ile): BTC getiri korelasyonu, ZEC ATR% / akran ATR% (BTC, XMR, BAT, PEPE), ATR korelasyonu
import numpy as np, pandas as pd, glob
P = '/home/claude/peer/'
def _one(f, tcol):
    d = pd.read_csv(f, usecols=[tcol, 'high', 'low', 'close']); t = d[tcol]
    d.index = pd.to_datetime(t, unit='ms') if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t); return d
def rd(fs, tcol, sym=None):
    L = [_one(f, tcol) for f in fs]
    if sym:
        for f in sorted(glob.glob(P + 'R_' + sym + '_1m_*')):
            hd = pd.read_csv(f, nrows=0).columns; L.append(_one(f, 'open_time' if 'open_time' in hd else 'time'))
    d = pd.concat(L); d = d[~d.index.duplicated(keep='last')].sort_index()
    return d[['high', 'low', 'close']].astype('float64')
def h1(d):
    H = d.resample('1h', label='left', closed='left').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs()))
    return pd.DataFrame({'atr': tr.ewm(alpha=1 / 14, adjust=False).mean() / H.close * 100, 'r': np.log(H.close).diff()})
def build(B):
    Z = h1(B[['high', 'low', 'close']])
    peers = {'BTC': rd([P + 'BTCUSDT_PERP_1m_2019_20261115.csv.gz'], 'time', 'BTCUSDT'), 'XMR': rd([P + 'XMRUSDT_1m_FULL_THROUGH_20261116.csv.gz'], 'time', 'XMRUSDT'),
             'BAT': rd(sorted(glob.glob(P + 'BATUSDT_1m_FULL*')), 'open_time_ms', 'BATUSDT'), 'PEPE': rd([P + '1000PEPEUSDT_1m_FULL_20230505_THROUGH_20261119.csv.gz'], 'open_time_ms', '1000PEPEUSDT')}
    Hs = {k: h1(v).reindex(Z.index) for k, v in peers.items()}
    pa = pd.concat([Hs[k].atr.rename(k) for k in Hs], axis=1); pm = pa.median(axis=1, skipna=True)
    rel = np.log(Z.atr / pm); relz = (rel - rel.rolling(720, min_periods=240).mean()) / rel.rolling(720, min_periods=240).std()
    atrC = Z.atr.rolling(168, min_periods=96).corr(pm)
    btcC = Z.r.rolling(72, min_periods=48).corr(Hs['BTC'].r)
    btcC_med = btcC.rolling(2160, min_periods=500).median(); atrC_med = atrC.rolling(2160, min_periods=500).median()
    F = pd.DataFrame({'relz': relz, 'atrC': atrC - atrC_med, 'btcC': btcC - btcC_med, 'btcCraw': btcC,
                      'btc24': Hs['BTC'].r.rolling(24).sum(), 'peerAtrZ': (np.log(pm) - np.log(pm).rolling(720, min_periods=240).mean()) / np.log(pm).rolling(720, min_periods=240).std()})
    print('akran son zamanlar:', {k: str(v.index[-1]) for k, v in peers.items()}, flush=True)
    return F
def to1m(F, B, lag=0):                                                        # bar kapandıktan sonra bilinir (+1 s), lag = ek saat gecikme
    G = F.copy(); G.index = G.index + pd.Timedelta(hours=1 + lag)
    return G.reindex(B.index + pd.Timedelta('1min') - pd.Timedelta('1ms'), method='ffill')
