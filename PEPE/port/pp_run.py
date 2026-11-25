import sys; sys.path.insert(0, '/home/claude/zecwork'); import numpy as np, pandas as pd, glob, pepeport as pp
def ld1m(files, tcol=None):
    out = []
    for f in files:
        d = pd.read_csv(f); tc = 'open_time' if 'open_time' in d else ('open_time_ms' if 'open_time_ms' in d else 'time')
        d.index = pd.to_datetime(d[tc].astype('int64'), unit='ms'); out.append(d[['open', 'high', 'low', 'close']])
    d = pd.concat(out); d = d[~d.index.duplicated(keep='last')].sort_index(); d.columns = list('ohlc'); return d
def ldoi(f):
    d = pd.read_csv(f, usecols=['timestamp_utc', 'open_interest_qty']); return pd.Series(d.open_interest_qty.values, pd.to_datetime(d.timestamp_utc).dt.tz_localize(None))
btc = ld1m(sorted(glob.glob('/home/claude/peer/BTCUSDT_PERP_1m_*.csv.gz')) + sorted(glob.glob('/home/claude/peer/R_BTCUSDT_*')))
boi = ldoi('/home/claude/peer/BTCUSDT_OI_5m.csv.gz'); boi = boi[~boi.index.duplicated()]
def metrics(nm, d5, eq, T, mid_t=None):
    t = d5.index; x30 = eq[::6]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    mid = len(eq) // 2 if mid_t is None else np.searchsorted(t.values, np.datetime64(mid_t)); yr = pd.Series(eq, t).resample('YE').last().diff().fillna(pd.Series(eq, t).resample('YE').last())
    r0, r1 = np.searchsorted(t.values, np.datetime64('2025-09-01')), np.searchsorted(t.values, np.datetime64('2026-01-01'))
    d_ = np.diff(x30[::48]); y = eq[np.searchsorted(t.values, np.datetime64('2023-01-01')):][::6]; y = y - y[0]
    return dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, win=(T[:, 5] > 0).mean() * 100, MaxDD=dd.max(), DDp95=np.percentile(dd, 95), DDp99=np.percentile(dd, 99),
                Sharpe=d_.mean() / d_.std() * np.sqrt(365), ilkYarı=eq[mid], ikinciYarı=eq[-1] - eq[mid], ralliHariç=eq[-1] - (eq[min(r1, len(eq) - 1)] - eq[min(r0, len(eq) - 1)]),
                R2_2023p=np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2 if len(y) > 10 else np.nan, maeYok=0, yıllar=' '.join(f'{v:.0f}' for v in yr.values))
