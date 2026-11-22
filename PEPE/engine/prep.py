import pandas as pd, numpy as np, sys
src = sys.argv[1]
d = pd.read_csv(src, usecols=['open_time_ms', 'open', 'high', 'low', 'close', 'volume'])
for c in ['open', 'high', 'low', 'close']:
    d[c] = np.round(d[c].values / 1e-7) * 1e-7          # TV price grid (1000PEPE tick 1e-7)
d['t'] = pd.to_datetime(d.open_time_ms, unit='ms', utc=True)
d = d.set_index('t').drop(columns='open_time_ms')
agg = dict(open='first', high='max', low='min', close='last', volume='sum')
for tf in ['5min', '30min', '60min', '240min']:
    b = d.resample(tf, label='left', closed='left').agg(agg).dropna()
    b.to_parquet(f'/home/claude/data/pepe_{tf}.parquet')
    print(tf, len(b), b.index[0], b.index[-1])
