import pandas as pd, numpy as np, glob, shutil
P = '/home/claude/peer/'
def rd(f):
    d = pd.read_csv(f); d = d.rename(columns={'open_time': 'open_time_ms', 'time': 'open_time_ms'}); return d[['open_time_ms', 'open', 'high', 'low', 'close', 'volume']]
btc = pd.concat([rd(f) for f in sorted(glob.glob(P + 'BTCUSDT_PERP_1m_*.csv.gz')) + sorted(glob.glob(P + 'R_BTCUSDT_*'))])
btc = btc.dropna(subset=['open_time_ms']).drop_duplicates('open_time_ms', keep='last').sort_values('open_time_ms').rename(columns={'open_time_ms': 'time'})
for d in ('bat', 'zec'): btc.to_csv(f'/home/claude/zbat/{d}/btc1m.csv.gz', index=False); shutil.copy(P + 'BTCUSDT_OI_5m.csv.gz', f'/home/claude/zbat/{d}/btcoi5m.csv.gz')
print('BTC son', pd.to_datetime(btc.time.iloc[-1], unit='ms'))
# BAT
b = pd.concat([rd(f) for f in sorted(glob.glob(P + 'BATUSDT_1m_FULL_*')) + sorted(glob.glob(P + 'R_BATUSDT_*'))]).dropna(subset=['open_time_ms']).drop_duplicates('open_time_ms', keep='last').sort_values('open_time_ms')
b.to_csv('/home/claude/zbat/bat/BATUSDT_1DK_OHLCV_UTC.csv', index=False)
m = pd.read_csv(P + 'BAT_METRICS_5m.csv.gz'); m['create_time_utc'] = m.create_time
m[['create_time_utc', 'sum_open_interest']].to_csv('/home/claude/zbat/bat/oi/BATUSDT_OI_KAYNAK_5DK_UTC.csv', index=False)
m['ms'] = (pd.to_datetime(m.create_time) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
m[['ms', 'sum_toptrader_long_short_ratio', 'count_toptrader_long_short_ratio', 'count_long_short_ratio']].to_pickle('/home/claude/zbat/bat/top5.pkl')
# ZEC
Z = pd.concat([pd.read_parquet(f, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'oi_zec', 'tt_pos_ls', 'tt_acc_ls', 'glb_ls']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
tm = Z.time.astype('int64') if Z.time.dtype.kind in 'iuf' else (pd.to_datetime(Z.time) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
Z['open_time_ms'] = tm.values; Z = Z.drop_duplicates('open_time_ms').sort_values('open_time_ms')
Z[['open_time_ms', 'open', 'high', 'low', 'close', 'volume']].to_csv('/home/claude/zbat/zec/BATUSDT_1DK_OHLCV_UTC.csv', index=False)
z5 = Z[Z.open_time_ms % 300000 == 0]
o = pd.DataFrame({'create_time_utc': pd.to_datetime(z5.open_time_ms, unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S'), 'sum_open_interest': z5.oi_zec.values}).dropna()
o.to_csv('/home/claude/zbat/zec/oi/BATUSDT_OI_KAYNAK_5DK_UTC.csv', index=False)
pd.DataFrame({'ms': z5.open_time_ms.values, 'sum_toptrader_long_short_ratio': z5.tt_pos_ls.values, 'count_toptrader_long_short_ratio': z5.tt_acc_ls.values,
              'count_long_short_ratio': z5.glb_ls.values}).dropna(subset=['sum_toptrader_long_short_ratio']).to_pickle('/home/claude/zbat/zec/top5.pkl')
print('ZEC 1m', len(Z), '· OI 5m', len(o), '· top5', z5.tt_pos_ls.notna().sum())
