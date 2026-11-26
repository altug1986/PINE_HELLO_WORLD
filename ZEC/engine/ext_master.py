# MASTER 2026 uzatma: 24 Kas perp tam (günlük arşiv) + 25 Kas 00:00 → 26 Kas spot vekil (OI/metrics/defter/index YOK → NaN). build_master.py kurallarıyla.
import numpy as np, pandas as pd, glob
S = '/home/claude/pine_hello_world/data/sync2611/'; f32 = np.float32; M5 = 300_000
old = pd.read_parquet('/home/claude/master_old/ZECUSDT_MASTER_1m_SYNC_2026.parquet'); cols = list(old.columns)
rd = lambda f: pd.read_csv(S + f)
z = rd('ZECUSDT_PERP_1m_UTC_20261123_20261125.csv.gz').drop_duplicates('time').set_index('time').sort_index()
sp = rd('ZECUSDT_SPOT_PROXY_1m_UTC_20261125_20261126.csv.gz').drop_duplicates('time').set_index('time').sort_index()
# denetim: 23 Kas arşiv ↔ master
o23 = old.set_index('time').loc[z.index[0]:z.index[0] + 86_340_000]; zz = z.reindex(o23.index)
print('23 Kas kapanış eşleşme %', round(float((np.abs(zz.close - o23.close) < 1e-6).mean() * 100), 2), '| hacim', round(float((np.abs(zz.volume - o23.volume) < 1e-3).mean() * 100), 2))
def asof(st, v, cut, tol, T):
    o = np.argsort(st); st = st[o]; v = v[o]; i = np.searchsorted(st, cut, side='right') - 1; ok = i >= 0; i2 = np.where(ok, i, 0); age = np.where(ok, cut - st[i2], np.nan); ok &= age <= tol
    return np.where(ok[:, None] if v.ndim == 2 else ok, v[i2], np.nan), np.where(ok, (age + (T - cut)) / 1000, np.nan)
def blk(z, perp=True):
    T = z.index.values + 60_000; X = pd.DataFrame(index=z.index)
    for c in ('open', 'high', 'low', 'close', 'volume'): X[c] = z[c].astype(f32)
    for c in ('quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'): X[c] = z[c].astype(f32)
    X['taker_delta_base'] = (2 * z.taker_buy_base - z.volume).astype(f32)
    if perp:
        ix = rd('ZECUSDT_INDEX_1m_UTC_20261123_20261125.csv.gz').drop_duplicates('time').set_index('time').close.reindex(z.index)
        pr = rd('ZECUSDT_PREMIUM_1m_UTC_20261123_20261125.csv.gz').drop_duplicates('time').set_index('time').close.reindex(z.index)
        X['index_close'] = ix.astype(f32); X['basis_bps'] = ((z.close / ix - 1) * 1e4).astype(f32); X['premium_bps'] = (pr * 1e4).astype(f32)
        mt = rd('ZECUSDT_METRICS_5m_RAW_20261123_20261125.csv'); ms = ((pd.to_datetime(mt.create_time) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).values
        MC = {'sum_open_interest': 'oi_zec', 'sum_open_interest_value': 'oi_usdt', 'count_toptrader_long_short_ratio': 'tt_acc_ls', 'sum_toptrader_long_short_ratio': 'tt_pos_ls', 'count_long_short_ratio': 'glb_ls', 'sum_taker_long_short_vol_ratio': 'taker_ls_5m'}
        v, age = asof(ms, mt[list(MC)].values.astype(float), T - M5, 20 * 60_000, T)
        for j, c in enumerate(MC.values()): X[c] = v[:, j].astype(f32)
        X['metrics_age_s'] = age.astype(f32)
        bk = rd('ZECUSDT_BOOKDEPTH_1m_UTC_20261123_20261125.csv.gz'); BC = [c for c in bk.columns if c.startswith('notional_')]
        v, age = asof(bk.snap_s.values * 1000, bk[BC].values.astype(float), T - 1, 120_000, T)
        for j, c in enumerate(BC): X['book_' + c.replace('notional_', '')] = v[:, j].astype(f32)
        X['book_age_s'] = age.astype(f32)
        wh = rd('ZECUSDT_WHALE_PRINTS_1m_UTC_20261123_20261125.csv.gz').groupby('time').sum().reindex(z.index).fillna(0)
        for c in wh.columns: X['wh_' + c] = wh[c].astype(f32)
        for s_, k in (('BTCUSDT', 'btc_close'), ('XMRUSDT', 'xmr_close')): X[k] = rd(f'{s_}_PERP_1m_UTC_20261123_20261125.csv.gz').drop_duplicates('time').set_index('time').close.reindex(z.index).astype(f32)
    else:
        for k in ('btc_close',): X[k] = rd('BTCUSDT_SPOT_PROXY_1m_UTC_20261125_20261126.csv.gz').drop_duplicates('time').set_index('time').close.reindex(z.index).astype(f32)
        for c in [c for c in cols if c.startswith('wh_')]: X[c] = f32(0)
    return X.reindex(columns=[c for c in cols if c != 'time'])
A = blk(z); Bp = blk(sp, perp=False)
o23x = A.loc[o23.index]
for c in ('oi_zec', 'tt_pos_ls', 'book_m1', 'wh_buy_q10k', 'basis_bps', 'premium_bps', 'btc_close', 'xmr_close'):
    a_, b_ = o23x[c].astype(float), o23[c].astype(float); m = a_.notna() & b_.notna(); print(f'23 Kas {c}: eşleşme %{float((np.abs(a_[m] - b_[m]) <= 1e-4 * np.abs(b_[m]) + 1e-6).mean() * 100):.1f} (n={int(m.sum())}, NaN yeni/eski {int(a_.isna().sum())}/{int(b_.isna().sum())})')
t24 = 1795478400000; t25 = 1795564800000
N = pd.concat([old.set_index('time'), A.loc[t24:t25 - 60_000], Bp.loc[t25:]]); N = N[~N.index.duplicated(keep='first')].sort_index()
N['src'] = np.where(N.index >= t25, 'spot_vekil', 'perp'); N.index.name = 'time'
print('yeni son', pd.to_datetime(N.index[-1], unit='ms'), 'satır', len(N), '| 24 Kas', len(A.loc[t24:t25 - 60_000]), '| vekil', len(Bp.loc[t25:]))
N.reset_index().to_parquet('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_2026.parquet', compression='zstd', index=False)
