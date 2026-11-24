"""ZECUSDT.P MASTER 1m — tüm veri tek saatte (tam senkron).

Satır = 1 dk bar, `time` = bar AÇILIŞI (UTC ms). Satırdaki HER değer bar KAPANIŞINDA (T = time + 60 s) bilinen bilgidir.
Kural sözlüğü: ZEC/README.md §2.1. Çalıştır: python3 -I build_master.py <data_dir> <out_dir>
"""
import sys, glob, json, numpy as np, pandas as pd

D, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
M5 = 300_000          # metrics güvenlik payı: damga <= T - 5 dk
f32 = np.float32


def rd(pat, **k):
    fs = sorted(glob.glob(D + pat))
    assert fs, pat
    return pd.concat([pd.read_csv(f, **k) for f in fs], ignore_index=True)


# ---------- 1) ızgara: ZEC 1m OHLCV ----------
z = rd('ZECUSDT_1m_UTC_*.csv.gz').drop_duplicates('time').sort_values('time').set_index('time')
T = z.index.values + 60_000                    # bilgi kesimi (bar kapanışı)
X = z[['open', 'high', 'low', 'close', 'volume']].astype(f32)

# ---------- 2) aynı bar: akış, index, premium ----------
fl = rd('ZECUSDT_1m_FLOW_UTC_*.csv.gz').drop_duplicates('time').set_index('time').reindex(z.index)
for c in ['quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']:
    X[c] = fl[c].astype(f32)
X['taker_delta_base'] = (2 * fl.taker_buy_base - z.volume).astype(f32)
for pat, c in [('ZECUSDT_INDEX_PRICE_1m_*', 'index_close'), ('ZECUSDT_PREMIUM_INDEX_1m_*', 'premium_close')]:
    s = rd(pat).drop_duplicates('time').set_index('time')[c].reindex(z.index)
    X[c] = s.astype(f32) if c == 'index_close' else s.astype('float64')
X['basis_bps'] = ((z.close / X.index_close.astype('float64') - 1) * 1e4).astype(f32)
X['premium_bps'] = (X.premium_close * 1e4).astype(f32); X = X.drop(columns='premium_close')


def asof(stamp_ms, vals, cut_ms, tol_ms):
    """cut_ms anında bilinen son değer: damga <= cut. Döner: değerler, yaş (s)."""
    o = np.argsort(stamp_ms); st = stamp_ms[o]; v = vals[o]
    i = np.searchsorted(st, cut_ms, side='right') - 1
    ok = i >= 0; i2 = np.where(ok, i, 0)
    age = np.where(ok, cut_ms - st[i2], np.nan)
    ok &= age <= tol_ms
    out = np.where(ok[:, None] if v.ndim == 2 else ok, v[i2], np.nan)
    return out, np.where(ok, (age + (T - cut_ms)) / 1000, np.nan)     # yaş = T'ye göre


# ---------- 3) funding: olay anında bilinir ----------
fu = pd.read_csv(glob.glob(D + 'ZECUSDT_FUNDING_EVENTS_*.csv')[0])
v, _ = asof(fu.funding_time_ms.values, fu.funding_rate_percent.values.astype(float), T, 9 * 3_600_000)
X['funding_pct'] = v.astype(f32)

# ---------- 4) metrics 5m (son etiketli) → damga <= T - 5 dk ----------
mt = rd('ZECUSDT_METRICS_5m_UTC_*.csv.gz').drop_duplicates('timestamp_ms')
mt.loc[mt.open_interest_zec <= 0, ['open_interest_zec', 'open_interest_usdt']] = np.nan
MC = {'open_interest_zec': 'oi_zec', 'open_interest_usdt': 'oi_usdt', 'top_trader_accounts_ls': 'tt_acc_ls',
      'top_trader_positions_ls': 'tt_pos_ls', 'global_accounts_ls': 'glb_ls', 'taker_long_short_vol_ratio': 'taker_ls_5m'}
v, age = asof(mt.timestamp_ms.values, mt[list(MC)].values.astype(float), T - M5, 20 * 60_000)
for j, c in enumerate(MC.values()):
    X[c] = v[:, j].astype(f32)
X['metrics_age_s'] = age.astype(f32)

# ---------- 5) emir defteri derinliği: aynı dakikadaki son anlık görüntü (snap < T) ----------
bk = rd('ZECUSDT_BOOKDEPTH_1m_UTC_*.csv.gz').drop_duplicates('time')
BC = [c for c in bk.columns if c.startswith('notional_')]
v, age = asof(bk.snap_s.values * 1000, bk[BC].values.astype(float), T - 1, 120_000)
for j, c in enumerate(BC):
    X['book_' + c.replace('notional_', '')] = v[:, j].astype(f32)
X['book_age_s'] = age.astype(f32)

# ---------- 6) balina printleri: bar içi işlemler (seyrek → 0) ----------
wh = rd('ZECUSDT_WHALE_PRINTS_1m_UTC_*.csv.gz').groupby('time').sum()   # yıl sınırı tekrarına karşı
WC = list(wh.columns)
w = wh.reindex(z.index).fillna(0)
for c in WC:
    X['wh_' + c] = w[c].astype(f32)

# ---------- 7) çapraz piyasa: BTC, XMR 1m kapanış (aynı bar) + OI (metrics kuralı) ----------
def closes(pats):
    parts = []
    for p in pats:
        for f in sorted(glob.glob(D + p)):
            d = pd.read_csv(f)
            t = 'time' if 'time' in d else 'open_time'
            parts.append(d[[t, 'close']].rename(columns={t: 'time'}))
    s = pd.concat(parts).drop_duplicates('time', keep='last').set_index('time').close
    return s.reindex(z.index).astype(f32)
X['btc_close'] = closes(['BTCUSDT_PERP_1m_2019_*.csv.gz', 'BTCUSDT_1m_UTC_2026*'])
X['xmr_close'] = closes(['XMRUSDT_1m_FULL_THROUGH_*.csv.gz', 'XMRUSDT_1m_UTC_2026*'])
bo = pd.read_csv(glob.glob(D + 'BTCUSDT_OI_5m_UTC_*.csv.gz')[0], usecols=['timestamp_utc', 'open_interest_qty'])
bms = (pd.to_datetime(bo.timestamp_utc).dt.tz_localize(None) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
v, _ = asof(bms.values, bo.open_interest_qty.where(bo.open_interest_qty > 0).values.astype(float), T - M5, 20 * 60_000)
X['btc_oi'] = v.astype(f32)
xo = pd.read_csv(glob.glob(D + 'XMRUSDT_METRICS_5m_UTC_*.csv.gz')[0], usecols=['timestamp_ms', 'open_interest_xmr'])
v, _ = asof(xo.timestamp_ms.values, xo.open_interest_xmr.where(xo.open_interest_xmr > 0).values.astype(float), T - M5, 20 * 60_000)
X['xmr_oi'] = v.astype(f32)

# ---------- 8) senkron denetimi ----------
q = X.quote_volume.astype('float64'); tb = X.taker_buy_quote.astype('float64')
A = {}
A['rows'] = len(X); A['start'] = str(pd.to_datetime(X.index[0], unit='ms')); A['end'] = str(pd.to_datetime(X.index[-1], unit='ms'))
A['taker_buy_gt_volume'] = int((X.taker_buy_base > X.volume * 1.0001).sum())
A['whale_buy_gt_taker_buy_quote_pct'] = round(float(((X.wh_buy_q10k > tb * 1.01 + 1) & (X.wh_buy_q10k > 0)).sum() / max(1, (X.wh_buy_q10k > 0).sum()) * 100), 3)
A['whale_sell_gt_taker_sell_quote_pct'] = round(float(((X.wh_sell_q10k > (q - tb) * 1.01 + 1) & (X.wh_sell_q10k > 0)).sum() / max(1, (X.wh_sell_q10k > 0).sum()) * 100), 3)
# kaydırma testi: balina alışı taker alışıyla hangi gecikmede en çok örtüşüyor (0 = senkron)
ws = pd.Series(X.wh_buy_q10k.values); ts = pd.Series(tb.values)
A['whale_vs_takerbuy_corr_lag'] = {k: round(float(ws.corr(ts.shift(k))), 4) for k in (-2, -1, 0, 1, 2)}
r = np.log(X.close.astype('float64')).diff()
for c in ('btc_close', 'xmr_close'):
    rr = np.log(X[c].astype('float64')).diff()
    A[c + '_ret_corr_lag'] = {k: round(float(r.corr(rr.shift(k))), 4) for k in (-2, -1, 0, 1, 2)}
A['book_snap_inside_bar_pct'] = round(float(((X.book_age_s >= 0) & (X.book_age_s <= 60)).sum() / X.book_age_s.notna().sum() * 100), 3)
A['metrics_age_s_P50_P99'] = [float(np.nanpercentile(X.metrics_age_s, 50)), float(np.nanpercentile(X.metrics_age_s, 99))]
A['coverage'] = {c: round(float(X[c].notna().mean()), 4) for c in X.columns}
A['first_valid'] = {c: str(pd.to_datetime(X[c].first_valid_index(), unit='ms')) for c in
                    ['oi_zec', 'tt_pos_ls', 'book_m1', 'book_m02', 'btc_oi', 'xmr_oi', 'xmr_close']}
A['last_valid'] = {c: str(pd.to_datetime(X[c].last_valid_index(), unit='ms')) for c in
                   ['funding_pct', 'btc_close', 'xmr_close', 'btc_oi', 'xmr_oi', 'book_m1']}
json.dump(A, open(OUT + 'MASTER_SENKRON_DENETIM.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in A.items() if k != 'coverage'}, indent=1))

# ---------- 9) yaz: yıllık parquet (zstd) ----------
X.index.name = 'time'
yr = pd.to_datetime(X.index, unit='ms').year
for y in sorted(set(yr)):
    s = X[yr == y]
    s.reset_index().to_parquet(OUT + f'ZECUSDT_MASTER_1m_SYNC_{y}.parquet', compression='zstd', index=False)
print('kolon', len(X.columns), list(X.columns))
