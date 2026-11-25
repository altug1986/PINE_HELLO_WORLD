# TV-YERLİ model: yalnız TradingView'de hesaplanabilen oylar (OHLCV + BTC/XMR/BAT/PEPE fiyatı + mum yönü hacim akışı), 9.128 hedefi
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
from numba import njit
sh_ = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
sgn = lambda x, b=0.0: np.where(x > b, 1, np.where(x < -b, -1, 0)).astype(np.float32)
emaP = lambda x, n_: pd.Series(x).ewm(span=n_, adjust=False).mean().values
vol = B['volume'].values.astype(float); bt = B['btc_close'].ffill().values.astype(float); xmv = B['xmr_close'].ffill().values.astype(float)
@njit(cache=True)
def stl(h, l, c, a, m):
    n = len(c); d = np.ones(n, np.float32); fu = np.empty(n); fd = np.empty(n); hl = (h + l) / 2
    fu[0] = hl[0] - m * a[0]; fd[0] = hl[0] + m * a[0]
    for i in range(1, n):
        up = hl[i] - m * a[i]; dn = hl[i] + m * a[i]
        fu[i] = max(up, fu[i - 1]) if c[i - 1] > fu[i - 1] else up
        fd[i] = min(dn, fd[i - 1]) if c[i - 1] < fd[i - 1] else dn
        d[i] = 1 if c[i] > fd[i - 1] else (-1 if c[i] < fu[i - 1] else d[i - 1])
    return d
an = np.nan_to_num(a, nan=np.nanmedian(a))
V['P01_ema'] = sgn(np.nan_to_num(emaP(c, 40) - emaP(c, 120)))
V['P02_st'] = stl(h, l, c, an, 3 * np.sqrt(2))
V['P03_htf30'] = sgn(np.nan_to_num(ztf.htf(B[['open', 'high', 'low', 'close']], 1, '30min', lambda H: np.r_[np.nan, np.diff(emaP(H.close.values, 20))])))
HH = pd.Series(h).rolling(240).max().shift(1).values; LL = pd.Series(l).rolling(240).min().shift(1).values
V['P05_don'] = sgn(np.nan_to_num(c - (HH + LL) / 2))
V['P06_roc'] = sgn(np.nan_to_num((c - sh_(c, 120)) / a), np.sqrt(2))
V['P13_btc'] = sgn(np.nan_to_num((bt / sh_(bt, 60) - 1) * 100), 0.1 * np.sqrt(2))
V['P14_xmr'] = sgn(np.nan_to_num((xmv / sh_(xmv, 60) - 1) * 100), 0.2 * np.sqrt(2))
sv = np.sign(c - o) * vol; V['X16_flow'] = sgn(np.nan_to_num(pd.Series(sv).rolling(20).sum().values / pd.Series(vol).rolling(20).sum().values), 0.15)
TVO = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h', 'P01_ema', 'P02_st', 'P03_htf30', 'P05_don', 'P06_roc', 'P13_btc', 'P14_xmr', 'X16_flow']
order = TVO; sel.cache_clear()
np.savez_compressed('/home/claude/zecwork/tv_V.npz', **{k: np.asarray(V[k], np.int8) for k in TVO})
rows = []; t0 = time.time()
for thr, hz in itertools.product((10, 15, 25), (240, 480, 960)):
    for kL, kS, ts, tH in itertools.product((1, 2, 3), (1, 2, 3), (16.0, 28.0, 40.0), (960, 1920, 3840)):
        cf = (thr, hz, kL, kS, 1, 0, 10, 0.4, 0.0, ts, tH, 4.0, 2); eq, T = run(0, *cf)
        if len(T) < 500: continue
        x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; mp = np.diff(np.r_[0, eq[mb]])[:-1]
        rows.append(dict(thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99),
                         R2=lin(x30), ayMin=mp.min(), Kasım=eq[-1] - eq[p('2026-11-01')], son2ay=eq[-1] - eq[s2], short=T[T[:, 2] < 0, 5].sum()))
    print(thr, hz, round(time.time() - t0), flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/TVYERLI_1_1DK.csv', index=False); pd.set_option('display.width', 250)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0)]; print('en yüksek net', round(f.net.max()), '≥9.128:', (f.net >= 9128).sum())
print(f.sort_values('net', ascending=False).head(15).round(2).to_string(index=False))
