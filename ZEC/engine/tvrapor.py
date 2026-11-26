# Motorla TV formatında rapor: V3.2 vs V3.3 (TV penceresi 2023-02-03 → 2026-11-23 ve tüm geçmiş)
exec(open('/home/claude/zecwork/tvsync.py').read().split("T = pd.read_csv")[0])
@njit(cache=False)
def trades(o, c, rp, rx, rn):
    n = len(c); out = np.zeros((n // 50, 7)); k = 0; real = 0; ep = 0.0; ei = 0; q = 0.0; mae = 0.0; hi_ = 0.0
    for i in range(1, n):
        if real != 0 and rx[i] > 0 and real == rp[i - 1]:
            xp = rx[i]; out[k, 0] = ei; out[k, 1] = i; out[k, 2] = real; out[k, 3] = ep; out[k, 4] = xp; out[k, 5] = q * (xp - ep) * real - 0.0004 * q * (ep + xp); out[k, 6] = mae; k += 1; real = 0
        if rp[i] != real:
            if real != 0:
                xp = c[i] - 0.01 * real; out[k, 0] = ei; out[k, 1] = i; out[k, 2] = real; out[k, 3] = ep; out[k, 4] = xp; out[k, 5] = q * (xp - ep) * real - 0.0004 * q * (ep + xp); out[k, 6] = mae; k += 1; real = 0
            if rp[i] != 0: real = rp[i]; ep = c[i] + 0.01 * real; q = 500.0 / c[i]; ei = i; mae = 0.0
        if real != 0: mae = min(mae, (c[i] / ep - 1) * real)
    return out[:k]
T = pd.read_csv('/home/claude/tvx/v32.csv'); g = T[T.Tip.str.contains('giriş')]; t0 = pd.to_datetime(g['Tarih ve saat']).min() - pd.Timedelta('3h')
i0 = np.searchsorted(ixh.values, t0.to_datetime64()); rows = []
for lab, cL, cS in (('V3.2', 0.10, 0.10), ('V3.3', 0.10, 0.04)):
    e, _, _ = real(cL, cS)
    eqb, Tb = simY(*args, *XV, TS6, THv, spk, 60, 3, byp, cL, cS, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR0 = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR0)
    rp, rx, rn = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08); TT = trades(args[0], cc, rp, rx, rn)
    for win, a_ in (('TV penceresi', i0), ('tüm geçmiş', 0)):
        k = TT[:, 0] >= a_; pn = TT[k, 5]; sh = TT[k, 2] < 0; x = e[a_::30] - e[a_]; d = np.maximum.accumulate(x) - x
        cl = np.cumsum(pn); cdd = (np.maximum.accumulate(cl) - cl).max(); dd_d = pd.Series(e[a_:], ixh[a_:]).resample('1D').last(); dd_d = (dd_d.cummax() - dd_d)
        wk = pd.Series(e[a_:], ixh[a_:]).resample('W').last().diff().dropna()
        rows.append(dict(sürüm=lab, pencere=win, net=pn.sum(), işlem=len(pn), PF=pn[pn > 0].sum() / -pn[pn < 0].sum(), kazanma=(pn > 0).mean() * 100, long=pn[~sh].sum(), short=pn[sh].sum(),
                         MaxDD_kapanış=cdd, MaxDD_mtm=d.max(), DD95=dd_d.quantile(0.95), DD98=dd_d.quantile(0.98), DD99=dd_d.quantile(0.99), Sharpe=wk.mean() / wk.std() * np.sqrt(52), Sortino=wk.mean() / wk[wk < 0].std() * np.sqrt(52),
                         enKötüLong=pn[~sh].min(), enKötüShort=pn[sh].min(), shortMAE_P99=np.percentile(TT[k][sh, 6], 1) * 100))
R = pd.DataFrame(rows); pd.set_option('display.width', 320); print(R.round(2).to_string(index=False)); R.to_csv('/home/claude/zecwork/tvrapor.csv', index=False)
print(f"TV V3.2 gerçek (aynı pencere, 11-23'e kadar): net 5.495,5 · 1.497 giriş · PF {(lambda p: p[p>0].sum()/-p[p<0].sum())(g[pd.to_datetime(g['Tarih ve saat']) <= '2026-11-24 02:59']['Net K/Z USDT'].astype(float)):.3f}")
