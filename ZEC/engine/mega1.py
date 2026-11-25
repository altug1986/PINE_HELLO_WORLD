exec(open('/home/claude/zecwork/reg6_head.py').read())
z = np.zeros(n, np.int64); A4 = np.zeros(n); p = lambda s: np.searchsorted(idx.values, np.datetime64(s))
F[960] = np.r_[c[960:] / c[:-960] - 1, np.full(960, np.nan)] * 1e4
xm = B['xmr_close'].values if 'xmr_close' in B.columns else None
print('ZEC fiyat 1 Kas / 23 Kas:', round(c[p('2026-11-01')], 1), round(c[-1], 1), '· xmr var:', xm is not None, flush=True)
Bq = B[['open', 'high', 'low', 'close']]
De = np.nan_to_num(ztf.htf(Bq, 1, '1D', lambda H: (H.close > H.close.ewm(span=20, adjust=False).mean()).astype(float).values)).astype(np.int8)
xr = pd.Series(xm).ffill().values; Xb = np.r_[np.zeros(1441), (xr[1440:-1] / xr[:-1441] > 1)].astype(np.int8)[:n]       # XMR 24 sa getiri > 0 (kapanmış bar)
K = bit(PF.btcC.values); Vh = bit(PF.relz.values); A_ = bit(PF.atrC.values); g = RG + 1
REG = {'A': g * 8 + K * 4 + Vh * 2 + A_, 'A×G': (g * 8 + K * 4 + Vh * 2 + A_) * 2 + De, 'A×XMR': (g * 8 + K * 4 + Vh * 2 + A_) * 2 + Xb,
       'BKS×G×XMR': ((g * 4 + K * 2 + Vh) * 2 + De) * 2 + Xb, 'BKS×XMR': (g * 4 + K * 2 + Vh) * 2 + Xb, 'A×G×XMR': ((g * 8 + K * 4 + Vh * 2 + A_) * 2 + De) * 2 + Xb}
W = {'Ekim': (p('2026-10-01'), p('2026-11-01')), 'Kasım': (p('2026-11-01'), n - 1), 'çöküş3-11': (p('2026-11-03'), p('2026-11-12')), 'son2ay': (p('2026-09-24'), n - 1),
     '2024orta': (p('2024-04-01'), p('2024-10-01')), 'zayıf': (p('2022-06-01'), p('2023-04-01'))}
rows = []; t0 = time.time()
for rn, R in REG.items():
    for thr, hz in itertools.product((5, 10, 15, 25), (120, 240, 480, 960)):
        L, S = select(R, thr, hz)
        for kL, kS, ts, tH in itertools.product((1, 2, 3), (1, 2, 3, 4), (16.0, 24.0, 32.0), (480, 960, 1920, 2880)):
            aL = L >= kL; eq, T = z03.sim10(o, h, l, c, a, aL, (S >= kS) & ~aL, ts, tH, spk, 60, z, A4, 0)
            if not (1200 <= len(T) <= 2800) or eq[-1] < 5000: continue
            x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w = eq[W['2024orta'][0]:W['2024orta'][1]]
            r = dict(rej=rn, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=np.percentile(dd, 98),
                     dd2024=(np.maximum.accumulate(w) - w).max(), top3=np.sort(T[:, 5])[-3:].sum() / eq[-1] * 100, short=T[T[:, 2] < 0, 5].sum(), tutDk=np.mean(T[:, 1] - T[:, 0]))
            for k_, (a_, b_) in W.items(): r[k_] = eq[b_] - eq[a_]
            rows.append(r)
    print(rn, round(time.time() - t0), len(rows), flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/MEGA_8000_1DK.csv', index=False); pd.set_option('display.width', 290)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0)]
print('\nRejime göre en yüksek net (iki yarı +):'); print(f.groupby('rej').net.agg(['max', 'count']).round(0).sort_values('max', ascending=False).to_string())
print('\nnet ≥ 8.000 (iki yarı +):', (f.net >= 8000).sum()); print(f[f.net >= 8000].sort_values('net', ascending=False).head(15).round(1).to_string(index=False))
h_ = f[(f.net >= 7000) & (f.Kasım > 0) & (f['çöküş3-11'] > 0)]; print('\nnet ≥ 7.000 + Kasım + ve çöküş + :', len(h_)); print(h_.sort_values('Kasım', ascending=False).head(15).round(1).to_string(index=False))
