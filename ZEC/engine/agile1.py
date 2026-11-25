exec(open('/home/claude/zecwork/reg6_head.py').read())
z = np.zeros(n, np.int64); A4 = np.zeros(n)
K = bit(PF.btcC.values); Vh = bit(PF.relz.values); A_ = bit(PF.atrC.values); Pk = bit(PF.peerAtrZ.values)
REG = {'BTCkor×sıcak×ATRkor': (RG + 1) * 8 + K * 4 + Vh * 2 + A_, 'BTCkor×sıcak': (RG + 1) * 4 + K * 2 + Vh, 'sıcak×piyasaVol': (RG + 1) * 4 + Vh * 2 + Pk, 'BTCkor': (RG + 1) * 2 + K, 'baz': RG + 1}
Sp = pd.read_pickle('/home/claude/port/portfoy_10dk_utc.pkl'); Pd = Sp.resample('1D').last().diff()
days = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq='1D'); di = np.searchsorted(idx.values, days.values); di = np.r_[di[1:], n - 1]   # gün sonu
Pd = Pd.reindex(days); ov = Pd.notna().values & (days >= pd.Timestamp('2023-05-09')) & (days < pd.Timestamp('2026-11-13'))
pv = Pd.values[ov]; worst = pv <= np.percentile(pv, 10); l90 = ov & (days >= pd.Timestamp('2026-08-14'))
p = lambda s: np.searchsorted(idx.values, np.datetime64(s)); W24 = (p('2024-04-01'), p('2024-10-01')); S30 = (p('2026-10-24'), n - 1)
rows = []; t0 = time.time()
for rn, R in REG.items():
    for thr, hz in itertools.product((10, 15, 25), (120, 240, 480)):
        L, S = select(R, thr, hz)
        for kL, kS, ts, tH in itertools.product((2, 3), (2, 3, 4), (4.0, 8.0, 16.0, 24.0), (240, 480, 960, 1920)):
            aL = L >= kL; aS = (S >= kS) & ~aL
            eq, T = z03.sim10(o, h, l, c, a, aL, aS, ts, tH, spk, 60, z, A4, 0)
            if len(T) < 800: continue
            x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w = eq[W24[0]:W24[1]]
            zd = np.diff(np.r_[eq[di[0]], eq[di]])[: len(days)]; zd = np.diff(eq[np.r_[0, di]])
            zo = zd[ov]
            r = dict(rej=rn, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=np.percentile(dd, 98),
                     dd2024=(np.maximum.accumulate(w) - w).max(), son30=eq[S30[1]] - eq[S30[0]], son2ay=eq[-1] - eq[p('2026-09-24')], zayıf=eq[p('2023-04-01')] - eq[p('2022-06-01')],
                     korPort=np.corrcoef(zo, pv)[0, 1], korPort90=np.corrcoef(zd[l90], Pd.values[l90])[0, 1], portKötüGün=zo[worst].mean(), tutDk=np.mean(T[:, 1] - T[:, 0]),
                     maeP95=np.percentile(T[:, 6], 95) * 100, top3=np.sort(T[:, 5])[-3:].sum() / max(eq[-1], 1) * 100, short=T[T[:, 2] < 0, 5].sum())
            rows.append(r)
    print(rn, round(time.time() - t0), flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/CEVIK_PORTFOY_1DK.csv', index=False); pd.set_option('display.width', 280)
ref = Rr[(Rr.rej == 'BTCkor×sıcak×ATRkor') & (Rr.thr == 10) & (Rr.hz == 480) & (Rr.kL == 3) & (Rr.kS == 3) & (Rr.ts == 24) & (Rr.tH == 960)]
print('\nAKRAN-C ref:\n', ref.round(2).to_string(index=False))
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0) & (Rr.MaxDD <= 450) & (Rr.son30 > 0) & (Rr.zayıf > 0) & Rr.n.between(1300, 2600)]
print(f'\nFiltre (iki yarı +, MaxDD ≤450, son 30 gün +, zayıf +, işlem 1.300–2.600): {len(f)} aday')
print(f.sort_values('net', ascending=False).head(20).round(2).to_string(index=False))
g = f[(f.dd2024 <= 150) & (f.korPort <= 0.05)]; print(f'\n+ 2024 ortası DD ≤150 ve portföy kor ≤0,05: {len(g)}'); print(g.sort_values('net', ascending=False).head(15).round(2).to_string(index=False))
