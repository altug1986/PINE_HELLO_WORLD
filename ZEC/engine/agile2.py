# Günlük yön biti eklenmiş akıllı rejim + portföy korelasyonu + 2024 ortası + son 30 gün; kazananlara sağlamlık bataryası
exec(open('/home/claude/zecwork/agile1.py').read().split("rows = []; t0 = time.time()")[0])
import functools
def e20(H): return (H.close > H.close.ewm(span=20, adjust=False).mean()).astype(float).values
def r7(H): return (H.close / H.close.shift(7) > 1).astype(float).values
Bq = B[['open', 'high', 'low', 'close']]; De = np.nan_to_num(ztf.htf(Bq, 1, '1D', e20)).astype(np.int8); Dr = np.nan_to_num(ztf.htf(Bq, 1, '1D', r7)).astype(np.int8)
def regs(PF_):
    K = bit(PF_.btcC.values); Vh = bit(PF_.relz.values); A_ = bit(PF_.atrC.values)
    return {'BTCkor×sıcak×ATRkor×G-EMA': (RG + 1) * 16 + K * 8 + Vh * 4 + A_ * 2 + De, 'BTCkor×sıcak×G-EMA': (RG + 1) * 8 + K * 4 + Vh * 2 + De,
            'BTCkor×sıcak×G-7g': (RG + 1) * 8 + K * 4 + Vh * 2 + Dr, 'sıcak×G-EMA': (RG + 1) * 4 + Vh * 2 + De, 'BTCkor×G-EMA': (RG + 1) * 4 + K * 2 + De,
            'BTCkor×sıcak×ATRkor': (RG + 1) * 8 + K * 4 + Vh * 2 + A_}
RL = {lg: regs(peerfeat.to1m(FH, B, lg)) for lg in (0, 1, 2)}; REG = RL[0]
sel = functools.lru_cache(None)(lambda lg, rn, thr, hz: select(RL[lg][rn], thr, hz))
def met(eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w = eq[W24[0]:W24[1]]; zd = np.diff(eq[np.r_[0, di]]); zo = zd[ov]
    return dict(n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=np.percentile(dd, 98), dd2024=(np.maximum.accumulate(w) - w).max(),
                son30=eq[S30[1]] - eq[S30[0]], son2ay=eq[-1] - eq[p('2026-09-24')], zayıf=eq[p('2023-04-01')] - eq[p('2022-06-01')], korPort=np.corrcoef(zo, pv)[0, 1],
                portKötüGün=zo[worst].mean(), tutDk=np.mean(T[:, 1] - T[:, 0]), maeP95=np.percentile(T[:, 6], 95) * 100, top3=np.sort(T[:, 5])[-3:].sum() / max(eq[-1], 1) * 100, short=T[T[:, 2] < 0, 5].sum())
rows = []
for rn in REG:
    for thr, hz in itertools.product((10, 15, 25), (120, 240, 480)):
        L, S = sel(0, rn, thr, hz)
        for kL, kS, ts, tH in itertools.product((2, 3), (2, 3, 4), (4.0, 8.0, 16.0, 24.0), (240, 480, 960, 1920)):
            eq, T = z03.sim10(o, h, l, c, a, L >= kL, (S >= kS) & ~(L >= kL), ts, tH, spk, 60, z, A4, 0)
            if len(T) >= 800: rows.append(dict(rej=rn, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, **met(eq, T)))
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/CEVIK_GUNLUK_1DK.csv', index=False); pd.set_option('display.width', 280)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0) & (Rr.MaxDD <= 450) & (Rr.son30 > 0) & (Rr.son2ay > 0) & (Rr.zayıf > 0) & Rr.n.between(1300, 2600) & (Rr.korPort <= 0.1)]
print('Rejim türüne göre filtre geçen / en yüksek net / en düşük 2024 DD:'); print(f.groupby('rej').agg(aday=('net', 'size'), maxNet=('net', 'max'), min2024=('dd2024', 'min')).round(0).to_string())
g = f[f.dd2024 <= 150]; print(f'\n2024 ortası DD ≤150: {len(g)}'); print(g.sort_values('net', ascending=False).head(12).round(2).to_string(index=False))
# Sağlamlık: 2024 ≤150 grubundan en yüksek 3 net
rng = np.random.default_rng(3); fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq; out = []
for _, r in g.sort_values('net', ascending=False).head(3).iterrows():
    cf = (r.rej, int(r.thr), int(r.hz), int(r.kL), int(r.kS), float(r.ts), int(r.tH)); L, S = sel(0, *cf[:4][:1], cf[1], cf[2]) if False else sel(0, cf[0], cf[1], cf[2])
    eq, T = z03.sim10(o, h, l, c, a, L >= cf[3], (S >= cf[4]) & ~(L >= cf[3]), cf[5], cf[6], spk, 60, z, A4, 0)
    tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]; x30 = eq[::30]
    sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum(); mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
    du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
    inc = np.diff(x30); nb = len(inc) // 48; path = np.cumsum(inc[:nb * 48].reshape(nb, 48)[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1)
    fd = (np.maximum.accumulate(np.c_[np.zeros(2000), path], 1) - np.c_[np.zeros(2000), path]).max(1)
    I = rng.integers(0, len(tp), (5000, len(tp))); cs = np.where(rng.random((5000, len(tp))) > 0.05, tp[I] - 0.05, 0.0).sum(1)
    lg = [z03.sim10(o, h, l, c, a, sel(k, cf[0], cf[1], cf[2])[0] >= cf[3], (sel(k, cf[0], cf[1], cf[2])[1] >= cf[4]) & ~(sel(k, cf[0], cf[1], cf[2])[0] >= cf[3]), cf[5], cf[6], spk, 60, z, A4, 0)[0][-1] for k in (1, 2)]
    out.append(dict(ayar=str(cf), net=eq[-1], monkeyYön=f'{sysCC:.0f}>{mY.max():.0f}', monkeyZaman=f'{sysCC:.0f}>{mZ.max():.0f}', geçti=bool(sysCC > max(mY.max(), mZ.max())),
                    MC_P5=np.percentile(cs, 5), ileri12_DDp95=np.percentile(fd, 95), ileri12_DDp99=np.percentile(fd, 99), zarar12ay=(path[:, -1] < 0).mean() * 100,
                    stres=eq[-1] - (0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q).sum(), gecikme1s=lg[0], gecikme2s=lg[1]))
    np.save('/home/claude/zecwork/eq_cevik_%d.npy' % len(out), x30)
print(pd.DataFrame(out).round(1).to_string(index=False))
