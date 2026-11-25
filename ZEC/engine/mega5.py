# Fitil + dar alan girişi derinleştirme (A×G×XMR), dikkat kapıları genel fiyat koşulları (portföy verisi seçimde KULLANILMAZ); sağlamlık dahil
exec(open('/home/claude/zecwork/mega3.py').read().split("a8 = g * 8")[0])
import functools; rng = np.random.default_rng(9); yrs = n / 525600; fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq
F[960] = np.r_[c[960:] / c[:-960] - 1, np.full(960, np.nan)] * 1e4
def regs(PF_):
    K_ = bit(PF_.btcC.values); V_ = bit(PF_.relz.values); A2 = bit(PF_.atrC.values); return ((g * 8 + K_ * 4 + V_ * 2 + A2) * 2 + De) * 2 + Xb
RL = {lg: regs(peerfeat.to1m(FH, B, lg)) for lg in (0, 1, 2)}
sel = functools.lru_cache(None)(lambda lg, thr, hz: select(RL[lg], thr, hz))
WK = {}
for w in (3, 5, 10):
    Hw = pd.Series(h).rolling(w).max().values; Lw = pd.Series(l).rolling(w).min().values; Ow = np.r_[np.full(w - 1, np.nan), o[:-(w - 1)]]; rw = np.maximum(Hw - Lw, 1e-9)
    WK[w] = ((np.minimum(Ow, c) - Lw) / rw, (Hw - np.maximum(Ow, c)) / rw, rw / (a * np.sqrt(w)))
mb = np.searchsorted(idx.values, pd.date_range('2021-01-01', idx[-1], freq='MS').values); mb = np.r_[mb[mb > 0], n - 1]; s2 = p('2026-09-24')
def lin(x): t_ = np.arange(len(x)); return np.corrcoef(t_, x)[0, 1] ** 2 * np.sign(x[-1] - x[0])
XR = np.nan_to_num(xr24)
def gates(L, S, kL, kS, useB, cS, w, wf, nar, thX=6.0, cLm=1):
    cX = XR >= thX
    wl, ws, rr = WK[w]; nz = (rr <= nar) if nar else True
    eL = (L >= kL + cLm * ((cX | BATup) if useB else cX)) & (wkL_ := (wl >= wf)) & nz; eS = (S >= kS + cS * PEPdn) & (ws >= wf) & nz
    return np.asarray(eL, np.bool_), np.asarray(eS, np.bool_)
def run(lg, thr, hz, kL, kS, useB, cS, w, wf, nar, ts, tH, thX=6.0, cLm=1):
    L, S = sel(lg, thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, nar, thX, cLm)
    return z03.sim11(o, h, l, c, a, aL, aS, eL, eS, ts, tH, spk, 60)
rows = []; t0 = time.time()
for thr, hz in itertools.product((20, 25, 30), (480,)):
    for thX, cLm, cS, w, wf, ts, tH in itertools.product((4.0, 6.0, 8.0), (1, 2), (0, 1), (3, 5, 10), (0.3, 0.4, 0.5), (28.0, 32.0, 36.0), (3840, 4800, 5760)):
        kS = 2; useB = 1; nar = 0.0; cf = (thr, hz, 2, kS, useB, cS, w, wf, nar, ts, tH, thX, cLm); eq, T = run(0, *cf)
        if len(T) < 1000 or eq[-1] < 7000: continue
        x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; mp = np.diff(np.r_[0, eq[mb]])[:-1]
        rows.append(dict(thr=thr, thX=thX, cLm=cLm, cS=cS, w=w, wf=wf, dar=nar, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(),
                         DDp99=np.percentile(dd, 99), R2=lin(x30), ayMin=mp.min(), bant=((mp >= -100) & (mp <= 300)).mean() * 100, ayPoz=(mp > 0).mean() * 100, son2ay=eq[-1] - eq[s2],
                         son2ayR2=lin(eq[s2::30]), Kasım=eq[-1] - eq[p('2026-11-01')], short=T[T[:, 2] < 0, 5].sum(), tut=np.mean(T[:, 1] - T[:, 0])))
    print(thr, hz, round(time.time() - t0), len(rows), flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/FITIL2_9500_1DK.csv', index=False); pd.set_option('display.width', 300)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0)]; print('\nen yüksek net', round(f.net.max()), '· ≥9.500:', (f.net >= 9500).sum(), '· ≥9.000:', (f.net >= 9000).sum())
print(f.sort_values('net', ascending=False).head(10).round(2).to_string(index=False))
g2 = f[f.n.between(1400, 2600) & (f.MaxDD <= 450)]; print('\nişlem 1.400–2.600 & MaxDD ≤450:'); print(g2.sort_values('net', ascending=False).head(10).round(2).to_string(index=False))
# Sağlamlık: genel en yüksek net + kısıtlı en yüksek net
out = []
for tag, r in (('NET-MAKS', f.sort_values('net', ascending=False).iloc[0]), ('DENGELİ', g2.sort_values('net', ascending=False).iloc[0] if len(g2) else None)):
    if r is None: continue
    cf = (int(r.thr), 480, 2, 2, 1, int(r.cS), int(r.w), float(r.wf), 0.0, float(r.ts), int(r.tH), float(r.thX), int(r.cLm)); eq, T = run(0, *cf)
    tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]; x30 = eq[::30]
    sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum(); mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
    du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
    inc = np.diff(x30); nb = len(inc) // 48; path = np.cumsum(inc[:nb * 48].reshape(nb, 48)[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1)
    fd = (np.maximum.accumulate(np.c_[np.zeros(2000), path], 1) - np.c_[np.zeros(2000), path]).max(1); I = rng.integers(0, len(tp), (5000, len(tp)))
    out.append(dict(aday=tag, ayar=str(cf), net=eq[-1], monkey='✓' if sysCC > max(mY.max(), mZ.max()) else f'✗ {sysCC:.0f}/{max(mY.max(), mZ.max()):.0f}', MC_P5=np.percentile(np.where(rng.random((5000, len(tp))) > 0.05, tp[I] - 0.05, 0).sum(1), 5),
                    ileri12_P95=np.percentile(fd, 95), ileri12_P99=np.percentile(fd, 99), stres=eq[-1] - (0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q).sum(), gec1=run(1, *cf)[0][-1], gec2=run(2, *cf)[0][-1]))
    np.save(f'/home/claude/zecwork/eq2_{tag}.npy', x30)
print(pd.DataFrame(out).round(1).to_string(index=False))
