exec(open('/home/claude/zecwork/reg6_head.py').read())
import functools; rng = np.random.default_rng(11); z = np.zeros(n, np.int64); A4 = np.zeros(n); yrs = n / 525600
WIN['23Kas'] = (pos_('2026-11-23'), n - 1)
b6 = np.searchsorted(idx.values, pd.date_range('2021-01-01', idx[-1], freq='6MS').values); b6 = b6[b6 > 0]
def regs(PF_):
    K = bit(PF_.btcC.values); Vh = bit(PF_.relz.values); A_ = bit(PF_.atrC.values); Pk = bit(PF_.peerAtrZ.values)
    return {'baz': RG + 1, 'BTCkor×sıcak×ATRkor': (RG + 1) * 8 + K * 4 + Vh * 2 + A_, 'sıcak×piyasaVol': (RG + 1) * 4 + Vh * 2 + Pk, 'BTCkor×sıcak': (RG + 1) * 4 + K * 2 + Vh}
RL = {lg: regs(peerfeat.to1m(FH, B, lg)) for lg in (0, 1, 2)}
sel = functools.lru_cache(None)(lambda lg, rn, thr, hz: select(RL[lg][rn], thr, hz))
def run(lg, rn, thr, hz, kL, kS, ts, tH):
    L, S = sel(lg, rn, thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL; return z03.sim10(o, h, l, c, a, aL, aS, ts, tH, spk, 60, z, A4, 0)
CAND = {'A': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 24.0, 960), 'B': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 24.0, 1920), 'C': ('BTCkor×sıcak×ATRkor', 10, 480, 3, 3, 24.0, 960),
        'D': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 16.0, 1920), 'E': ('BTCkor×sıcak×ATRkor', 25, 480, 2, 3, 24.0, 960), 'F': ('sıcak×piyasaVol', 15, 480, 2, 3, 24.0, 1920),
        'G': ('BTCkor×sıcak', 15, 480, 2, 3, 24.0, 960), 'R5-eski': ('baz', 15, 240, 2, 3, 16.0, 1920)}
def mdd(x): return (np.maximum.accumulate(x, axis=-1) - x).max(axis=-1)
def flat(x):
    sm = np.minimum.accumulate(x[::-1])[::-1]; s = np.searchsorted(sm, x, side='right'); return (s - np.arange(len(x))).max() * 30 / 1440
out = []; fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq
for nm, cf in CAND.items():
    eq, T = run(0, *cf); tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]
    x30 = eq[::30]; d30 = np.maximum.accumulate(x30) - x30; bp = np.diff(np.r_[0, eq[b6], eq[-1]])
    r = dict(aday=nm, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=d30.max(), DDp98=np.percentile(d30, 98), Calmar=eq[-1] / yrs / d30.max(), altıAyPoz=f'{(bp > 0).sum()}/{len(bp)}')
    I = rng.integers(0, len(tp), (5000, len(tp))); P_ = np.where(rng.random((5000, len(tp))) > 0.05, tp[I] - 0.05, 0.0); cs = np.cumsum(P_, 1)
    r['MC_netP5'] = np.percentile(cs[:, -1], 5); r['MC_zarar%'] = (cs[:, -1] < 0).mean() * 100
    inc = np.diff(x30); nb = len(inc) // 48; path = np.cumsum(inc[:nb * 48].reshape(nb, 48)[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1); fd = mdd(np.c_[np.zeros(2000), path])
    r['12ay_DDp95'] = np.percentile(fd, 95); r['12ay_DDp99'] = np.percentile(fd, 99); r['12ay_zarar%'] = (path[:, -1] < 0).mean() * 100
    sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum()
    mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
    du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]
    mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
    r.update(sysCC=sysCC, mYönMaks=mY.max(), mZamanMaks=mZ.max(), monkeyGeçti=bool(sysCC > mY.max() and sysCC > mZ.max()))
    xc = 0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q; r['stresNet'] = eq[-1] - xc.sum()
    r['gecikme1s'] = run(1, *cf)[0][-1]; r['gecikme2s'] = run(2, *cf)[0][-1]
    r['yatayGün'] = flat(x30); r['top3%'] = np.sort(tp)[-3:].sum() / eq[-1] * 100; r['short'] = tp[d_ < 0].sum()
    for k_, (a_, b_) in WIN.items(): r[k_] = eq[b_] - eq[a_]
    out.append(r); np.save(f'/home/claude/zecwork/eq_{nm}.npy', x30)
Rr = pd.DataFrame(out); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/REJIM_AKRAN_SAGLAMLIK.csv', index=False)
pd.set_option('display.width', 250); print(Rr.set_index('aday').T.to_string())
G = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/REJIM_AKRAN_1DK.csv'); G = G[G.n.between(1400, 2300)]
bi = G.sort_values('IS', ascending=False).iloc[0]; print('\nGama: IS en iyisi →', bi[['rej', 'thr', 'hz', 'kL', 'kS', 'ts', 'tH', 'IS', 'OOS', 'net']].to_dict(), '· OOS yüzdelik:', round((G.OOS < bi.OOS).mean() * 100, 1))
a = CAND['A']; nb_ = G[(G.rej == a[0]) & (G.thr.isin([10, 15, 25])) & (G.hz.isin([240, 480])) & (G.kL.isin([2, 3])) & (G.kS.isin([2, 3, 4])) & (G.ts.isin([16.0, 24.0])) & (G.tH.isin([960, 1920]))]
print('A komşuları:', len(nb_), 'medyan net', round(nb_.net.median()), 'iki yarı pozitif', f'{((nb_.IS > 0) & (nb_.OOS > 0)).sum()}/{len(nb_)}', '≥5.000:', (nb_.net >= 5000).sum())
