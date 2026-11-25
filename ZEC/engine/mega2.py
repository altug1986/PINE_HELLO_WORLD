exec(open('/home/claude/zecwork/mega1.py').read().split("W = {")[0])
import functools; rng = np.random.default_rng(5); yrs = n / 525600; fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq
def regs(PF_):
    K = bit(PF_.btcC.values); Vh = bit(PF_.relz.values); A_ = bit(PF_.atrC.values); a8 = g * 8 + K * 4 + Vh * 2 + A_
    return {'A': a8, 'A×G': a8 * 2 + De, 'A×G×XMR': (a8 * 2 + De) * 2 + Xb, 'A×XMR': a8 * 2 + Xb}
RL = {lg: regs(peerfeat.to1m(FH, B, lg)) for lg in (0, 1, 2)}
sel = functools.lru_cache(None)(lambda lg, rn, thr, hz: select(RL[lg][rn], thr, hz))
def run(lg, rn, thr, hz, kL, kS, ts, tH):
    L, S = sel(lg, rn, thr, hz); aL = L >= kL; return z03.sim10(o, h, l, c, a, aL, (S >= kS) & ~aL, ts, tH, spk, 60, z, A4, 0)
CAND = {'X1': ('A×G×XMR', 25, 480, 2, 3, 32.0, 960), 'X2': ('A×G×XMR', 15, 480, 2, 3, 32.0, 2880), 'X3': ('A×G×XMR', 10, 480, 2, 3, 32.0, 2880),
        'H1': ('A×G', 10, 480, 3, 3, 24.0, 960), 'H2': ('A', 5, 960, 3, 4, 32.0, 960), 'AKRAN-A': ('A', 15, 480, 2, 3, 24.0, 960)}
P = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); P30 = P.resample('30min').last()
i30 = idx[::30]; out = []
for nm, cf in CAND.items():
    eq, T = run(0, *cf); tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]; x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
    r = dict(aday=nm, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp95=np.percentile(dd, 95), DDp99=np.percentile(dd, 99), Calmar=eq[-1] / yrs / dd.max())
    sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum(); mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
    du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
    r['monkey'] = 'geçti' if sysCC > max(mY.max(), mZ.max()) else f'✗ ({sysCC:.0f} vs {max(mY.max(), mZ.max()):.0f})'
    inc = np.diff(x30); nb = len(inc) // 48; path = np.cumsum(inc[:nb * 48].reshape(nb, 48)[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1)
    fd = (np.maximum.accumulate(np.c_[np.zeros(2000), path], 1) - np.c_[np.zeros(2000), path]).max(1); r['12ay_DDp95'] = np.percentile(fd, 95); r['12ay_DDp99'] = np.percentile(fd, 99)
    I = rng.integers(0, len(tp), (5000, len(tp))); r['MC_P5'] = np.percentile(np.where(rng.random((5000, len(tp))) > 0.05, tp[I] - 0.05, 0.0).sum(1), 5)
    r['stres'] = eq[-1] - (0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q).sum(); r['gecikme1s'] = run(1, *cf)[0][-1]; r['gecikme2s'] = run(2, *cf)[0][-1]
    Z = pd.Series(x30, index=i30[:len(x30)]).reindex(P30.index, method='ffill').dropna(); Pp = P30.reindex(Z.index)
    for nom in (200, 500):
        C_ = Pp.values + (Z.values - Z.values[0]) * nom / 500; ddc = np.maximum.accumulate(C_) - C_; s = pd.Series(C_, index=Z.index)
        r[f'port+{nom}_MaxDD'] = ddc.max(); r[f'port+{nom}_Kasım'] = s.iloc[-1] - s[:'2026-10-31 23:30'].iloc[-1]; r[f'port+{nom}_Ekim'] = s[:'2026-10-31 23:30'].iloc[-1] - s[:'2026-09-30 23:30'].iloc[-1]
    r['zecKasım'] = Z.iloc[-1] - Z[:'2026-10-31 23:30'].iloc[-1]; r['günlükKor'] = Z.resample('1D').last().diff().corr(Pp.resample('1D').last().diff())
    out.append(r); np.save(f'/home/claude/zecwork/eq_{nm}.npy', x30)
pp = P30['2023-05-08':]; ddp = (pp.cummax() - pp).max(); print('Portföy tek: MaxDD', round(ddp, 1), 'Ekim', round(pp[:'2026-10-31 23:30'].iloc[-1] - pp[:'2026-09-30 23:30'].iloc[-1], 1), 'Kasım', round(pp.iloc[-1] - pp[:'2026-10-31 23:30'].iloc[-1], 1))
pd.set_option('display.width', 250); print(pd.DataFrame(out).set_index('aday').T.round(2).to_string())
