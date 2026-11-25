exec(open('/home/claude/zecwork/orch4.py').read().split("rows = []; import gc as _gc")[0])
st = lambda *E: tuple(np.ascontiguousarray(np.stack([e[j] for e in E])) for j in range(4))
rng = np.random.default_rng(21); fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq; yrs = n / 525600
def stack(lg):
    global R0
    R0 = RL[lg]; LS.clear()
    E = [sig('B', PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), sig('A', PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
         sig('B', PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), sig('G', PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
    AL, AS, EL, ES = st(*E); return z03.simN(o, h, l, c, a, AL, AS, EL, ES, np.array([28.0, 28.0, 28.0, 28.0]), np.array([3840, 480, 960, 480]), spk, 60)
eq, T = stack(0); tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]; x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; dly = np.diff(x30[::48])
sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum(); mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
inc = np.diff(x30); nb = len(inc) // 48; path = np.cumsum(inc[:nb * 48].reshape(nb, 48)[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1)
fd = (np.maximum.accumulate(np.c_[np.zeros(2000), path], 1) - np.c_[np.zeros(2000), path]).max(1); I = rng.integers(0, len(tp), (5000, len(tp)))
gp = tp[tp > 0].sum(); gl = -tp[tp < 0].sum(); yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
r = dict(net=eq[-1], n=len(T), PF=gp / gl, MaxDD=dd.max(), DDp95=np.percentile(dd, 95), DDp99=np.percentile(dd, 99), Sharpe=dly.mean() / dly.std() * np.sqrt(365), Calmar=eq[-1] / yrs / dd.max(),
         monkeyYön=f'{sysCC:.0f} > {mY.max():.0f}', monkeyZaman=f'{sysCC:.0f} > {mZ.max():.0f}', MC_P5=np.percentile(np.where(rng.random((5000, len(tp))) > 0.05, tp[I] - 0.05, 0).sum(1), 5),
         ileri12_P95=np.percentile(fd, 95), ileri12_P99=np.percentile(fd, 99), stres=eq[-1] - (0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q).sum(), yıllar=np.round(np.diff(yb)).tolist(),
         son2ay=eq[-1] - eq[p('2026-09-24')], Kasım=eq[-1] - eq[p('2026-11-01')], short=T[d_ < 0, 5].sum(), maeP95=np.percentile(T[:, 6], 95) * 100)
r['gecikme1s'] = stack(1)[0][-1]; r['gecikme2s'] = stack(2)[0][-1]
for k, v in r.items(): print(f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}')
np.save('/home/claude/zecwork/eq_ORK4.npy', x30); np.save('/home/claude/zecwork/T_ORK4.npy', T)
