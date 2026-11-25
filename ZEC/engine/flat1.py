exec(open('/home/claude/zecwork/tsvar.py').read().split("t23 = p('2023-02-06'); out = []")[0])
import importlib; importlib.reload(z03); t23 = p('2023-02-06'); out = []; rng = np.random.default_rng(4); fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq
def met(nm, eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; g = T[1:, 0] - T[:-1, 1]; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum(); d_ = np.diff(x30[::48])
    yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
    return dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), Sharpe=d_.mean() / d_.std() * np.sqrt(365), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], TVpencere=eq[-1] - eq[t23],
                aynıYön=int(((g <= 1) & (T[1:, 2] == T[:-1, 2])).sum()), dönüş=int(((g <= 1) & (T[1:, 2] != T[:-1, 2])).sum()), tekBar=int(((T[:, 1] - T[:, 0]) <= 1).sum()), maeMaks=T[:, 6].max() * 100,
                yıllar=' '.join(f'{v:.0f}' for v in np.diff(yb)))
TSa = np.array([28.0] * 4); THa = np.array([3840, 480, 960, 480])
eq, T = z03.simN3(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60); out.append(met('V2.2 öncelik/devralma', eq, T))
for ce in (1, 0):
    eq, T = z03.simN5(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60, ce); out.append(met(f'ÇARPIŞMA=FLAT {"(çıkışlı)" if ce else "(yalnız giriş)"}', eq, T))
    if ce: np.save('/home/claude/zecwork/T_FLAT.npy', T); np.save('/home/claude/zecwork/eq_FLAT.npy', eq[::30]); TF = T; eqF = eq
for th in ((960, 480, 960, 480), (1920, 960, 960, 960)):
    eq, T = z03.simN5(o, h, l, c, a, AL, AS, EL, ES, TSa, np.array(th), spk, 60, 1); out.append(met(f'FLAT TH {th}', eq, T))
pd.set_option('display.width', 260); print(pd.DataFrame(out).round(2).to_string(index=False))
T = TF; tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 / T[:, 3]
sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum(); mY = (q * (c[x_] - c[e_]) * rng.choice([-1.0, 1.0], (1000, len(tp))) - fee(c[e_], c[x_], q)).sum(1)
du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du; qZ = 500 / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
print('FLAT monkey yön', round(sysCC), '>', round(mY.max()), '· zaman', round(mZ.max()))
