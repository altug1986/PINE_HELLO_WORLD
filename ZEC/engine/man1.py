# ZARAR TAVANI + TERS MANEVRA (V2.6 tabanı: 4 motor + POMPA)
exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
import importlib; importlib.reload(z03); pd.set_option('display.width', 320); out = []
vol = B.volume.values; vr = pd.Series(vol).rolling(10).mean().values / pd.Series(vol).rolling(1440).mean().shift(1).values; pmp = (spk & np.nan_to_num(vr >= 3)).astype(np.bool_)
z = np.zeros((1, n), np.bool_); P = pmp[None, :]
X = tuple(np.ascontiguousarray(np.vstack(x)) for x in ((z, AL, P), (z, AS, z), (z, EL, P), (z, ES, z)))
TS6 = np.r_[1e6, TSa, 8.0]; TH6 = lambda m: np.r_[m, THa, 480]; byp = np.array([0, 0, 0, 0, 0, 1], np.bool_)
s1 = lambda x: np.r_[False, np.asarray(x, bool)[:-1]]
ema = lambda x, k: pd.Series(x).ewm(span=k, adjust=False).mean().values
COND = {'hep': (np.ones(n, bool), np.ones(n, bool))}
for k in (240, 1440): e_ = ema(c, k); COND[f'trend{k // 60}s'] = (s1(c > e_), s1(c < e_))
COND['oy'] = (s1(AL.any(0)), s1(AS.any(0)))
r60 = np.nan_to_num(c / np.r_[np.full(60, np.nan), c[:-60]] - 1); COND['mom1s'] = (s1(r60 > 0), s1(r60 < 0))
def ad(nm, eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum(); m0 = T[:, 7] == 0
    r = dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], TVpencere=eq[-1] - eq[t23],
             maeMaks=T[:, 6].max() * 100, maeP99=np.percentile(T[:, 6], 99) * 100, enKötüİşlem=T[:, 5].min() / 5, manevraN=m0.sum(), manevra=T[m0, 5].sum())
    out.append(r); print(pd.DataFrame([r]).round(2).to_string(index=False, header=len(out) == 1), flush=True)
    return r
F0 = np.zeros(n, bool)
eq, T = z03.simN8(o, h, l, c, a, *X, TS6, TH6(480), spk, 60, 3, byp, 0.0, False, F0, F0); ad('V2.6 baz', eq, T)
for cap in (0.03, 0.05, 0.07, 0.10):
    eq, T = z03.simN8(o, h, l, c, a, *X, TS6, TH6(480), spk, 60, 3, byp, cap, False, F0, F0); ad(f'tavan %{cap * 100:.0f}', eq, T)
    for cn, (fl, fs) in COND.items():
        for mth in (240, 960):
            eq, T = z03.simN8(o, h, l, c, a, *X, TS6, TH6(mth), spk, 60, 3, byp, cap, True, fl, fs); ad(f'tavan %{cap * 100:.0f} +manevra {cn} {mth}', eq, T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/TAVAN_MANEVRA_1DK.csv', index=False)
