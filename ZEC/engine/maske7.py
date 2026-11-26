exec(open('/home/claude/zecwork/maske2.py').read().split("rows = []")[0])
def mk(rule, lag=0):
    td = pd.Timedelta(rule); D = bars(rule); C = D.c; ax = adx(D); st = np.sign(C.ewm(span=10, adjust=False).mean() - C.ewm(span=30, adjust=False).mean()).values
    if lag: st = np.r_[np.zeros(lag), st[:-lag]]; ax = np.r_[np.zeros(lag), ax[:-lag]]
    return m1(st, D, td), m1(ax, D, td)
bn = B['net']; R = lambda S, A, z, c: met(comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S, A, 5, z, c), '')
def pr(t, m): print(f'{t:26s} {m["net"] - bn:+6.0f} ({(m["net"] / bn - 1) * 100:+.1f}%) yarılar {m["IS"] - B["IS"]:+5.0f}/{m["OOS"] - B["OOS"]:+5.0f} MaxDD {m["MaxDD"]:5.1f} DD99 {m["DD99"]:5.1f} Calmar {m["Calmar"]:.2f}')
S3, A3 = mk('3h')
print('--- 3 s EMA10/30, ADX komşuları (stop %5)')
for z in (36.0, 38.0, 40.0, 42.0, 45.0): pr(f'ADX≥{z:.0f}', R(S3, A3, z, 0.05))
print('--- stop komşuları (ADX≥40)')
for c in (0.03, 0.04, 0.06, 0.08): pr(f'stop %{c * 100:.0f}', R(S3, A3, 40.0, c))
print('--- zaman dilimi komşuları (ADX≥40)')
for rule in ('150min', '210min', '240min'): S_, A_ = mk(rule); pr(rule, R(S_, A_, 40.0, 0.05))
print('--- 1 bar ek gecikme')
S_, A_ = mk('3h', 1); pr('3 s +1 bar gecikme', R(S_, A_, 40.0, 0.05))
e = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S3, A3, 5, 40.0, 0.05); x = pd.Series(e - e0, ixh).resample('YE').last()
print('ek katkı yıllık:', dict(zip(x.index.year, np.diff(np.r_[0, x.values]).round(0))))
t23 = np.searchsorted(tms, 1675382400000); print(f'TV penceresi: V3.3 {e0[-1] - e0[t23]:.1f} → V3.4 {e[-1] - e[t23]:.1f}')
act = (rp == 0) & (S3 != 0) & (A3 >= 40); print(f'devrede olabilir %{act.mean() * 100:.1f}')
rng = np.random.default_rng(7); cut = np.r_[0, np.flatnonzero(np.diff(S3) != 0) + 1]; mk_ = []
for k in range(40):
    sg = rng.choice([-1.0, 1.0], len(cut)); Sm = S3.copy()
    for i, a in enumerate(cut): b = cut[i + 1] if i + 1 < len(cut) else len(S3); Sm[a:b] = np.abs(S3[a:b]) * sg[i]
    mk_.append(R(Sm, A3, 40.0, 0.05)['net'] - bn)
mk_ = np.array(mk_); print(f'maymun (rastgele yön, aynı zaman) medyan {np.median(mk_):+.0f}, P95 {np.percentile(mk_, 95):+.0f}, gerçek ≥ maymun oranı %{(mk_ < 677).mean() * 100:.0f}')
