# YÜRÜYEN SEÇİM: her blokta kurallar yalnız geçmiş veriden seçilir (bincount hızlı), sonraki bloğa uygulanır
exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
import importlib; importlib.reload(z03); pd.set_option('display.width', 320)
R0i = np.asarray(R0, np.int64)
def selF(thr, hz, segs, od):
    L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]
    for (t0, t1), (a0, a1) in segs:
        Rt = R0i[t0:t1]; ft = f[t0:t1]; fv = ~np.isnan(ft); Ra = R0i[a0:a1]
        for k in od:
            vt = V[k][t0:t1]; va = V[k][a0:a1]
            for side in (1, -1):
                m = vt == side; ca = np.bincount(Rt[m], minlength=96); mv = m & fv
                sm = np.bincount(Rt[mv], weights=ft[mv], minlength=96); cv = np.bincount(Rt[mv], minlength=96)
                ok = (ca >= 400) & (cv > 0) & (sm / np.maximum(cv, 1) * side >= thr)
                (L if side > 0 else S)[a0:a1] += (va == side) & ok[Ra]
    return L, S
CF = [(PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), (PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
      (PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), (PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
# doğrulama: ilk yarı seçimi orijinal select ile aynı mı
order = CF[1][0]; Lo, So = select(R0, 10, 120); Lf, Sf = selF(10, 120, [((0, mid), (0, n))], CF[1][0]); print('selF = select:', np.array_equal(Lo, Lf) and np.array_equal(So, Sf), flush=True)
vol = B.volume.values; vr = pd.Series(vol).rolling(10).mean().values / pd.Series(vol).rolling(1440).mean().shift(1).values; pmp = (spk & np.nan_to_num(vr >= 3)).astype(np.bool_)
def build(segs, pump=True):
    E = []
    for od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm in CF:
        L, S = selF(thr, hz, segs, od); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); E.append((aL, aS, eL, eS))
    if pump: z = np.zeros(n, np.bool_); E.append((pmp, z, pmp, z))
    return st(*E)
t22 = p('2022-01-01'); out = []
def run(nm, segs, pump=True):
    X = build(segs, pump); K_ = X[0].shape[0]
    eq, T = z03.simN7(o, h, l, c, a, *X, np.r_[TSa, 8.0][:K_], np.r_[THa, 480][:K_], spk, 60, 3, np.array([0, 0, 0, 0, 1][:K_], np.bool_))
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    w = eq[t23::30] - eq[t23]; ww = np.maximum.accumulate(w) - w; tt = np.arange(len(w))
    yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
    r = dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), net2022p=eq[-1] - eq[t22], TVpencere=w[-1], TVpenMaxDD=ww.max(),
             R2pencere=np.corrcoef(tt, w)[0, 1] ** 2, son12ay=eq[-1] - eq[max(0, n - 525600)], yıllar=' '.join(f'{v:.0f}' for v in np.diff(yb)))
    out.append(r); print(pd.DataFrame([r]).round(2).to_string(index=False, header=len(out) == 1), flush=True); return eq, T
run('V2.6 baz (ilk yarı seçim)', [((0, mid), (0, n))])
def segs_for(freq, look):
    bs = [p(d) for d in pd.date_range('2022-01-01', idx[-1], freq=freq)]; ends = bs[1:] + [n]; sg = [((0, t22 - 480), (0, t22))]   # 2021: seçim içi (rapor 2022+)
    for b0, b1 in zip(bs, ends): sg.append(((0 if look is None else max(0, b0 - look), b0 - 480), (b0, b1)))
    return sg
for nm, fq, lk in (('YS çapa 6 ay', '6MS', None), ('YS çapa 3 ay', '3MS', None), ('YS son 2 yıl 3 ay', '3MS', 1051200), ('YS son 1 yıl 3 ay', '3MS', 525600)):
    eq, T = run(nm, segs_for(fq, lk)); np.save(f'/home/claude/zecwork/eq_WF_{fq}_{lk}.npy', eq[::30]); np.save(f'/home/claude/zecwork/T_WF_{fq}_{lk}.npy', T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/YURUYEN_SECIM_1DK.csv', index=False)
