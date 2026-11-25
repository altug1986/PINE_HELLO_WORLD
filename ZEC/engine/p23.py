# 2023 SONRASI ÇEKİRDEK: kurallar yalnız 2023-01→2024-12 verisinden; test 2025-01→2026-11 (ileri) ve 2021–22 (geri)
exec(open('/home/claude/zecwork/ls1.py').read().split("out = []\ndef ad(")[0])
out = []
t23a = p('2023-01-01'); t25 = p('2025-01-01'); t25b = p('2025-07-01')
def selW(R, thr, hz, od, a0, a1):
    nc = int(R.max()) + 1; L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]; Rt = R[a0:a1]; ft = f[a0:a1]; fv = ~np.isnan(ft)
    for k in od:
        vt = V[k][a0:a1]; va = V[k]
        for side in (1, -1):
            m = vt == side; ca = np.bincount(Rt[m], minlength=nc); mv = m & fv
            sm = np.bincount(Rt[mv], weights=ft[mv], minlength=nc); cv = np.bincount(Rt[mv], minlength=nc)
            ok = (ca >= 400) & (cv > 0) & (sm / np.maximum(cv, 1) * side >= thr); (L if side > 0 else S)[:] += (va == side) & ok[R]
    return L, S
def runW(sc, a0, a1, pump=True):
    E = []; z = np.zeros(n, np.bool_)
    for od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm in CF:
        L, S = selW(R0i, thr * sc, hz, od, a0, a1 - 480); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); E.append((aL, aS, eL, eS))
    X = st(*([(z, z, z, z)] + E + [(pmp if pump else z, z, pmp if pump else z, z)]))
    return z03.simN8(o, h, l, c, a, *X, TS6, TH6, spk, 60, 3, byp, 0.10, False, F0, F0)
def ad2(nm, eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    def R2(a_, b_): y = eq[a_:b_:30] - eq[a_]; return np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2 * np.sign(y[-1])
    y = eq[t23a::30] - eq[t23a]; ddp = (np.maximum.accumulate(y) - y).max(); i300 = np.searchsorted(idx.values, w300); i90 = np.searchsorted(idx.values, w90)
    yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
    ex = (eq[p('2025-09-01')] - eq[t23a]) + (eq[-1] - eq[p('2026-01-01')])
    r = dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), net2023p=y[-1], MaxDD2023p=ddp, R2_2023p=R2(t23a, n), geri21_22=eq[t23a], seçim23_24=eq[t25] - eq[t23a],
             ileri25_26=eq[-1] - eq[t25], R2ileri=R2(t25, n), ralliHariç23p=ex, son300=eq[-1] - eq[i300], son90=eq[-1] - eq[i90], yıllar=' '.join(f'{v:.0f}' for v in np.diff(yb)))
    cn, ck, kp, kr, kg = portm(eq, w300); r.update({'çuk300$': cn, 'çuk300%kapa': kp}); cn, ck, kp, kr, kg = portm(eq, w90); r.update({'çuk90$': cn})
    out.append(r); print(pd.DataFrame([r]).round(2).to_string(index=False, header=len(out) == 1), flush=True); return r
eq, T = runR(R0i); ad2('V2.7 baz (2021–23 seçim)', eq, T)
for sc in (1.0, 0.6, 0.4, 0.25):
    eq, T = runW(sc, t23a, t25); ad2(f'2023–24 seçim eşik×{sc}', eq, T); np.save(f'/home/claude/zecwork/eq_P23_{sc}.npy', eq); np.save(f'/home/claude/zecwork/T_P23_{sc}.npy', T)
for sc in (0.6, 0.4):
    eq, T = runW(sc, t23a, t25, False); ad2(f'2023–24 seçim eşik×{sc} pompasız', eq, T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/CEKIRDEK_2023_1DK.csv', index=False)
