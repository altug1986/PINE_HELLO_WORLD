# V2.3 + POMPA motoru (spk barında long; short'taysa çevir) + yakın yarıda kural seçimi
exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
import importlib; importlib.reload(z03); out = []
def ad(nm, eq, T, keep=None):
    r = met(nm, eq, T); r['pompa$'] = T[T[:, 7] == 0, 5].sum() if keep else np.nan; r['son12ay'] = eq[-1] - eq[max(0, n - 525600)]; out.append(r)
    if keep: np.save(f'/home/claude/zecwork/T_{keep}.npy', T); np.save(f'/home/claude/zecwork/eq_{keep}.npy', eq[::30])
    print(pd.DataFrame([r]).round(1).to_string(index=False, header=len(out) == 1), flush=True)
pd.set_option('display.width', 300)
eq, T = z03.simN6(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60, 3); ad('V2.3 baz', eq, T)
def withP(AL, AS, EL, ES): z = np.zeros((1, n), np.bool_); P = spk[None, :]; return tuple(np.ascontiguousarray(np.vstack(x)) for x in ((P, AL), (z, AS), (P, EL), (z, ES)))
P4 = withP(AL, AS, EL, ES)
for tsP, thP in ((16.0, 1440), (8.0, 1440), (12.0, 1440), (24.0, 1440), (16.0, 720), (16.0, 2880)):
    eq, T = z03.simN7(o, h, l, c, a, *P4, np.r_[tsP, TSa], np.r_[thP, THa], spk, 60, 3, 1); ad(f'+POMPA ts{tsP:.0f} H{thP}', eq, T, 'POMPA' if (tsP, thP) == (16.0, 1440) else None)
# yakın yarıda seçim (seçim yalnız ikinci yarı verisi; ilk yarı = test)
def select_rev(R, thr, hz):
    L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]; fi = f[mid:]; Ri = R[mid:]
    for k in order:
        vv = V[k]; vi = vv[mid:]
        for side in (1, -1):
            for val in np.unique(R):
                m = (vi == side) & (Ri == val)
                if m.sum() < 400: continue
                if np.nanmean(fi[m] * side) >= thr: (L if side > 0 else S)[(vv == side) & (R == val)] += 1
    return L, S
select = select_rev; LS.clear()
E2 = [sig('B', PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), sig('A', PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
      sig('B', PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), sig('G', PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
ALr, ASr, ELr, ESr = st(*E2)
eq, T = z03.simN6(o, h, l, c, a, ALr, ASr, ELr, ESr, TSa, THa, spk, 60, 3); ad('YAKIN seçim', eq, T, 'REV')
Pr = withP(ALr, ASr, ELr, ESr)
eq, T = z03.simN7(o, h, l, c, a, *Pr, np.r_[16.0, TSa], np.r_[1440, THa], spk, 60, 3, 1); ad('YAKIN seçim +POMPA', eq, T, 'REVP')
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/POMPA_YAKIN_1DK.csv', index=False)
