exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
import importlib; importlib.reload(z03); out = []; pd.set_option('display.width', 300)
def ad(nm, eq, T, own=None, keep=None):
    r = met(nm, eq, T); r['pompa$'] = T[T[:, 7] == own, 5].sum() if own is not None else np.nan; r['pompaN'] = (T[:, 7] == own).sum() if own is not None else 0; r['son12ay'] = eq[-1] - eq[max(0, n - 525600)]; out.append(r)
    if keep: np.save(f'/home/claude/zecwork/T_{keep}.npy', T); np.save(f'/home/claude/zecwork/eq_{keep}.npy', eq[::30])
    print(pd.DataFrame([r]).round(1).to_string(index=False, header=len(out) == 1), flush=True)
eq, T = z03.simN6(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60, 3); ad('V2.3 baz', eq, T)
z = np.zeros((1, n), np.bool_); vol = B.volume.values if 'volume' in B else None
vr = pd.Series(vol).rolling(10).mean().values / pd.Series(vol).rolling(1440).mean().shift(1).values if vol is not None else None
r10_ = np.nan_to_num(r10)
for nm, ev in (('%3', spk), ('%3 hacim≥3', spk & np.nan_to_num(vr >= 3)), ('%5', r10_ >= 5)):
    P = ev[None, :].astype(np.bool_)
    X = tuple(np.ascontiguousarray(np.vstack(x)) for x in ((AL, P), (AS, z), (EL, P), (ES, z)))
    for tsP, thP in ((16.0, 1440), (16.0, 480), (8.0, 480)):
        eq, T = z03.simN7(o, h, l, c, a, *X, np.r_[TSa, tsP], np.r_[THa, thP], spk, 60, 3, np.array([0, 0, 0, 0, 1], np.bool_))
        ad(f'POMPA-alt {nm} ts{tsP:.0f} H{thP}', eq, T, 4, f'PA_{nm}_{tsP:.0f}_{thP}'.replace('%', 'p').replace(' ', '').replace('≥', 'v'))
# iki yarı uzlaşma seçimi (tüm veri; OOS yok → yıl dağılımı ile bak)
def select_both(R, thr, hz):
    L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]
    for k in order:
        vv = V[k]
        for side in (1, -1):
            for val in np.unique(R):
                ok = True
                for sl in (slice(0, mid), slice(mid, n)):
                    m = (vv[sl] == side) & (R[sl] == val)
                    if m.sum() < 400 or not (np.nanmean(f[sl][m] * side) >= thr): ok = False; break
                if ok: (L if side > 0 else S)[(vv == side) & (R == val)] += 1
    return L, S
select = select_both; LS.clear()
E2 = [sig('B', PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), sig('A', PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
      sig('B', PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), sig('G', PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
A2 = st(*E2); eq, T = z03.simN6(o, h, l, c, a, *A2, TSa, THa, spk, 60, 3); ad('İKİ YARI uzlaşma', eq, T, None, 'BOTH')
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/POMPA_ALT_1DK.csv', index=False)
